"""
data_generator.py
==================
Runs many randomized plant episodes (varying operating setpoints, fault
type, onset time, severity, and random seed) through the CSTRSimulator and
assembles them into a single labeled dataset used by every downstream ML
model (soft sensor, fault classifier, anomaly detector, RUL predictor).

Each episode gets a unique episode_id. Faults are sampled according to the
probabilities in config['faults']['types'].
"""

import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from simulator import CSTRSimulator, FaultState


def sample_fault(rng, faults_cfg):
    names = [f["name"] for f in faults_cfg["types"]]
    probs = np.array([f["probability"] for f in faults_cfg["types"]], dtype=float)
    probs = probs / probs.sum()
    choice = rng.choice(len(names), p=probs)
    spec = faults_cfg["types"][choice]

    if spec["name"] == "none":
        return FaultState("none", None, onset_min=0.0, severity=0.0)

    sev_lo, sev_hi = spec["severity_range"]
    severity = rng.uniform(sev_lo, sev_hi)

    if spec["kind"] == "abrupt":
        onset_min = rng.uniform(300, 2200)  # somewhere well into the episode
        return FaultState(spec["name"], "abrupt", onset_min=onset_min, severity=severity)
    else:  # incipient
        ramp_lo, ramp_hi = spec["ramp_hours_range"]
        ramp_min = rng.uniform(ramp_lo, ramp_hi) * 60.0
        onset_min = rng.uniform(200, 1400)
        return FaultState(spec["name"], "incipient", onset_min=onset_min, severity=severity, ramp_min=ramp_min)


def _run_one_episode(args):
    cfg, episode_id, seed = args
    rng = np.random.default_rng(seed)
    sim = CSTRSimulator(cfg)
    fault = sample_fault(rng, cfg["faults"])
    df = sim.run_episode(rng, fault)
    df["episode_id"] = episode_id
    df["seed"] = seed
    return df


def generate_dataset(cfg, n_episodes=None, out_path=None, deadline=None, log=print, max_workers=None):
    """Generate n_episodes worth of simulated plant data.

    deadline: optional unix timestamp (time.time()-style). If set, stops
    launching new episodes once passed (so this respects the overnight
    time budget instead of running unboundedly).
    """
    n_episodes = n_episodes or cfg["dataset"]["n_episodes"]
    base_seed = cfg["dataset"]["random_seed"]
    max_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)

    jobs = [(cfg, i, base_seed + i) for i in range(n_episodes)]
    frames = []
    t0 = time.time()

    log(f"[data_generator] generating up to {n_episodes} episodes with {max_workers} workers...")

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_run_one_episode, job): job[1] for job in jobs}
            done = 0
            for fut in as_completed(futures):
                if deadline is not None and time.time() > deadline:
                    log("[data_generator] time budget reached, stopping early "
                        f"with {done}/{n_episodes} episodes.")
                    for f in futures:
                        f.cancel()
                    break
                df = fut.result()
                frames.append(df)
                done += 1
                if done % max(1, n_episodes // 20) == 0:
                    log(f"[data_generator] {done}/{n_episodes} episodes "
                        f"({time.time()-t0:.0f}s elapsed)")
    except Exception as e:
        # Multiprocessing can fail in some sandboxed / restricted environments.
        # Fall back to sequential execution rather than losing the whole run.
        log(f"[data_generator] parallel execution failed ({e!r}); falling back to sequential.")
        frames = []
        for job in jobs:
            if deadline is not None and time.time() > deadline:
                log(f"[data_generator] time budget reached at {len(frames)}/{n_episodes} episodes.")
                break
            frames.append(_run_one_episode(job))
            if len(frames) % max(1, n_episodes // 20) == 0:
                log(f"[data_generator] {len(frames)}/{n_episodes} episodes "
                    f"({time.time()-t0:.0f}s elapsed)")

    if not frames:
        raise RuntimeError("No episodes were generated -- check time budget / simulator.")

    full = pd.concat(frames, ignore_index=True)
    log(f"[data_generator] done: {full['episode_id'].nunique()} episodes, "
        f"{len(full)} rows, {time.time()-t0:.0f}s total.")

    if out_path:
        full.to_parquet(out_path, index=False)
        log(f"[data_generator] saved to {out_path}")

    return full
