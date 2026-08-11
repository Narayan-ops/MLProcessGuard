"""
run.py
======
The single entry point for an unattended training run:

    python run.py --hours 5

Loads config.yaml, generates the simulated dataset, trains all four model
families (soft sensor, fault classifier, anomaly detector, RUL predictor)
each within its configured wall-clock share of --hours, runs evaluation,
and writes a final report. Designed to survive a machine going to sleep
overnight being run under `nohup` / `tmux` and to leave a valid, useful
set of models + report behind even if interrupted partway through.

Typical use:

    nohup python run.py --hours 5 > run_output/console.log 2>&1 &

Every phase is wrapped in try/except so one phase failing does not take
down the rest of the run, and the report is rewritten after every phase.
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

import yaml
import pandas as pd

import data_generator
import train
import evaluate

DEFAULT_OUT = "run_output"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_logger(out_dir):
    log_path = os.path.join(out_dir, "logs", "run.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = open(log_path, "a", buffering=1)

    def log(msg):
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
        print(line, flush=True)
        fh.write(line + "\n")

    return log


def ensure_dirs(out_dir):
    for sub in ("data", "models", "plots", "logs"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)


def write_report(out_dir, metrics, total_hours, log):
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    lines = [
        "# MLProcessGuard \u2014 Training Report",
        "",
        f"Total wall-clock budget: {total_hours:.2f} hours",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Dataset",
    ]
    dg = metrics.get("data_generation")
    if dg:
        lines += [
            f"- Episodes simulated: {dg.get('n_episodes', 'n/a')}",
            f"- Total rows: {dg.get('n_rows', 'n/a')}",
            f"- Generation time: {dg.get('elapsed_s', 0)/60:.1f} min",
        ]
    else:
        lines.append("- (not completed)")

    lines += ["", "## 1. Soft sensor (predicts lab Ca from fast tags)"]
    ss = metrics.get("soft_sensor")
    if ss and ss.get("model_path"):
        lines += [
            f"- Optuna trials: {ss['n_trials']}",
            f"- Test RMSE: {ss['test_rmse']:.5f} mol/L",
            "- See `plots/soft_sensor.png`",
        ]
    else:
        lines.append("- (phase did not complete)")

    lines += ["", "## 2. Fault classifier (XGBoost + Optuna)"]
    fc = metrics.get("fault_classifier")
    if fc and fc.get("model_path"):
        lines += [
            f"- Optuna trials: {fc['n_trials']}",
            f"- Test macro-F1: {fc['test_macro_f1']:.4f}",
            "- See `plots/fault_classifier_confusion.png`",
        ]
    else:
        lines.append("- (phase did not complete)")

    lines += ["", "## 3. Anomaly detector (Isolation Forest, trained on normal data only)"]
    ad = metrics.get("anomaly_detector")
    if ad and ad.get("model_path"):
        lines += [
            f"- Best contamination: {ad['contamination']}",
            f"- Test ROC-AUC: {ad['test_auc']:.4f}",
            "- See `plots/anomaly_detector.png`",
        ]
    else:
        lines.append("- (phase did not complete)")

    lines += ["", "## 4. RUL predictor (time-to-saturation for incipient faults)"]
    rul = metrics.get("rul_predictor")
    if rul and rul.get("model_path"):
        lines += [
            f"- Optuna trials: {rul['n_trials']}",
            f"- Test RMSE: {rul['test_rmse']:.1f} minutes",
            "- See `plots/rul_predictor.png`",
        ]
    elif rul and rul.get("note") == "insufficient_data":
        lines.append("- Skipped: not enough episodes reached saturation within the "
                      "simulated horizon to build labeled RUL examples. Increase "
                      "`dataset.n_episodes` or `simulator.episode_hours` and re-run.")
    else:
        lines.append("- (phase did not complete)")

    lines += ["", "## Next steps", "Run `python app/backend/main.py` (or deploy it) "
              "to serve the trained models through the live dashboard."]

    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))
    log(f"Report written to {out_dir}/report.md and {out_dir}/metrics.json")


def main():
    parser = argparse.ArgumentParser(description="MLProcessGuard overnight training run")
    parser.add_argument("--hours", type=float, default=5.0, help="total wall-clock budget, in hours")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--episodes", type=int, default=None,
                         help="override dataset.n_episodes (useful for a quick smoke test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg["dataset"]["n_episodes"] = args.episodes

    out_dir = args.out
    ensure_dirs(out_dir)
    log = make_logger(out_dir)

    total_seconds = args.hours * 3600
    alloc = cfg["training"]["time_allocation"]
    frac_sum = sum(alloc.values())
    if frac_sum > 1.0 + 1e-6:
        log(f"WARNING: time_allocation fractions sum to {frac_sum:.3f} > 1.0; "
            f"phases will be scaled down proportionally.")
        alloc = {k: v / frac_sum for k, v in alloc.items()}

    run_start = time.time()
    log(f"###### MLProcessGuard run starting | budget={args.hours}h | "
        f"episodes={cfg['dataset']['n_episodes']} ######")

    metrics = {}

    # ---------------- Phase 0: data generation ----------------
    parquet_path = os.path.join(out_dir, "data", "episodes.parquet")
    df = None
    try:
        t0 = time.time()
        if os.path.exists(parquet_path):
            log(f"Found existing dataset at {parquet_path}, loading it instead of regenerating.")
            df = pd.read_parquet(parquet_path)
        else:
            deadline = run_start + alloc["data_generation"] * total_seconds
            df = data_generator.generate_dataset(cfg, out_path=parquet_path, deadline=deadline, log=log)
        metrics["data_generation"] = dict(
            n_episodes=int(df["episode_id"].nunique()), n_rows=int(len(df)),
            elapsed_s=time.time() - t0,
        )
    except Exception as e:
        log(f"FATAL: data generation failed: {e}\n{traceback.format_exc()}")
        write_report(out_dir, metrics, (time.time() - run_start) / 3600, log)
        sys.exit(1)

    write_report(out_dir, metrics, (time.time() - run_start) / 3600, log)

    # ---------------- Phases 1-4: model training ----------------
    phase_specs = [
        ("soft_sensor", train.train_soft_sensor),
        ("fault_classifier", train.train_fault_classifier),
        ("anomaly_detector", train.train_anomaly_detector),
        ("rul_predictor", train.train_rul_predictor),
    ]
    for key, fn in phase_specs:
        budget = alloc.get(key, 0.0) * total_seconds
        log(f"=== Starting phase '{key}' | budget={budget/60:.1f} min ===")
        try:
            result = fn(df, cfg, budget, out_dir, log=log)
            metrics[key] = result
            log(f"=== Finished phase '{key}' ===")
        except Exception as e:
            log(f"Phase '{key}' crashed: {e}\n{traceback.format_exc()}")
            metrics[key] = {"error": str(e)}
        write_report(out_dir, metrics, (time.time() - run_start) / 3600, log)

    # ---------------- Phase 5: evaluation / plots ----------------
    try:
        evaluate.run_all_evaluation(cfg, out_dir, raw_df=df, log=log)
    except Exception as e:
        log(f"Evaluation phase crashed: {e}\n{traceback.format_exc()}")

    total_elapsed_h = (time.time() - run_start) / 3600
    write_report(out_dir, metrics, total_elapsed_h, log)
    log(f"###### MLProcessGuard run COMPLETE | actual elapsed={total_elapsed_h:.2f}h ######")


if __name__ == "__main__":
    main()
