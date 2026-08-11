"""
preprocessing.py
=================
Turns the raw per-timestep simulation log into ML-ready tables:

1. A "fast-tag" feature frame (soft sensor + fault classification input):
   easy/cheap/frequent measurements only -- this mirrors what's actually
   available in real time on a real plant (the analyzer reading is NOT
   used as a feature, since predicting it is the whole point of a soft
   sensor).
2. Forward-filled analyzer readings to create a continuous regression
   target for the soft sensor (matches how soft sensors are actually
   trained in practice: interpolate/hold the slow lab value between
   samples).
3. Rolling windows for the sequence models (fault classifier, anomaly
   detector).
4. An episode-level train/val/test split (never split *within* an
   episode -- that would leak information across the split boundary).
"""

import numpy as np
import pandas as pd

FAST_FEATURES = ["T_meas", "Tc_meas", "Tc_cmd", "q_meas", "level_meas", "T_setpoint", "q_setpoint", "Caf_nom"]


def add_soft_sensor_target(df):
    """Forward-fill the intermittent analyzer reading within each episode
    to build a continuous regression target column `Ca_target`."""
    df = df.sort_values(["episode_id", "t_min"]).copy()
    df["Ca_target"] = df.groupby("episode_id")["Ca_analyzer"].ffill()
    # rows before the very first analyzer reading in an episode have no
    # valid target yet -- drop those at train time.
    return df


def episode_split(df, train_frac, val_frac, test_frac, seed):
    ids = df["episode_id"].unique().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    if n == 0:
        return set(), set(), set()
    # Guarantee at least 1 episode in val/test (when there's enough data to
    # spare) -- plain int(n*frac) rounds to 0 for small n and silently
    # produces an empty split, which crashes/skews downstream training.
    n_val = max(1, round(n * val_frac)) if n >= 3 else 0
    n_test = max(1, round(n * test_frac)) if n >= 3 else 0
    n_val = min(n_val, n - 1) if n > 1 else 0
    n_test = min(n_test, max(0, n - n_val - 1))
    n_train = n - n_val - n_test
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])
    return train_ids, val_ids, test_ids


def make_soft_sensor_table(df):
    d = add_soft_sensor_target(df)
    d = d.dropna(subset=["Ca_target"])
    X = d[FAST_FEATURES].copy()
    y = d["Ca_target"].copy()
    meta = d[["episode_id", "t_min", "fault_name", "fault_active"]].copy()
    return X, y, meta


def make_windows(df, window_size, stride, feature_cols, label_col="fault_name"):
    """Slice each episode into overlapping windows of `window_size` rows.
    Returns X of shape (n_windows, window_size, n_features), labels (majority
    label in window), and episode ids (for grouping/splitting)."""
    X_list, y_list, ep_list, t_list = [], [], [], []
    for ep_id, g in df.groupby("episode_id"):
        g = g.sort_values("t_min")
        arr = g[feature_cols].to_numpy(dtype=np.float32)
        labels = g[label_col].to_numpy()
        active = g["fault_active"].to_numpy()
        t_vals = g["t_min"].to_numpy()
        n = len(g)
        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            window = arr[start:end]
            if np.isnan(window).any():
                continue
            # Label a window as the fault only if the fault is active for
            # at least half the window (avoids ambiguous transition windows
            # dominating training).
            active_frac = active[start:end].mean()
            lbl = labels[end - 1] if active_frac >= 0.5 else "none"
            X_list.append(window)
            y_list.append(lbl)
            ep_list.append(ep_id)
            t_list.append(t_vals[end - 1])
    X = np.stack(X_list) if X_list else np.empty((0, window_size, len(feature_cols)))
    y = np.array(y_list)
    ep = np.array(ep_list)
    t = np.array(t_list)
    return X, y, ep, t


def compute_rul_table(df, deviation_threshold):
    """For incipient (gradual) faults, compute a Remaining-Useful-Life
    label at each timestep: minutes until the controller's cooling
    command saturates (pid_saturated flips to 1 and stays 1), which is
    the physically meaningful 'can no longer compensate' failure event
    for both fouling and catalyst deactivation in this plant.

    Rows from episodes that never reach saturation within the episode are
    excluded (right-censored -- no ground truth RUL available); this is
    standard practice for RUL datasets rather than inventing a label.
    """
    rows = []
    incipient_faults = {"fouling", "catalyst_deactivation", "sensor_bias_T", "sensor_bias_Ca"}
    for ep_id, g in df.groupby("episode_id"):
        g = g.sort_values("t_min").reset_index(drop=True)
        fname = g["fault_name"].iloc[-1] if g["fault_active"].any() else "none"
        if fname not in incipient_faults:
            continue
        sat = g["pid_saturated"].to_numpy()
        # first index where saturation begins and stays on for a sustained stretch
        breach_idx = None
        for i in range(len(sat) - 30):
            if sat[i:i + 30].mean() > 0.9:
                breach_idx = i
                break
        if breach_idx is None:
            continue  # censored: never actually broke within episode horizon
        breach_t = g["t_min"].iloc[breach_idx]
        sub = g.iloc[:breach_idx].copy()
        sub["RUL_min"] = breach_t - sub["t_min"]
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=list(df.columns) + ["RUL_min"])
    return pd.concat(rows, ignore_index=True)
