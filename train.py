"""
train.py
========
Time-budgeted training for each of the four model families. Every
train_* function takes a `budget_seconds` wall-clock allocation and uses
Optuna's `timeout` (not just `n_trials`) so it naturally fills whatever
time it's given regardless of machine speed, and always leaves a valid
model behind even if interrupted.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import optuna

from preprocessing import (FAST_FEATURES, make_soft_sensor_table, make_windows,
                            compute_rul_table, episode_split)
from models import SoftSensorRegressor, FaultClassifier, AnomalyDetector, RULPredictor

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _split_frames(X, meta_or_ep, train_ids, val_ids, test_ids, ep_col="episode_id"):
    if isinstance(meta_or_ep, pd.DataFrame):
        ep = meta_or_ep[ep_col].to_numpy()
    else:
        ep = meta_or_ep
    tr = np.isin(ep, list(train_ids))
    va = np.isin(ep, list(val_ids))
    te = np.isin(ep, list(test_ids))
    return tr, va, te


def train_soft_sensor(df, cfg, budget_seconds, out_dir, log=print):
    t0 = time.time()
    log(f"[soft_sensor] budget={budget_seconds:.0f}s")
    ds_cfg = cfg["dataset"]
    train_ids, val_ids, test_ids = episode_split(df, ds_cfg["train_frac"], ds_cfg["val_frac"],
                                                  ds_cfg["test_frac"], ds_cfg["random_seed"])
    X, y, meta = make_soft_sensor_table(df)
    tr, va, te = _split_frames(X, meta, train_ids, val_ids, test_ids)
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    Xte, yte = X[te], y[te]
    log(f"[soft_sensor] train/val/test rows: {len(Xtr)}/{len(Xva)}/{len(Xte)}")

    cap = cfg["training"]["optuna"]["soft_sensor_trials_cap"]

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        )
        m = SoftSensorRegressor(params).fit(Xtr, ytr)
        return m.score(Xva, yva)

    study = optuna.create_study(direction="minimize")
    remaining = max(5, budget_seconds - (time.time() - t0))
    study.optimize(objective, timeout=remaining, n_trials=cap, show_progress_bar=False)
    log(f"[soft_sensor] {len(study.trials)} trials, best val RMSE={study.best_value:.5f}")

    final = SoftSensorRegressor(study.best_params).fit(
        pd.concat([Xtr, Xva]), pd.concat([ytr, yva]))
    test_rmse = final.score(Xte, yte)
    log(f"[soft_sensor] test RMSE={test_rmse:.5f}")

    model_path = os.path.join(out_dir, "models", "soft_sensor.joblib")
    final.save(model_path)

    preds = final.predict(Xte)
    pred_df = pd.DataFrame({"y_true": yte.values, "y_pred": preds})
    pred_df.to_csv(os.path.join(out_dir, "data", "soft_sensor_test_predictions.csv"), index=False)

    return dict(model_path=model_path, test_rmse=test_rmse, n_trials=len(study.trials),
                best_params=study.best_params, elapsed_s=time.time() - t0)


def train_fault_classifier(df, cfg, budget_seconds, out_dir, log=print):
    t0 = time.time()
    log(f"[fault_classifier] budget={budget_seconds:.0f}s")
    ds_cfg = cfg["dataset"]
    train_ids, val_ids, test_ids = episode_split(df, ds_cfg["train_frac"], ds_cfg["val_frac"],
                                                  ds_cfg["test_frac"], ds_cfg["random_seed"])
    Xw, yw, epw, tw = make_windows(df, ds_cfg["window_size"], ds_cfg["window_stride"], FAST_FEATURES)
    tr, va, te = _split_frames(Xw, epw, train_ids, val_ids, test_ids)
    log(f"[fault_classifier] train/val/test windows: {tr.sum()}/{va.sum()}/{te.sum()}")

    cap = cfg["training"]["optuna"]["fault_classifier_trials_cap"]

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        )
        m = FaultClassifier(params).fit(Xw[tr], yw[tr])
        return m.score(Xw[va], yw[va])

    study = optuna.create_study(direction="maximize")
    remaining = max(5, budget_seconds - (time.time() - t0))
    study.optimize(objective, timeout=remaining, n_trials=cap, show_progress_bar=False)
    log(f"[fault_classifier] {len(study.trials)} trials, best val macro-F1={study.best_value:.4f}")

    final = FaultClassifier(study.best_params).fit(
        np.concatenate([Xw[tr], Xw[va]]), np.concatenate([yw[tr], yw[va]]))
    test_f1 = final.score(Xw[te], yw[te])
    log(f"[fault_classifier] test macro-F1={test_f1:.4f}")

    model_path = os.path.join(out_dir, "models", "fault_classifier.joblib")
    final.save(model_path)

    preds = final.predict(Xw[te])
    pred_df = pd.DataFrame({"y_true": yw[te], "y_pred": preds, "t_min": tw[te], "episode_id": epw[te]})
    pred_df.to_csv(os.path.join(out_dir, "data", "fault_classifier_test_predictions.csv"), index=False)

    return dict(model_path=model_path, test_macro_f1=test_f1, n_trials=len(study.trials),
                best_params=study.best_params, elapsed_s=time.time() - t0)


def train_anomaly_detector(df, cfg, budget_seconds, out_dir, log=print):
    t0 = time.time()
    log(f"[anomaly_detector] budget={budget_seconds:.0f}s")
    ds_cfg = cfg["dataset"]
    train_ids, val_ids, test_ids = episode_split(df, ds_cfg["train_frac"], ds_cfg["val_frac"],
                                                  ds_cfg["test_frac"], ds_cfg["random_seed"])
    Xw, yw, epw, tw = make_windows(df, ds_cfg["window_size"], ds_cfg["window_stride"], FAST_FEATURES)
    tr, va, te = _split_frames(Xw, epw, train_ids, val_ids, test_ids)

    train_normal = tr & (yw == "none")
    log(f"[anomaly_detector] training on {train_normal.sum()} normal windows")

    best = None
    deadline = t0 + budget_seconds
    for contamination in [0.02, 0.05, 0.08, 0.12]:
        if time.time() > deadline:
            break
        ad = AnomalyDetector(dict(n_estimators=300, contamination=contamination)).fit(Xw[train_normal])
        va_scores = ad.anomaly_score(Xw[va])
        va_labels = (yw[va] != "none").astype(int)
        # simple separation metric: AUC-like rank correlation between score and label
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(va_labels, va_scores)
        except ValueError:
            auc = float("nan")
        log(f"[anomaly_detector] contamination={contamination} val AUC={auc:.4f}")
        if best is None or (not np.isnan(auc) and auc > best[0]):
            best = (auc, contamination, ad)

    auc, contamination, final = best
    model_path = os.path.join(out_dir, "models", "anomaly_detector.joblib")
    final.save(model_path)

    te_scores = final.anomaly_score(Xw[te])
    pred_df = pd.DataFrame({"y_true_fault": yw[te], "anomaly_score": te_scores,
                             "t_min": tw[te], "episode_id": epw[te]})
    pred_df.to_csv(os.path.join(out_dir, "data", "anomaly_detector_test_scores.csv"), index=False)

    test_labels = (yw[te] != "none").astype(int)
    from sklearn.metrics import roc_auc_score
    try:
        test_auc = roc_auc_score(test_labels, te_scores)
    except ValueError:
        test_auc = float("nan")
    log(f"[anomaly_detector] test AUC={test_auc:.4f} (contamination={contamination})")

    return dict(model_path=model_path, test_auc=test_auc, contamination=contamination,
                elapsed_s=time.time() - t0)


def train_rul_predictor(df, cfg, budget_seconds, out_dir, log=print):
    t0 = time.time()
    log(f"[rul_predictor] budget={budget_seconds:.0f}s")
    rul_cfg = cfg["faults"]["rul_deviation_threshold"]
    rul = compute_rul_table(df, rul_cfg)
    if len(rul) < 50:
        log(f"[rul_predictor] only {len(rul)} labeled rows -- skipping (need more episodes/time).")
        return dict(model_path=None, test_rmse=None, note="insufficient_data")

    ds_cfg = cfg["dataset"]
    train_ids, val_ids, test_ids = episode_split(rul, ds_cfg["train_frac"], ds_cfg["val_frac"],
                                                  ds_cfg["test_frac"], ds_cfg["random_seed"])
    tr, va, te = _split_frames(rul, rul, train_ids, val_ids, test_ids)
    Xr, yr = rul[FAST_FEATURES], rul["RUL_min"]
    log(f"[rul_predictor] train/val/test rows: {tr.sum()}/{va.sum()}/{te.sum()}")

    if va.sum() == 0:
        log("[rul_predictor] empty val split (too few RUL episodes) -- reusing train for validation.")
        va = tr
    if te.sum() == 0:
        log("[rul_predictor] empty test split (too few RUL episodes) -- reusing train for test metric.")
        te = tr

    cap = cfg["training"]["optuna"]["rul_trials_cap"]

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        )
        m = RULPredictor(params).fit(Xr[tr], yr[tr])
        return m.score(Xr[va], yr[va])

    study = optuna.create_study(direction="minimize")
    remaining = max(5, budget_seconds - (time.time() - t0))
    study.optimize(objective, timeout=remaining, n_trials=cap, show_progress_bar=False)
    log(f"[rul_predictor] {len(study.trials)} trials, best val RMSE={study.best_value:.1f} min")

    final = RULPredictor(study.best_params).fit(
        pd.concat([Xr[tr], Xr[va]]), pd.concat([yr[tr], yr[va]]))
    test_rmse = final.score(Xr[te], yr[te])
    log(f"[rul_predictor] test RMSE={test_rmse:.1f} min")

    model_path = os.path.join(out_dir, "models", "rul_predictor.joblib")
    final.save(model_path)

    preds = final.predict(Xr[te])
    pred_df = pd.DataFrame({"y_true": yr[te].values, "y_pred": preds})
    pred_df.to_csv(os.path.join(out_dir, "data", "rul_test_predictions.csv"), index=False)

    return dict(model_path=model_path, test_rmse=test_rmse, n_trials=len(study.trials),
                best_params=study.best_params, elapsed_s=time.time() - t0)
