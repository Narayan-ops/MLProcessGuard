"""
evaluate.py
===========
Generates diagnostic plots (saved as PNGs) and metrics summaries for each
model family, reading back the *_test_predictions.csv / *_test_scores.csv
files written during training. Kept independent of the training step so
the report can be regenerated any time from saved predictions.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_soft_sensor(csv_path, plot_dir):
    if not os.path.exists(csv_path):
        return None
    d = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(d.y_true, d.y_pred, s=6, alpha=0.35, color="#2b6cb0")
    lims = [min(d.y_true.min(), d.y_pred.min()), max(d.y_true.max(), d.y_pred.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1, label="perfect prediction")
    axes[0].set_xlabel("True Ca (mol/L), lab analyzer")
    axes[0].set_ylabel("Predicted Ca (mol/L), soft sensor")
    axes[0].set_title("Soft sensor: predicted vs. true concentration")
    axes[0].legend()

    resid = d.y_pred - d.y_true
    axes[1].hist(resid, bins=40, color="#2b6cb0", alpha=0.8)
    axes[1].axvline(0, color="r", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Prediction error (mol/L)")
    axes[1].set_title(f"Residuals (RMSE={np.sqrt((resid**2).mean()):.4f})")

    fig.tight_layout()
    path = os.path.join(plot_dir, "soft_sensor.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_fault_classifier(csv_path, plot_dir):
    if not os.path.exists(csv_path):
        return None
    d = pd.read_csv(csv_path)
    labels = sorted(set(d.y_true) | set(d.y_pred))
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(d.y_true, d.y_pred):
        cm[idx[t], idx[p]] += 1
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Fault classifier: normalized confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                     fontsize=7, color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(plot_dir, "fault_classifier_confusion.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_anomaly_detector(csv_path, plot_dir):
    if not os.path.exists(csv_path):
        return None
    d = pd.read_csv(csv_path)
    d["is_fault"] = (d.y_true_fault != "none")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, grp in d.groupby("is_fault"):
        axes[0].hist(grp.anomaly_score, bins=40, alpha=0.55,
                     label=("fault" if label else "normal"))
    axes[0].set_xlabel("Anomaly score (higher = more unusual)")
    axes[0].set_title("Anomaly score distribution")
    axes[0].legend()

    from sklearn.metrics import roc_curve, roc_auc_score
    try:
        fpr, tpr, _ = roc_curve(d.is_fault.astype(int), d.anomaly_score)
        auc = roc_auc_score(d.is_fault.astype(int), d.anomaly_score)
        axes[1].plot(fpr, tpr, color="#2b6cb0", label=f"AUC={auc:.3f}")
        axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
        axes[1].set_xlabel("False positive rate")
        axes[1].set_ylabel("True positive rate")
        axes[1].set_title("Anomaly detector ROC (any-fault vs. normal)")
        axes[1].legend()
    except Exception:
        pass

    fig.tight_layout()
    path = os.path.join(plot_dir, "anomaly_detector.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_rul(csv_path, plot_dir):
    if not os.path.exists(csv_path):
        return None
    d = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(d.y_true, d.y_pred, s=6, alpha=0.35, color="#c05621")
    lims = [0, max(d.y_true.max(), d.y_pred.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1)
    axes[0].set_xlabel("True remaining useful life (min)")
    axes[0].set_ylabel("Predicted RUL (min)")
    axes[0].set_title("RUL predictor: predicted vs. true")

    resid = d.y_pred - d.y_true
    axes[1].hist(resid, bins=40, color="#c05621", alpha=0.8)
    axes[1].axvline(0, color="r", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Prediction error (min)")
    axes[1].set_title(f"Residuals (RMSE={np.sqrt((resid**2).mean()):.1f} min)")

    fig.tight_layout()
    path = os.path.join(plot_dir, "rul_predictor.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_example_episode(df, plot_dir, fault_name="fouling"):
    """Show one example episode's raw trajectory for intuition/storytelling
    in the report -- picks the first episode with the given fault."""
    candidates = df[df.fault_name == fault_name]["episode_id"].unique()
    if len(candidates) == 0:
        return None
    ep = candidates[0]
    g = df[df.episode_id == ep].sort_values("t_min")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(g.t_min / 60, g.T_true, color="#2b6cb0", linewidth=1)
    axes[0].plot(g.t_min / 60, g.T_setpoint, color="gray", linestyle="--", linewidth=1, label="setpoint")
    axes[0].set_ylabel("Reactor T (K)")
    axes[0].legend(fontsize=8)

    axes[1].plot(g.t_min / 60, g.Tc_cmd, color="#c05621", linewidth=1)
    axes[1].set_ylabel("Commanded Tc (K)")

    axes[2].plot(g.t_min / 60, g.Ca_true, color="#2f855a", linewidth=1, label="true Ca")
    axes[2].scatter(g.t_min / 60, g.Ca_analyzer, s=8, color="black", label="analyzer sample")
    axes[2].set_ylabel("Ca (mol/L)")
    axes[2].set_xlabel("Time (hours)")
    axes[2].legend(fontsize=8)

    onset_hr = g.loc[g.fault_active == 1, "t_min"].min() / 60 if g.fault_active.any() else None
    if onset_hr and not np.isnan(onset_hr):
        for ax in axes:
            ax.axvline(onset_hr, color="red", linestyle=":", linewidth=1)
    axes[0].set_title(f"Example episode: {fault_name} fault (red dotted = onset)")

    fig.tight_layout()
    path = os.path.join(plot_dir, f"example_episode_{fault_name}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_all_evaluation(cfg, out_dir, raw_df=None, log=print):
    plot_dir = os.path.join(out_dir, "plots")
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(plot_dir, exist_ok=True)

    results = {}
    results["soft_sensor_plot"] = plot_soft_sensor(
        os.path.join(data_dir, "soft_sensor_test_predictions.csv"), plot_dir)
    results["classifier_plot"] = plot_fault_classifier(
        os.path.join(data_dir, "fault_classifier_test_predictions.csv"), plot_dir)
    results["anomaly_plot"] = plot_anomaly_detector(
        os.path.join(data_dir, "anomaly_detector_test_scores.csv"), plot_dir)
    results["rul_plot"] = plot_rul(
        os.path.join(data_dir, "rul_test_predictions.csv"), plot_dir)

    if raw_df is not None:
        for fname in ["fouling", "catalyst_deactivation", "sensor_bias_T"]:
            p = plot_example_episode(raw_df, plot_dir, fname)
            if p:
                results[f"example_{fname}"] = p

    log(f"[evaluate] plots written to {plot_dir}")
    return results
