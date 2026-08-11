"""
app/backend/main.py
====================
Lightweight inference API for the MLProcessGuard live demo.

This process does NOT train anything -- it loads the model artifacts that
`run.py` produced during the overnight training run (from
`run_output/models/*.joblib`) and serves fast, on-demand simulated
episodes through them. That split matters for deployment: training needs
hours on your own machine; serving a demo needs milliseconds on a free-tier
web dyno.

Endpoints
---------
GET  /api/health           -> liveness check + which models are loaded
GET  /api/report           -> the metrics.json produced by the training run
                               (headline numbers shown on the dashboard)
GET  /api/fault-types      -> fault catalogue (from config.yaml) for the UI
POST /api/simulate         -> {fault_type, severity} -> simulates one
                               episode and returns raw traces + every
                               model's predictions over that episode
"""

import os
import time
import yaml
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from simulator import CSTRSimulator, FaultState  # noqa: E402
from preprocessing import FAST_FEATURES, make_windows, make_soft_sensor_table  # noqa: E402
from models import SoftSensorRegressor, FaultClassifier, AnomalyDetector, RULPredictor  # noqa: E402

CONFIG_PATH = os.environ.get("MLPG_CONFIG", os.path.join(ROOT, "config.yaml"))
MODEL_DIR = os.environ.get("MLPG_MODEL_DIR", os.path.join(ROOT, "run_output", "models"))
REPORT_PATH = os.environ.get("MLPG_REPORT", os.path.join(ROOT, "run_output", "metrics.json"))
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
# Demo episodes are shorter than the 48h training episodes so a button
# click on the deployed site returns in well under a second.
DEMO_EPISODE_HOURS = float(os.environ.get("MLPG_DEMO_HOURS", "12"))

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

app = FastAPI(title="MLProcessGuard API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_models = {}


def _try_load(name, cls):
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    if os.path.exists(path):
        try:
            _models[name] = cls.load(path)
            return True
        except Exception as e:
            print(f"[startup] failed to load {name}: {e}")
    return False


@app.on_event("startup")
def load_models():
    ok = {
        "soft_sensor": _try_load("soft_sensor", SoftSensorRegressor),
        "fault_classifier": _try_load("fault_classifier", FaultClassifier),
        "anomaly_detector": _try_load("anomaly_detector", AnomalyDetector),
        "rul_predictor": _try_load("rul_predictor", RULPredictor),
    }
    print(f"[startup] models loaded: {ok}")


class SimulateRequest(BaseModel):
    fault_type: str = "none"
    severity: float | None = None  # None -> sample a representative severity


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": {k: True for k in _models.keys()},
            "config_loaded": CFG is not None}


@app.get("/api/report")
def report():
    if not os.path.exists(REPORT_PATH):
        raise HTTPException(status_code=404, detail="No training report found yet. Run run.py first.")
    import json
    with open(REPORT_PATH) as f:
        return json.load(f)


@app.get("/api/fault-types")
def fault_types():
    out = []
    for spec in CFG["faults"]["types"]:
        out.append({
            "name": spec["name"],
            "kind": spec.get("kind"),
            "description": spec.get("description", "Normal operation \u2014 no fault injected."),
        })
    return out


def _severity_for(spec, requested):
    if requested is not None:
        return requested
    if spec["name"] == "none":
        return 0.0
    lo, hi = spec["severity_range"]
    return (lo + hi) / 2.0


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    spec = next((f for f in CFG["faults"]["types"] if f["name"] == req.fault_type), None)
    if spec is None:
        raise HTTPException(status_code=400, detail=f"Unknown fault_type '{req.fault_type}'")

    t0 = time.time()
    cfg = dict(CFG)
    # shrink episode length for a snappy demo without touching the trained models
    cfg = {**cfg, "simulator": {**cfg["simulator"], "episode_hours": DEMO_EPISODE_HOURS}}

    rng = np.random.default_rng(int(time.time() * 1000) % (2**31 - 1))
    severity = _severity_for(spec, req.severity)

    if spec["name"] == "none":
        fault = FaultState("none", None, onset_min=0.0, severity=0.0)
    elif spec["kind"] == "abrupt":
        onset_min = min(DEMO_EPISODE_HOURS * 60 * 0.35, 300)
        fault = FaultState(spec["name"], "abrupt", onset_min=onset_min, severity=severity)
    else:
        ramp_min = min(DEMO_EPISODE_HOURS * 60 * 0.4, 480)
        onset_min = DEMO_EPISODE_HOURS * 60 * 0.15
        fault = FaultState(spec["name"], "incipient", onset_min=onset_min, severity=severity, ramp_min=ramp_min)

    sim = CSTRSimulator(cfg)
    df = sim.run_episode(rng, fault)
    df["episode_id"] = 0

    result = {
        "fault_type": spec["name"],
        "severity": severity,
        "onset_min": fault.onset_min if fault.kind else None,
        "t_min": df["t_min"].tolist(),
        "T_true": df["T_true"].round(3).tolist(),
        "T_meas": df["T_meas"].round(3).tolist(),
        "T_setpoint": df["T_setpoint"].round(3).tolist(),
        "Tc_cmd": df["Tc_cmd"].round(3).tolist(),
        "Ca_true": df["Ca_true"].round(5).tolist(),
        "Ca_analyzer": [None if pd.isna(v) else round(float(v), 5) for v in df["Ca_analyzer"]],
        "pid_saturated": df["pid_saturated"].tolist(),
    }

    # --- soft sensor: continuous Ca estimate from fast tags only ---
    if "soft_sensor" in _models:
        Xs = df[FAST_FEATURES]
        pred = _models["soft_sensor"].predict(Xs)
        result["soft_sensor_pred"] = np.round(pred, 5).tolist()

    # --- windowed models: fault classifier + anomaly detector ---
    window_size = cfg["dataset"]["window_size"]
    stride = max(1, cfg["dataset"]["window_stride"])
    Xw, yw, epw, tw = make_windows(df, window_size, stride, FAST_FEATURES)
    result["window_t_min"] = tw.tolist()

    if "fault_classifier" in _models and len(Xw):
        preds = _models["fault_classifier"].predict(Xw)
        probs = _models["fault_classifier"].predict_proba(Xw)
        confidence = probs.max(axis=1)
        result["classifier_pred"] = preds.tolist()
        result["classifier_confidence"] = np.round(confidence, 4).tolist()

    if "anomaly_detector" in _models and len(Xw):
        scores = _models["anomaly_detector"].anomaly_score(Xw)
        result["anomaly_score"] = np.round(scores, 5).tolist()

    # --- RUL: only meaningful for incipient faults, but we show the raw
    # trace whenever the model is available so the UI can visualize it ---
    if "rul_predictor" in _models:
        Xr = df[FAST_FEATURES]
        rul_pred = _models["rul_predictor"].predict(Xr)
        result["rul_pred_min"] = np.round(rul_pred, 1).tolist()

    result["compute_ms"] = round((time.time() - t0) * 1000, 1)
    return result


# --- serve the static frontend last, so /api/* routes above take priority ---
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
