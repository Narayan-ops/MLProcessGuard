"""
models.py
=========
Four model families, each solving a real industrial problem:

1. SoftSensorRegressor   -- infer the slow/expensive lab composition (Ca)
                             continuously from fast/cheap sensors (T, Tc, q,
                             level). This is "inferential sensing", widely
                             used in refining/petrochemicals to close the
                             gap between infrequent lab samples and the
                             need for continuous quality control.
2. FaultClassifier        -- given a rolling window of fast tags, identify
                             which (if any) fault is present. Fault
                             Detection & Diagnosis (FDD) is core to process
                             safety and abnormal-situation management.
3. AnomalyDetector         -- unsupervised model trained only on normal
                             operation; flags any deviation, including
                             fault types never seen in training (an
                             important complement to the supervised
                             classifier, which can only recognize faults
                             it was trained on).
4. RULPredictor            -- for incipient (slowly developing) faults,
                             predicts minutes remaining until the control
                             system's cooling authority saturates --
                             i.e. predictive maintenance.

Deep-learning variants (LSTM classifier, autoencoder anomaly detector) are
used when torch is available; otherwise everything gracefully falls back
to tree-based models so the whole pipeline still runs on a machine without
a working torch install.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, f1_score
import xgboost as xgb
import joblib

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except Exception:
    TORCH_OK = False


# ----------------------------------------------------------------------
# Feature engineering shared by classifier / anomaly / RUL models
# ----------------------------------------------------------------------

def window_summary_features(X_windows):
    """(n, window, n_feat) -> (n, n_feat*5) hand-crafted summary features:
    mean, std, min, max, and (last - first) 'slope' per raw feature. Tree
    models work very well on these and it keeps a robust, dependency-light
    path that doesn't require torch."""
    mean = X_windows.mean(axis=1)
    std = X_windows.std(axis=1)
    mn = X_windows.min(axis=1)
    mx = X_windows.max(axis=1)
    slope = X_windows[:, -1, :] - X_windows[:, 0, :]
    return np.concatenate([mean, std, mn, mx, slope], axis=1)


# ----------------------------------------------------------------------
# 1. Soft sensor
# ----------------------------------------------------------------------

class SoftSensorRegressor:
    def __init__(self, params=None):
        self.params = params or dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8)
        self.model = xgb.XGBRegressor(**self.params, n_jobs=-1, random_state=0)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def score(self, X, y):
        pred = self.predict(X)
        return float(np.sqrt(mean_squared_error(y, pred)))

    def save(self, path):
        joblib.dump({"model": self.model, "scaler": self.scaler, "params": self.params}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        obj = cls(d["params"])
        obj.model = d["model"]
        obj.scaler = d["scaler"]
        return obj


# ----------------------------------------------------------------------
# 2. Fault classifier (tree-based on window summary features)
# ----------------------------------------------------------------------

class FaultClassifier:
    def __init__(self, params=None):
        self.params = params or dict(n_estimators=300, max_depth=8, learning_rate=0.08,
                                      subsample=0.8, colsample_bytree=0.8)
        self.model = xgb.XGBClassifier(**self.params, n_jobs=-1, random_state=0,
                                        eval_metric="mlogloss")
        self.scaler = StandardScaler()
        self.classes_ = None

    def fit(self, X_windows, y_labels):
        feats = window_summary_features(X_windows)
        Xs = self.scaler.fit_transform(feats)
        self.classes_, y_int = np.unique(y_labels, return_inverse=True)
        self.model.fit(Xs, y_int)
        return self

    def predict(self, X_windows):
        feats = window_summary_features(X_windows)
        Xs = self.scaler.transform(feats)
        idx = self.model.predict(Xs)
        return self.classes_[idx]

    def predict_proba(self, X_windows):
        feats = window_summary_features(X_windows)
        Xs = self.scaler.transform(feats)
        return self.model.predict_proba(Xs)

    def score(self, X_windows, y_labels):
        pred = self.predict(X_windows)
        return float(f1_score(y_labels, pred, average="macro"))

    def save(self, path):
        joblib.dump({"model": self.model, "scaler": self.scaler, "classes_": self.classes_,
                      "params": self.params}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        obj = cls(d["params"])
        obj.model = d["model"]
        obj.scaler = d["scaler"]
        obj.classes_ = d["classes_"]
        return obj


if TORCH_OK:
    class LSTMClassifier(nn.Module):
        def __init__(self, n_features, n_classes, hidden=48, layers=1):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True)
            self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, n_classes))

        def forward(self, x):
            out, (h, c) = self.lstm(x)
            return self.head(h[-1])

    class AutoencoderNet(nn.Module):
        def __init__(self, n_features, window, latent=8):
            super().__init__()
            in_dim = n_features * window
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, 64), nn.ReLU(),
                nn.Linear(64, latent), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent, 64), nn.ReLU(),
                nn.Linear(64, in_dim),
            )
            self.in_dim = in_dim

        def forward(self, x):
            flat = x.reshape(x.shape[0], -1)
            z = self.encoder(flat)
            recon = self.decoder(z)
            return recon.reshape(x.shape)


# ----------------------------------------------------------------------
# 3. Anomaly detector (unsupervised, trained on normal data only)
# ----------------------------------------------------------------------

class AnomalyDetector:
    """IsolationForest on window summary features, trained only on
    'none'-labeled windows. Reports an anomaly score for any window;
    high score = looks unlike normal operation, regardless of whether the
    underlying fault type was ever seen in training."""

    def __init__(self, params=None):
        self.params = params or dict(n_estimators=300, contamination=0.05)
        self.model = IsolationForest(**self.params, random_state=0, n_jobs=-1)
        self.scaler = StandardScaler()

    def fit(self, X_windows_normal):
        feats = window_summary_features(X_windows_normal)
        Xs = self.scaler.fit_transform(feats)
        self.model.fit(Xs)
        return self

    def anomaly_score(self, X_windows):
        feats = window_summary_features(X_windows)
        Xs = self.scaler.transform(feats)
        # higher = more anomalous (flip sklearn's convention)
        return -self.model.score_samples(Xs)

    def save(self, path):
        joblib.dump({"model": self.model, "scaler": self.scaler, "params": self.params}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        obj = cls(d["params"])
        obj.model = d["model"]
        obj.scaler = d["scaler"]
        return obj


# ----------------------------------------------------------------------
# 4. RUL predictor
# ----------------------------------------------------------------------

class RULPredictor:
    def __init__(self, params=None):
        self.params = params or dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8)
        self.model = xgb.XGBRegressor(**self.params, n_jobs=-1, random_state=0)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict(self, X):
        return np.clip(self.model.predict(self.scaler.transform(X)), 0, None)

    def score(self, X, y):
        pred = self.predict(X)
        return float(np.sqrt(mean_squared_error(y, pred)))

    def save(self, path):
        joblib.dump({"model": self.model, "scaler": self.scaler, "params": self.params}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        obj = cls(d["params"])
        obj.model = d["model"]
        obj.scaler = d["scaler"]
        return obj
