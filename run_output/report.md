# MLProcessGuard — Training Report

Total wall-clock budget: 1.21 hours
Generated: 2026-08-11T05:57:22

## Dataset
- Episodes simulated: 400
- Total rows: 1152000
- Generation time: 2.7 min

## 1. Soft sensor (predicts lab Ca from fast tags)
- Optuna trials: 200
- Test RMSE: 0.07512 mol/L
- See `plots/soft_sensor.png`

## 2. Fault classifier (XGBoost + Optuna)
- Optuna trials: 150
- Test macro-F1: 0.5793
- See `plots/fault_classifier_confusion.png`

## 3. Anomaly detector (Isolation Forest, trained on normal data only)
- Best contamination: 0.02
- Test ROC-AUC: 0.6642
- See `plots/anomaly_detector.png`

## 4. RUL predictor (time-to-saturation for incipient faults)
- Optuna trials: 150
- Test RMSE: 553.8 minutes
- See `plots/rul_predictor.png`

## Next steps
Run `python app.py` to serve the trained models through the live dashboard at http://localhost:5000, or see `DEPLOYMENT.md` to host it.