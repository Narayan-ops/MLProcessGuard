# MLProcessGuard

A digital-twin simulator of a non-isothermal, jacket-cooled CSTR (the
classic Bequette bistable reactor) paired with four ML models that mirror
what refining and petrochemical plants actually run in production:

| Model | Problem | Method |
|---|---|---|
| Soft sensor | Infer the slow lab concentration from fast, cheap tags | XGBoost regression, Optuna-tuned |
| Fault classifier | Identify which of 7 fault conditions is active | XGBoost multiclass, Optuna-tuned |
| Anomaly detector | Flag deviations the classifier was never trained on | Isolation Forest, trained on normal data only |
| RUL predictor | Estimate time-to-cooling-saturation for incipient faults | XGBoost regression on right-censored labels |

A live dashboard (`app/`) serves the trained models through a FastAPI
backend and an industrial-HMI-styled frontend where you can pick a fault
scenario and watch all four models react in real time.

**For deploying the live demo, see [`DEPLOYMENT.md`](DEPLOYMENT.md).**

---

## 1. Set up

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run the overnight training

This is the one command you run before going to sleep:

```bash
python run.py --hours 5
```

What it does, in order:

1. Generates the simulated dataset (`dataset.n_episodes` in `config.yaml`,
   default 400 episodes x 48 simulated hours each). Takes a few minutes.
2. Trains the soft sensor (Optuna search over XGBoost hyperparameters).
3. Trains the fault classifier (same, multiclass).
4. Trains the anomaly detector (Isolation Forest, contamination sweep).
5. Trains the RUL predictor (Optuna search, right-censored labels).
6. Runs evaluation and writes plots + a final report.

Each phase gets a wall-clock share of `--hours`, defined in
`config.yaml` under `training.time_allocation`. Optuna is given a
`timeout`, not a fixed trial count, so it naturally uses whatever time
it's given regardless of how fast or slow your machine is.

**Run it so it survives your laptop sleeping / terminal closing:**

```bash
nohup python run.py --hours 5 > run_output/console.log 2>&1 &
```

(On Windows, run it inside WSL with `nohup`, or leave a terminal window
open and disable sleep in your power settings for the night.)

Every phase is wrapped in error handling and checkpoints its best model
continuously, and the report is rewritten after every phase — so even if
you stop it early or it crashes partway through, you still have valid,
usable models and a readable report.

### Quick smoke test first (recommended, ~2 minutes)

Before committing to the full 5-hour run, sanity-check the whole pipeline
end to end with a tiny dataset:

```bash
python run.py --hours 0.03 --episodes 20 --out run_output_test
```

If that finishes cleanly and `run_output_test/report.md` looks sensible,
the full run will too.

## 3. In the morning

```bash
cat run_output/report.md              # headline metrics
open run_output/plots/*.png           # confusion matrix, ROC curve, parity plots, etc.
```

## 4. Run the live dashboard locally

```bash
cd app/backend
pip install -r requirements.txt       # if you haven't already via the root requirements.txt
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000`. Pick a fault scenario, hit **Run scenario**,
and watch the temperature, concentration, anomaly score, and RUL charts
update from a freshly simulated episode passed through your trained
models.

If `run_output/models/` doesn't exist yet, the dashboard still loads —
it just tells you which models aren't available instead of crashing.

---

## Project layout

```
config.yaml              all physical, control, fault, and training parameters
simulator.py              the CSTR digital twin (physics + PID control)
data_generator.py         runs many episodes in parallel, assembles the dataset
preprocessing.py          windowing, feature engineering, episode-level splits
models.py                 the four model classes
train.py                  time-budgeted Optuna training for each model
evaluate.py                plots + metrics from saved test predictions
run.py                      orchestrates all of the above — run this one

app/
  backend/main.py            FastAPI inference server (loads run_output/models/)
  backend/requirements.txt   lean deployment-only dependencies
  frontend/                   static HTML/CSS/JS dashboard

run_output/                 created by run.py — models, plots, report, logs
render.yaml / Procfile       deployment configs, see DEPLOYMENT.md
```

## Notes for reviewers

- All models are trained and evaluated on simulator-generated data. No
  real plant data has validated any result in this repo.
- The reactor is a genuinely nonlinear, bistable system (real open-loop
  input multiplicity) — not a toy linear model.
- Fault labels and RUL targets come from the physics of the simulator
  itself (e.g. RUL = time until the PID controller's cooling command
  saturates), not from an arbitrary heuristic.
- Episodes that never reach saturation are right-censored and excluded
  from RUL training, rather than assigned an invented label.
