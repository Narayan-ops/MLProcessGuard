# Deploying the live demo

The app is split into two halves on purpose:

- **Training** (`run.py` + everything at the repo root) is heavy —
  Optuna searches, `xgboost`, `matplotlib`, `pyarrow` — and is meant to
  run on your own machine overnight (see the main [`README.md`](README.md)).
- **Serving** (`app/backend/main.py`) is light — it only loads the
  `.joblib` model files that training produced and answers requests in
  milliseconds. This is what you actually deploy.

So the deploy step is: **train locally, commit `run_output/models/` +
`run_output/metrics.json`, then deploy `app/backend`.**

## 1. Train, then check what you're about to ship

```bash
python run.py --hours 5
cat run_output/report.md
```

Make sure `run_output/models/*.joblib` and `run_output/metrics.json`
exist — the API and dashboard both read from those.

## 2. Commit the artifacts

`run_output/data/episodes.parquet` and the raw logs are excluded by
`.gitignore` (large and fully reproducible). The models, plots, and
report are small and *are* meant to be committed:

```bash
git add run_output/models run_output/metrics.json run_output/report.md run_output/plots
git commit -m "Add trained models for deployment"
```

## 3. Deploy `app/backend`

### Option A — Render (`render.yaml`)

The repo already includes a `render.yaml` blueprint:

```yaml
buildCommand: pip install -r app/backend/requirements.txt
startCommand: cd app/backend && uvicorn main:app --host 0.0.0.0 --port $PORT
```

In the Render dashboard: **New → Blueprint**, point it at this repo, and
it will pick up `render.yaml` automatically. Free tier is enough — the
backend is inference-only.

### Option B — Heroku / any buildpack platform that reads a `Procfile`

The included `Procfile`:

```
web: cd app/backend && uvicorn main:app --host 0.0.0.0 --port $PORT
```

```bash
heroku create mlprocessguard
git push heroku main
```

### Option C — any other host (Railway, Fly.io, a bare VM, etc.)

Install `app/backend/requirements.txt` and run:

```bash
cd app/backend
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Environment variables

All optional — sensible defaults are baked in (see
`app/backend/main.py`):

| Variable | Default | Purpose |
|---|---|---|
| `MLPG_CONFIG` | `config.yaml` at repo root | Which config to load fault types / simulator params from |
| `MLPG_MODEL_DIR` | `run_output/models` | Where to load the four `.joblib` model files from |
| `MLPG_REPORT` | `run_output/metrics.json` | Headline metrics shown on the dashboard |
| `MLPG_DEMO_HOURS` | `12` | Simulated episode length for on-demand demo runs (shorter than the 48h training episodes, so a button click stays fast) |
| `PORT` | `8000` | Port uvicorn binds to (most PaaS providers set this for you) |

## What happens if models aren't present

`app/backend/main.py` doesn't crash if `run_output/models/` is missing or
incomplete — `/api/health` reports which models loaded, and the frontend
shows "model not loaded" for anything missing instead of erroring out.
That means you can deploy the frontend/backend shell before training
finishes, and it'll light up as models become available.

## Updating the deployed models

Re-run `python run.py --hours N` locally, commit the updated
`run_output/models/*.joblib` and `run_output/metrics.json`, push, and
redeploy (or let autoDeploy pick it up, as configured in `render.yaml`).
