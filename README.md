<h1 align="center">📦 Parcel Delivery Days — Bhopal</h1>

<p align="center">
  <b>G027 · MP105</b> · SCSE3040 MLOps Mini Project<br>
  An end-to-end MLOps pipeline for <i>Bhopal Speed Parcel Hub</i>
</p>

<p align="center">
  <a href="https://github.com/Saurav021406/G027-mp105-mlops/actions/workflows/ci.yml"><img src="https://github.com/Saurav021406/G027-mp105-mlops/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.12-blue" alt="Python 3.12">
  <img src="https://img.shields.io/badge/task-regression-orange" alt="Regression">
  <img src="https://img.shields.io/badge/metric-MAE-green" alt="MAE">
  <img src="https://img.shields.io/badge/MLflow-tracked-0194E2" alt="MLflow">
  <img src="https://img.shields.io/badge/DVC-data%20versioned-945DD6" alt="DVC">
</p>

---

## 🎯 The problem

The hub in Bhopal promises delivery dates that are often wrong. We predict
**`delivery_days`** (days from booking to delivery) for every parcel, so that dates
are honest and slow lanes can be found and fixed. It is a **regression** problem,
measured with **MAE** (average mistake, in days).

| Group | Problem | Faculty | Batch |
|---|---|---|---|
| **G027** | **MP105** | Amit Soni | 69 |

## 👥 Team

| Enrolment no. | Name | GitHub | Built |
|---|---|---|---|
| S24CSEU2576 | Kumar Saurav | [@Saurav021406](https://github.com/Saurav021406) | repo, DVC, cleaning, training, MLflow, API, CI, K8s, drift & retraining |
| S24CSEU2606 | Urjita Jain | [@Urjitaa](https://github.com/Urjitaa) | duplicate and leakage checks |
| S24CSEU2605 | Lakshita Badlia | [@lakshitabadlia](https://github.com/lakshitabadlia) | spelling-variant check |
| S24CSEU0104 | Manan Garg | [@Manan7289](https://github.com/Manan7289) | stuck-sensor check |
| S24CSEU2604 | Ruhi Mishra | [@RuhiMishra2202](https://github.com/RuhiMishra2202) | sentinel check, config and requirements |

## 🗺️ Pipeline at a glance

```
 train.csv ──► data checks ──► clean_data ──► train (3 MLflow runs) ──► registry: v1 @champion
 (DVC)         (7 problems)                                                    │
                                                                               ▼
 later.csv ──► PSI drift ──► v1 MAE per group ──► retrain v2 ──► 12% rule ──► v2 @champion
                                                                               │
                              holdout_features.csv ──► predict ◄───────────────┘
                                                          │
                                  FastAPI  ──►  Docker  ──►  Kubernetes (4 replicas)
```

## 🔎 The 7 data problems in `train.csv`

Every answer is produced by code in [`src/parcel_days/data_checks.py`](src/parcel_days/data_checks.py)
and recorded in [`reports/findings.yaml`](reports/findings.yaml).

| Problem | Found? | Details |
|---|---|---|
| Leaking column | ❌ no | only the target is missing from the holdout file |
| Unit mix-up | ✅ | `weight_kg` for **Metro** is in grams (× 1000) |
| Sentinel value | ❌ no | no impossible repeated number |
| Duplicate rows | ✅ | **38** extra copies |
| Stuck sensor | ✅ | `declared_value_rs` stuck at 19805, **2026-05-10 → 2026-05-27** |
| Target typos | ✅ | **14** rows with `delivery_days` typed 10× too big |
| Spelling variants | ✅ | `service` written 18 ways, **3** real levels (express, standard, economy) |

All fixes live in [`clean.py`](src/parcel_days/clean.py), and the unit tests in `tests/` cover them.

## 📈 Results

**Model:** scikit-learn pipeline (one-hot encoding + Random Forest), every run tracked in MLflow.

| Run | n_estimators | max_depth | MAE (validation) |
|---|---|---|---|
| rf_0 | 50 | 5 | 0.508 |
| rf_1 | 100 | 10 | 0.416 |
| **rf_2 (v1)** | **200** | **None** | **0.413** |

**Two months later, on `later.csv`:**

| Check | Result |
|---|---|
| Most drifted column (PSI, 10 bins) | **`distance_km`** — PSI 4.34 (others below 0.06) |
| v1 MAE by destination | Metro 1.17 · Tier-2 1.05 · **Rural 2.59** (worst served) |
| v1 vs v2 on the last 30% of `later.csv` | MAE **1.92 → 0.80** (58.5% better) |
| Client rule: promote only if ≥ 12% better | ✅ **v2 promoted** to `@champion` |

The final predictions for September are in [`predictions/holdout_predictions.csv`](predictions/holdout_predictions.csv) (250 rows, `record_id,prediction`).

## 🚀 Quick start

```bash
git clone https://github.com/Saurav021406/G027-mp105-mlops.git
cd G027-mp105-mlops
pip install -r requirements.txt
```

Put the original `train.csv` and `later.csv` from the course zip into `data/raw/`
(they are tracked by DVC and are not stored in git; never open them in Excel).

```bash
# Windows (cmd)                      # Linux / macOS
set PYTHONPATH=src;.                 export PYTHONPATH=src:.
```

| Step | Command |
|---|---|
| Run the tests | `python -m pytest tests -v` |
| Run the 7 data checks | `python src/parcel_days/data_checks.py` |
| Train v1 (3 MLflow runs) | `python -m parcel_days.train` |
| PSI drift | `python -m parcel_days.monitor` |
| v1 MAE per destination | `python -m parcel_days.evaluate` |
| Retrain v2 and apply the 12% rule | `python -m parcel_days.retrain` |
| Predict the holdout file | `python -m parcel_days.predict` |
| Open the MLflow UI | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |

> `train` and `retrain` each create new model versions in the registry, so run them once.

## 🌐 The prediction service

```bash
uvicorn app.main:app --port 8000
```

Open <http://localhost:8000/docs> for the interactive page.

`GET /health` → `{"status": "ok"}`

`POST /predict`

```json
{
  "distance_km": 1200,
  "weight_kg": 10.5,
  "hub_backlog": 2500,
  "pin_code_tier": 2,
  "declared_value_rs": 8000,
  "transfers": 3,
  "service": "express",
  "destination": "Metro"
}
```

Returns `{"prediction": <days>}`.

The client's two rules are written in the request schema, so a bad request gets **422** and never reaches the model:

- `distance_km` above **4500** is rejected
- `transfers` below **0** is rejected

Every prediction writes one **JSON log line** to stdout (no personal details, never to a file).

## 🐳 Docker and ☸️ Kubernetes

```bash
docker build -t mp105-api:1.0 .
docker run -p 8000:8000 mp105-api:1.0
kubectl apply -f k8s/
```

| Thing | Value |
|---|---|
| Image | `mp105-api:1.0` (`python:3.12-slim`, dependencies installed before code is copied) |
| Deployment | `mp105-api`, **4 replicas**, container port 8000 |
| Service | `mp105-svc`, port 80 → 8000 |
| Probes | readiness and liveness on `/health` (liveness waits longer) |

## ⚙️ CI

[GitHub Actions](.github/workflows/ci.yml) runs `pytest` on every push.

## 📁 Repository layout

```
├── app/main.py                 FastAPI service (/predict, /health)
├── src/parcel_days/
│   ├── data_checks.py          the 7 data-problem checks
│   ├── clean.py                cleaning code
│   ├── train.py                v1 training with MLflow tracking and registry
│   ├── monitor.py              PSI drift
│   ├── evaluate.py             MAE per destination
│   ├── retrain.py              v2, the 12% promotion rule
│   └── predict.py              holdout predictions
├── tests/                      pytest tests (incl. TestClient tests for the API)
├── data/raw/                   *.dvc files (CSVs are tracked by DVC, not git)
├── models/                     champion model used by the API
├── k8s/                        deployment.yaml, service.yaml
├── .github/workflows/ci.yml    CI
├── reports/findings.yaml       the filled findings form
├── predictions/                holdout_predictions.csv
├── mlflow.db                   MLflow tracking database and model registry
├── params.yaml                 settings read by the code
├── Dockerfile
└── requirements.txt
```
