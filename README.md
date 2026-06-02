# Credit Risk MLOps — End-to-End Project

A complete, production-shaped machine-learning project for **credit risk (probability of default)** modelling. It covers the full lifecycle:

**Data in SQL Server → feature engineering → model training (XGBoost + MLflow) → REST API serving → automated tests → CI/CD → containerised deployment (local or free-tier cloud).**

Everything runs **locally for free** (Docker + SQL Server Developer edition), and the same artifacts deploy to a free cloud tier with no code changes.

---

## 1. Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │                 SQL Server                    │
   generate_data │  raw.loan_applications  (50k loans)           │
   ────────────► │  curated.vw_model_features  (engineered)      │
                 │  curated.vw_train / vw_test (deterministic)   │
                 │  curated.prediction_log     (audit trail)     │
                 └───────────────┬──────────────────────────────┘
                                 │  read_features()
                                 ▼
        ┌────────────────────────────────────────────┐
        │  src/train.py                               │
        │   preprocess (impute+scale+one-hot)         │
        │   XGBoost classifier                        │   ──► MLflow (params, metrics, model)
        │   quality gate (ROC-AUC ≥ 0.70)             │   ──► models/credit_risk_model.joblib
        └───────────────┬─────────────────────────────┘
                        │  load_bundle()
                        ▼
        ┌────────────────────────────────────────────┐        ┌─────────────────┐
        │  api/main.py  (FastAPI)                     │        │ curated.        │
        │   GET  /health   GET /metrics               │ ─────► │ prediction_log  │
        │   POST /score → {pd, decision, version}     │        │ (writeback)     │
        └────────────────────────────────────────────┘        └─────────────────┘
                        ▲
                        │  CI/CD (GitHub Actions): test → integration (real SQL) → build/push image
```

The same engineered feature definitions live in **both** the SQL view and `src/features.py`, so training and online scoring never drift.

---

## 2. Project structure

```
credit-risk-mlops/
├── data/
│   └── generate_data.py        # realistic synthetic data generator
├── sql/
│   ├── 01_create_database.sql  # database + schemas
│   ├── 02_create_tables.sql    # raw landing + prediction_log
│   └── 03_feature_views.sql    # engineered features + train/test split
├── src/
│   ├── config.py               # env-driven settings
│   ├── db.py                   # SQL Server access (load, read, log)
│   ├── features.py             # feature engineering + sklearn preprocessor
│   ├── train.py                # training + MLflow + quality gate
│   └── predict.py              # inference + decision policy
├── api/
│   ├── main.py                 # FastAPI app
│   └── schemas.py              # pydantic request/response models
├── scripts/
│   └── setup_database.py       # one-shot: generate → create objects → load
├── tests/
│   ├── test_data.py            # data + feature tests
│   └── test_api.py             # API + scoring tests
├── notebooks/
│   └── explore.py              # EDA + feature importances
├── .github/workflows/ci-cd.yml # CI/CD pipeline
├── Dockerfile                  # API image (with ODBC driver)
├── docker-compose.yml          # SQL Server + API
├── requirements.txt
├── Makefile
├── .env.example
└── README.md
```

---

## 3. The dataset

`data/generate_data.py` produces ~50,000 loan records. It is **synthetic but realistic**: default probability is driven by genuine economic relationships (debt-to-income, credit score, revolving utilisation, loan-to-income, delinquencies, employment length, term, interest rate) passed through a logistic link with irreducible noise.

Properties that match real credit portfolios:
- **~14% default rate** (mild class imbalance, handled with `scale_pos_weight`).
- **Realistic missingness** injected into `employment_length_yrs`, `dti`, `revol_util`.
- **Monotonic risk by grade** (A ≈ 9% default → D ≈ 55%).
- **Achievable ROC-AUC ≈ 0.76** — the right ballpark for credit scoring. (If a credit model scores 0.95+, the data is usually leaky.)

Columns: loan terms, applicant finances/demographics, credit-bureau attributes, and the binary `default` target. Full schema is in `sql/02_create_tables.sql`.

---

## 4. Prerequisites

- **Docker Desktop** (easiest path — bundles SQL Server), **or** a local/remote SQL Server instance.
- **Python 3.11+** if running outside Docker.
- **ODBC Driver 18 for SQL Server** (for local non-Docker runs):
  - macOS: `brew install msodbcsql18`
  - Ubuntu: see the install steps in `.github/workflows/ci-cd.yml`
  - Windows: download from Microsoft.

---

## 5. Quick start — full stack with Docker (recommended)

This brings up SQL Server **and** the API in one command.

```bash
cp .env.example .env            # adjust DB_PASSWORD if you like (min 8 chars, complex)

# 1) Start SQL Server + build the API image
docker compose up -d --build

# 2) Wait until SQL Server is healthy (~30s), then create objects + load data.
#    Run from your host (needs Python + ODBC driver) OR exec into the api container:
docker compose exec api python scripts/setup_database.py --rows 50000

# 3) Train the model on the SQL data (writes models/credit_risk_model.joblib)
docker compose exec api python -m src.train

# 4) The API auto-serves the new model. Test it:
curl http://localhost:8000/health
```

Open the interactive API docs at **http://localhost:8000/docs**.

---

## 6. Step-by-step — local (no Docker for the app)

Use this if you already have SQL Server running (Docker container, LocalDB, or a server).

```bash
# 0) Environment
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # set DB_SERVER, DB_PASSWORD, etc.

# (Optional) just start SQL Server in Docker without the app:
docker compose up -d sqlserver

# 1) Generate the dataset
make data                 # or: python data/generate_data.py --rows 50000

# 2) Create database objects and load the CSV
make db                   # or: python scripts/setup_database.py --rows 50000

# 3) Train (reads from the SQL curated views, logs to MLflow)
make train                # or: python -m src.train

# 4) Inspect experiments
mlflow ui                 # http://localhost:5000

# 5) Serve the API
make serve                # uvicorn at http://localhost:8000

# 6) Run the test suite
make test
```

> **Offline shortcut:** to train without a database (straight from the CSV), use
> `python -m src.train --from-csv data/credit_risk.csv`. Useful for laptops or CI smoke tests.

---

## 7. Using the API

**Score an applicant:**

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "member_id": 900001,
    "loan_amount": 15000, "term_months": 36, "int_rate": 9.5,
    "annual_income": 85000, "age": 41, "employment_length_yrs": 8,
    "dti": 14.2, "credit_score": 760, "revol_util": 22.0,
    "open_accounts": 9, "total_accounts": 22, "delinq_2yrs": 0,
    "inquiries_6m": 1, "pub_rec": 0, "grade": "A",
    "loan_purpose": "credit_card", "home_ownership": "MORTGAGE",
    "verification_status": "Verified"
  }'
```

**Response:**

```json
{ "probability_of_default": 0.0453, "decision": "APPROVE", "model_version": "20260602025231" }
```

Decision policy (configurable via `.env`): `pd < 0.15 → APPROVE`, `pd > 0.45 → DECLINE`, otherwise `REVIEW`. Every scored request is written to `curated.prediction_log` for auditing and future monitoring.

Endpoints: `GET /health`, `GET /metrics` (held-out test metrics), `POST /score`, plus auto-generated docs at `/docs`.

---

## 8. CI/CD pipeline

`.github/workflows/ci-cd.yml` runs on every push/PR to `main` and has three stages:

1. **test** — installs deps, lints with `ruff`, runs the unit test suite (no DB needed → fast).
2. **integration** — spins up a **real SQL Server 2022 service container**, runs `setup_database.py`, trains on the SQL data, and enforces a **quality gate** (`ROC-AUC ≥ 0.70`). Fails the build if the model regresses. Uploads the trained model as a build artifact.
3. **build-and-push** (main only) — builds the Docker image with the trained model baked in and pushes it to GitHub Container Registry (`ghcr.io`).

This is genuine continuous *delivery*: a green pipeline produces a deployable, versioned image whose model has passed a measurable quality bar.

To use it: push this repo to GitHub. The workflow runs automatically. No secrets needed beyond the auto-provided `GITHUB_TOKEN`.

---

## 9. Deploying to a free cloud tier

The container is self-contained, so any of these work without code changes:

- **Render (free web service):** create a *Web Service* from the repo, environment = Docker. Set env vars (`DB_*`). Point `DB_SERVER` at a free Azure SQL or a managed instance.
- **Fly.io (free allowance):** `fly launch` detects the Dockerfile; `fly deploy` ships it. Set secrets with `fly secrets set DB_PASSWORD=...`.
- **Azure SQL Database (free tier):** Microsoft offers a free Azure SQL instance — natural pairing for SQL Server. Run the `sql/` scripts against it and point the API there.
- **Hugging Face Spaces (Docker SDK, free):** push the repo; it builds and serves the FastAPI app.

The model artifact is portable: train locally (or in CI), then mount/copy `models/credit_risk_model.joblib` into the deployed container.

---

## 10. MLflow tracking

Training logs parameters, test metrics (ROC-AUC, PR-AUC, F1, Brier), and the model to MLflow. By default it uses a local `./mlruns` store (`MLFLOW_TRACKING_URI=file:./mlruns`). Launch the UI with `mlflow ui`. Point `MLFLOW_TRACKING_URI` at a remote server to centralise experiments.

---

## 11. Suggested next steps

- **Model monitoring:** the `prediction_log` table is your foundation — compare live score distributions against training to detect drift.
- **Hyperparameter search:** wrap `src/train.py` in Optuna and log trials to MLflow.
- **Explainability:** add SHAP values to the `/score` response for adverse-action reasons (often a regulatory requirement in lending).
- **Model registry:** promote MLflow runs through `Staging` → `Production` and have the API load by registry stage.
```
```

> **Disclaimer:** the dataset is synthetic and for educational/portfolio use. A real lending model must address fairness, regulatory compliance (e.g. ECOA/FCRA adverse-action notices), and validation requirements before any production use.
