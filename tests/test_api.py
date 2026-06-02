"""API + scoring tests. Trains a tiny model into a temp path if none exists."""
import os

import joblib
import pytest
from fastapi.testclient import TestClient

from data.generate_data import generate
from src import features as F
from src.train import build_model


@pytest.fixture(scope="session", autouse=True)
def ensure_model(tmp_path_factory):
    """Guarantee a model artifact exists before API tests run."""
    from src.config import settings
    if os.path.exists(settings.model_path):
        return
    os.makedirs(settings.model_dir, exist_ok=True)
    df = generate(4000, seed=11).rename(columns={"default": "target"})
    df = F.add_engineered_features(df)
    X, y = F.split_X_y(df)
    model = build_model()
    model.named_steps["clf"].set_params(n_estimators=60)
    model.fit(X, y)
    joblib.dump(
        {
            "model": model, "version": "test",
            "numeric_features": F.NUMERIC_FEATURES,
            "categorical_features": F.CATEGORICAL_FEATURES,
            "metrics": {"roc_auc": 0.7},
        },
        settings.model_path,
    )


@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)


def _example():
    from api.schemas import Applicant
    return Applicant.model_config["json_schema_extra"]["example"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in {"ok", "degraded"}


def test_score_low_risk(client):
    r = client.post("/score", json=_example())
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert body["decision"] in {"APPROVE", "REVIEW", "DECLINE"}


def test_score_validation_error(client):
    bad = _example() | {"credit_score": 9999}  # out of range
    r = client.post("/score", json=bad)
    assert r.status_code == 422


def test_high_risk_scores_higher(client):
    low = _example()
    high = low | {
        "credit_score": 540, "dti": 45, "revol_util": 110,
        "int_rate": 27, "delinq_2yrs": 4, "annual_income": 25000, "grade": "G",
    }
    p_low = client.post("/score", json=low).json()["probability_of_default"]
    p_high = client.post("/score", json=high).json()["probability_of_default"]
    assert p_high > p_low
