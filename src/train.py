"""Train the credit-risk model.

Reads features from SQL Server (or a local CSV fallback), fits an XGBoost
classifier inside an sklearn Pipeline (preprocessor + model), logs metrics and the
artifact to MLflow, and saves a joblib bundle the API serves.

Usage:
    python -m src.train                 # read from SQL views
    python -m src.train --from-csv data/credit_risk.csv   # offline fallback
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, classification_report,
    f1_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import features as F
from .config import settings


def _load_data(from_csv: str | None):
    """Return (train_df, test_df) with engineered features + target."""
    if from_csv:
        df = pd.read_csv(from_csv).rename(columns={"default": "target"})
        df = F.add_engineered_features(df)
        # Deterministic 80/20 split mirroring the SQL hash split.
        h = (df["member_id"] * 2654435761 % 100)
        return df[h >= 20].copy(), df[h < 20].copy()

    from .db import read_features
    train_df = read_features("train")
    test_df = read_features("test")
    return train_df, test_df


def build_model() -> Pipeline:
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=5,
        eval_metric="auc",
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline(steps=[
        ("prep", F.build_preprocessor()),
        ("clf", clf),
    ])


def evaluate(model: Pipeline, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "f1": float(f1_score(y, preds)),
        "brier": float(brier_score_loss(y, proba)),
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", default=None, help="Train from CSV instead of SQL")
    args = ap.parse_args()

    train_df, test_df = _load_data(args.from_csv)
    X_train, y_train = F.split_X_y(train_df)
    X_test, y_test = F.split_X_y(test_df)

    # Handle class imbalance via scale_pos_weight.
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    spw = neg / max(pos, 1.0)

    model = build_model()
    model.named_steps["clf"].set_params(scale_pos_weight=spw)

    # ---- MLflow tracking ----
    metrics = {}
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.experiment_name)
        with mlflow.start_run(run_name=f"xgb-{version}"):
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)
            mlflow.log_params(model.named_steps["clf"].get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name="model")
    except Exception as e:  # MLflow optional; never block a training run
        print(f"[mlflow] skipped tracking: {e}")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)

    # ---- Persist artifact for the API ----
    os.makedirs(settings.model_dir, exist_ok=True)
    bundle = {
        "model": model,
        "version": version,
        "numeric_features": F.NUMERIC_FEATURES,
        "categorical_features": F.CATEGORICAL_FEATURES,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, settings.model_path)

    with open(os.path.join(settings.model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("=== Test metrics ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model -> {settings.model_path} (version {version})")
    print("\nClassification report @0.5:")
    print(classification_report(y_test, (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)))


if __name__ == "__main__":
    main()
