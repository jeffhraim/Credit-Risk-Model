"""Database access layer for SQL Server."""
from __future__ import annotations

import json
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import settings

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    """Lazily create and cache a SQLAlchemy engine for SQL Server."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.sqlalchemy_url,
            fast_executemany=True,   # bulk insert acceleration for pyodbc
            pool_pre_ping=True,
        )
    return _engine


def load_csv_to_raw(csv_path: str, chunksize: int = 10_000) -> int:
    """Bulk-load the generated CSV into raw.loan_applications. Returns row count."""
    engine = get_engine()
    total = 0
    # Truncate first so re-runs are idempotent.
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE raw.loan_applications;"))

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, parse_dates=["issue_date"]):
        chunk = chunk.rename(columns={"default": "default"})
        chunk.to_sql(
            "loan_applications",
            engine,
            schema="raw",
            if_exists="append",
            index=False,
            method=None,
        )
        total += len(chunk)
    return total


def read_features(split: str = "train") -> pd.DataFrame:
    """Read engineered features from the curated views.

    split: 'train', 'test', or 'all'.
    """
    view = {
        "train": "curated.vw_train",
        "test": "curated.vw_test",
        "all": "curated.vw_model_features",
    }[split]
    return pd.read_sql(text(f"SELECT * FROM {view}"), get_engine())


def log_prediction(member_id, model_version, pd_score, decision, request_payload) -> None:
    """Write a scored request to the audit table."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO curated.prediction_log
                    (member_id, model_version, pd_score, decision, request_json)
                VALUES (:mid, :mv, :pd, :dec, :rj)
                """
            ),
            {
                "mid": member_id,
                "mv": model_version,
                "pd": float(pd_score),
                "dec": decision,
                "rj": json.dumps(request_payload),
            },
        )
