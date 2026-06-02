"""Inference helpers: load the model bundle and score applicants."""
from __future__ import annotations

import functools

import joblib
import pandas as pd

from . import features as F
from .config import settings


@functools.lru_cache(maxsize=1)
def load_bundle(path: str | None = None) -> dict:
    """Load and cache the trained model bundle from disk."""
    return joblib.load(path or settings.model_path)


def decide(pd_score: float) -> str:
    """Map a probability of default to a lending decision."""
    if pd_score < settings.approve_below:
        return "APPROVE"
    if pd_score > settings.decline_above:
        return "DECLINE"
    return "REVIEW"


def score_one(applicant: dict, path: str | None = None) -> dict:
    """Score a single raw applicant dict -> {pd, decision, model_version}."""
    bundle = load_bundle(path)
    df = pd.DataFrame([applicant])
    df = F.add_engineered_features(df)
    X = df[bundle["numeric_features"] + bundle["categorical_features"]]
    pd_score = float(bundle["model"].predict_proba(X)[:, 1][0])
    return {
        "probability_of_default": round(pd_score, 6),
        "decision": decide(pd_score),
        "model_version": bundle["version"],
    }
