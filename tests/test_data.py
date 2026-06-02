"""Unit tests for data generation and feature engineering."""
import numpy as np
import pandas as pd

from data.generate_data import generate
from src import features as F


def test_generate_shape_and_target():
    df = generate(2000, seed=1)
    assert len(df) == 2000
    assert set(["member_id", "default"]).issubset(df.columns)
    # Target is binary.
    assert set(df["default"].unique()).issubset({0, 1})
    # Default rate in a plausible credit range.
    assert 0.05 < df["default"].mean() < 0.30


def test_generate_is_deterministic():
    a = generate(500, seed=7)
    b = generate(500, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_signal_present():
    # Lower credit scores should default more often than higher ones.
    df = generate(5000, seed=3)
    low = df[df["credit_score"] < 600]["default"].mean()
    high = df[df["credit_score"] > 750]["default"].mean()
    assert low > high


def test_engineered_features():
    df = generate(100, seed=5).rename(columns={"default": "target"})
    out = F.add_engineered_features(df)
    for col in ["loan_to_income", "installment_proxy", "open_ratio", "score_band"]:
        assert col in out.columns
    assert out["score_band"].isin(
        ["prime", "near_prime", "subprime", "deep_subprime"]
    ).all()


def test_split_X_y():
    df = generate(100, seed=5).rename(columns={"default": "target"})
    df = F.add_engineered_features(df)
    X, y = F.split_X_y(df)
    assert list(X.columns) == F.NUMERIC_FEATURES + F.CATEGORICAL_FEATURES
    assert y is not None and len(y) == 100
