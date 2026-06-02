"""Feature engineering shared by training and serving.

The engineered ratios here mirror the SQL view `curated.vw_model_features`, so a
single raw applicant record produces identical features whether it comes from the
database (batch training) or from a live API request (online scoring).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "loan_amount", "term_months", "int_rate", "annual_income", "age",
    "employment_length_yrs", "dti", "credit_score", "revol_util",
    "open_accounts", "total_accounts", "delinq_2yrs", "inquiries_6m", "pub_rec",
    "loan_to_income", "installment_proxy", "open_ratio",
]

CATEGORICAL_FEATURES = [
    "grade", "loan_purpose", "home_ownership", "verification_status", "score_band",
]

TARGET = "target"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the same engineered columns the SQL view creates.

    Safe to call on data that already has them (e.g. read from the view) — it will
    simply recompute. Used directly for online scoring from raw applicant fields.
    """
    out = df.copy()
    income = out["annual_income"].replace(0, np.nan)
    out["loan_to_income"] = out["loan_amount"] / income
    out["installment_proxy"] = out["loan_amount"] / out["term_months"].replace(0, np.nan)
    out["open_ratio"] = out["open_accounts"] / out["total_accounts"].replace(0, np.nan)

    def band(s):
        if s >= 740:
            return "prime"
        if s >= 670:
            return "near_prime"
        if s >= 580:
            return "subprime"
        return "deep_subprime"

    out["score_band"] = out["credit_score"].apply(band)
    return out


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numerics, impute + one-hot encode categoricals."""
    numeric = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def split_X_y(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(int) if TARGET in df.columns else None
    return X, y
