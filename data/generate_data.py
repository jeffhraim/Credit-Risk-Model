"""
Synthetic but realistic credit-risk dataset generator.

The goal is NOT random noise. Default probability is driven by genuine,
economically sensible relationships (debt-to-income, credit score, loan-to-income,
recent delinquencies, employment length, etc.) passed through a logistic link with
noise, so a model has real signal to learn while still resembling production data
(missing values, skewed distributions, mild class imbalance).

Output: data/credit_risk.csv  (default ~50,000 rows)

Usage:
    python data/generate_data.py --rows 50000 --seed 42
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

RNG_DEFAULT_SEED = 42

HOME_OWNERSHIP = ["RENT", "MORTGAGE", "OWN", "OTHER"]
HOME_OWNERSHIP_P = [0.45, 0.40, 0.12, 0.03]

LOAN_PURPOSE = [
    "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
    "medical", "car", "small_business", "vacation", "moving", "other",
]
LOAN_PURPOSE_P = [0.42, 0.22, 0.09, 0.06, 0.04, 0.04, 0.04, 0.03, 0.02, 0.04]

GRADES = ["A", "B", "C", "D", "E", "F", "G"]
VERIFICATION = ["Verified", "Source Verified", "Not Verified"]
TERMS = [36, 60]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(rows: int, seed: int = RNG_DEFAULT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ----- Applicant demographics & finances -----
    age = np.clip(rng.normal(42, 12, rows).round(), 18, 85).astype(int)

    # Income is right-skewed (log-normal), correlated weakly with age.
    base_income = rng.lognormal(mean=10.9, sigma=0.55, size=rows)
    income = np.clip(base_income + (age - 42) * 250, 12_000, 500_000).round(-2)

    employment_length = np.clip(
        rng.gamma(shape=2.0, scale=3.0, size=rows).round(), 0, 40
    ).astype(int)

    home = rng.choice(HOME_OWNERSHIP, size=rows, p=HOME_OWNERSHIP_P)
    purpose = rng.choice(LOAN_PURPOSE, size=rows, p=LOAN_PURPOSE_P)
    term = rng.choice(TERMS, size=rows, p=[0.72, 0.28])
    verification = rng.choice(VERIFICATION, size=rows, p=[0.4, 0.35, 0.25])

    # ----- Loan characteristics -----
    # Loan amount scales with income but with wide spread.
    loan_amount = np.clip(
        income * rng.uniform(0.05, 0.6, rows) + rng.normal(0, 2000, rows),
        1_000, 60_000,
    ).round(-2)

    loan_to_income = loan_amount / income

    # Credit score: 300-850, left-skewed toward higher scores, nudged by income.
    credit_score = np.clip(
        rng.normal(690, 70, rows) + (np.log(income) - 10.8) * 18,
        300, 850,
    ).round().astype(int)

    # Revolving credit utilization (%) — higher for lower scores.
    revol_util = np.clip(
        rng.normal(48, 24, rows) - (credit_score - 690) * 0.18,
        0, 150,
    ).round(1)

    open_accounts = np.clip(rng.poisson(10, rows), 1, 50).astype(int)
    total_accounts = open_accounts + np.clip(rng.poisson(12, rows), 0, 60).astype(int)

    # Delinquencies in last 2 years — rare, more likely with low score.
    delinq_lambda = np.clip(0.9 - (credit_score - 600) / 250, 0.02, 1.2)
    delinq_2yrs = rng.poisson(delinq_lambda).astype(int)

    inquiries_6m = np.clip(rng.poisson(0.8, rows), 0, 12).astype(int)
    pub_rec = (rng.random(rows) < 0.07).astype(int)

    # Debt-to-income ratio (%): driven by existing debt load.
    dti = np.clip(
        rng.normal(18, 9, rows) + revol_util * 0.12 + loan_to_income * 8,
        0, 60,
    ).round(2)

    # Interest rate roughly follows risk grade / score (lender's own pricing).
    int_rate = np.clip(
        6.0 + (720 - credit_score) * 0.035 + (term == 60) * 1.5
        + rng.normal(0, 1.2, rows),
        5.0, 30.0,
    ).round(2)

    # Assign a grade from interest rate buckets.
    grade_idx = np.clip(((int_rate - 5) / 3.6).astype(int), 0, 6)
    grade = np.array(GRADES)[grade_idx]

    # ----- True default-generating process (latent) -----
    # Standardized contributions. Positive => more likely to default.
    z = (
        -3.10  # intercept -> tunes base default rate
        + 2.30 * (dti / 60.0)
        + 2.60 * (revol_util / 150.0)
        + 1.90 * np.clip(loan_to_income, 0, 1.5)
        - 2.80 * ((credit_score - 300) / 550.0)
        + 0.85 * (delinq_2yrs / 3.0)
        + 0.55 * (inquiries_6m / 12.0)
        + 0.70 * pub_rec
        - 0.45 * (np.minimum(employment_length, 15) / 15.0)
        + 0.60 * (term == 60).astype(float)
        + 0.50 * (int_rate / 30.0)
        - 0.30 * (np.log(income) - 10.8)
        + rng.normal(0, 0.55, rows)  # irreducible noise
    )
    p_default = _sigmoid(z)
    default = (rng.random(rows) < p_default).astype(int)

    # ----- Identifiers & dates -----
    start = datetime(2018, 1, 1)
    issue_date = [
        start + timedelta(days=int(d)) for d in rng.integers(0, 365 * 6, rows)
    ]
    member_id = 100_000 + np.arange(rows)

    df = pd.DataFrame(
        {
            "member_id": member_id,
            "issue_date": issue_date,
            "loan_amount": loan_amount,
            "term_months": term,
            "int_rate": int_rate,
            "grade": grade,
            "loan_purpose": purpose,
            "home_ownership": home,
            "annual_income": income,
            "verification_status": verification,
            "age": age,
            "employment_length_yrs": employment_length,
            "dti": dti,
            "credit_score": credit_score,
            "revol_util": revol_util,
            "open_accounts": open_accounts,
            "total_accounts": total_accounts,
            "delinq_2yrs": delinq_2yrs,
            "inquiries_6m": inquiries_6m,
            "pub_rec": pub_rec,
            "default": default,
        }
    )

    # ----- Inject realistic missingness (NOT in the target) -----
    for col, frac in [("employment_length_yrs", 0.05),
                      ("revol_util", 0.015),
                      ("dti", 0.01)]:
        mask = rng.random(rows) < frac
        df.loc[mask, col] = np.nan

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=RNG_DEFAULT_SEED)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, "credit_risk.csv")

    df = generate(args.rows, args.seed)
    df.to_csv(out, index=False)

    rate = df["default"].mean()
    print(f"Wrote {len(df):,} rows -> {out}")
    print(f"Default rate: {rate:.3%}")
    print(f"Columns: {list(df.columns)}")
    print(df.describe(include='all').T[['count']].head(25))


if __name__ == "__main__":
    main()
