"""Quick exploratory analysis + feature importance.

Run after training:  python notebooks/explore.py
Writes a feature-importance summary to stdout. Designed to be pasted into a
Jupyter notebook cell-by-cell if you prefer interactive exploration.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings  # noqa: E402

CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "credit_risk.csv")


def main():
    df = pd.read_csv(CSV)
    print("Shape:", df.shape)
    print("\nDefault rate:", round(df["default"].mean(), 4))
    print("\nDefault rate by grade:")
    print(df.groupby("grade")["default"].agg(["mean", "count"]).round(3))
    print("\nDefault rate by home ownership:")
    print(df.groupby("home_ownership")["default"].mean().round(3))
    print("\nNumeric correlation with default:")
    num = df.select_dtypes("number").drop(columns=["member_id"])
    print(num.corr()["default"].drop("default").sort_values().round(3))

    if os.path.exists(settings.model_path):
        bundle = joblib.load(settings.model_path)
        model = bundle["model"]
        prep = model.named_steps["prep"]
        clf = model.named_steps["clf"]
        names = prep.get_feature_names_out()
        imp = clf.feature_importances_
        order = np.argsort(imp)[::-1][:15]
        print("\nTop 15 model feature importances:")
        for i in order:
            print(f"  {names[i]:<40} {imp[i]:.4f}")
    else:
        print("\n(Train a model first to see feature importances.)")


if __name__ == "__main__":
    main()
