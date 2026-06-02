"""One-shot setup: generate data, create SQL objects, and bulk-load.

Runs the .sql files in order against the configured SQL Server, then loads the
generated CSV into raw.loan_applications.

Usage:
    python scripts/setup_database.py
    python scripts/setup_database.py --rows 50000 --regenerate
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

# Make project importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text  # noqa: E402

from src.config import settings  # noqa: E402
from src.db import load_csv_to_raw  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SQL_DIR = os.path.join(ROOT, "sql")
CSV_PATH = os.path.join(ROOT, "data", "credit_risk.csv")


def _split_batches(sql: str):
    """Split a T-SQL script on GO batch separators."""
    parts = re.split(r"^\s*GO\s*$", sql, flags=re.MULTILINE | re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def _run_sql_file(engine_url: str, path: str, autocommit: bool = True):
    eng = create_engine(engine_url, isolation_level="AUTOCOMMIT" if autocommit else None)
    with open(path, "r", encoding="utf-8") as f:
        script = f.read()
    with eng.connect() as conn:
        for batch in _split_batches(script):
            conn.exec_driver_sql(batch)
    print(f"  ran {os.path.basename(path)}")


def _master_url() -> str:
    """Connection URL pointed at master (for CREATE DATABASE)."""
    from urllib.parse import quote_plus
    params = quote_plus(
        f"DRIVER={{{settings.db_driver}}};SERVER={settings.db_server};"
        f"DATABASE=master;UID={settings.db_user};PWD={settings.db_password};"
        f"TrustServerCertificate={settings.db_trust_cert};Encrypt=yes;"
    )
    return f"mssql+pyodbc:///?odbc_connect={params}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50_000)
    ap.add_argument("--regenerate", action="store_true",
                    help="Regenerate the CSV even if it exists")
    args = ap.parse_args()

    if args.regenerate or not os.path.exists(CSV_PATH):
        print("Generating dataset...")
        from data.generate_data import generate
        generate(args.rows).to_csv(CSV_PATH, index=False)
        print(f"  wrote {CSV_PATH}")

    print("Creating database (master)...")
    _run_sql_file(_master_url(), os.path.join(SQL_DIR, "01_create_database.sql"))

    print("Creating tables & views (CreditRisk)...")
    for f in sorted(glob.glob(os.path.join(SQL_DIR, "0[2-9]_*.sql"))):
        _run_sql_file(settings.sqlalchemy_url, f)

    print("Loading CSV into raw.loan_applications...")
    n = load_csv_to_raw(CSV_PATH)
    print(f"  loaded {n:,} rows")
    print("\nDatabase setup complete.")


if __name__ == "__main__":
    main()
