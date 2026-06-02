"""Central configuration, read from environment variables (.env in dev)."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # python-dotenv optional at runtime
    pass


@dataclass(frozen=True)
class Settings:
    # --- SQL Server connection ---
    db_server: str = os.getenv("DB_SERVER", "localhost,1433")
    db_name: str = os.getenv("DB_NAME", "CreditRisk")
    db_user: str = os.getenv("DB_USER", "sa")
    db_password: str = os.getenv("DB_PASSWORD", "Your_strong_Pass123")
    db_driver: str = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")
    db_trust_cert: str = os.getenv("DB_TRUST_CERT", "yes")

    # --- MLflow ---
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT", "credit-risk")

    # --- Model artifact ---
    model_dir: str = os.getenv("MODEL_DIR", "models")
    model_name: str = os.getenv("MODEL_NAME", "credit_risk_model.joblib")

    # --- Decision policy thresholds (probability of default) ---
    approve_below: float = float(os.getenv("APPROVE_BELOW", "0.15"))
    decline_above: float = float(os.getenv("DECLINE_ABOVE", "0.45"))

    @property
    def sqlalchemy_url(self) -> str:
        from urllib.parse import quote_plus
        params = quote_plus(
            f"DRIVER={{{self.db_driver}}};"
            f"SERVER={self.db_server};"
            f"DATABASE={self.db_name};"
            f"UID={self.db_user};"
            f"PWD={self.db_password};"
            f"TrustServerCertificate={self.db_trust_cert};"
            f"Encrypt=yes;"
        )
        return f"mssql+pyodbc:///?odbc_connect={params}"

    @property
    def model_path(self) -> str:
        return os.path.join(self.model_dir, self.model_name)


settings = Settings()
