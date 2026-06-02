"""FastAPI scoring service for credit-risk decisions."""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from src.predict import load_bundle, score_one
from .schemas import Applicant, HealthResponse, ScoreResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("credit-risk-api")

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Probability-of-default scoring and lending decisions.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        bundle = load_bundle()
        return HealthResponse(status="ok", model_version=bundle["version"])
    except Exception as e:  # model not present yet
        log.warning("model not loaded: %s", e)
        return HealthResponse(status="degraded", model_version=None)


@app.get("/metrics")
def metrics() -> dict:
    """Expose the held-out test metrics captured at training time."""
    try:
        return load_bundle()["metrics"]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"model unavailable: {e}")


@app.post("/score", response_model=ScoreResponse)
def score(applicant: Applicant) -> ScoreResponse:
    payload = applicant.model_dump()
    try:
        result = score_one(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"scoring failed: {e}")

    # Best-effort audit write; never fail the request if the DB is down.
    try:
        from src.db import log_prediction
        log_prediction(
            member_id=payload.get("member_id"),
            model_version=result["model_version"],
            pd_score=result["probability_of_default"],
            decision=result["decision"],
            request_payload=payload,
        )
    except Exception as e:
        log.warning("prediction logging skipped: %s", e)

    return ScoreResponse(**result)
