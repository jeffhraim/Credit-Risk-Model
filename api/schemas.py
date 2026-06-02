"""Request/response schemas for the scoring API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Applicant(BaseModel):
    member_id: Optional[int] = Field(None, description="Optional applicant id for audit")
    loan_amount: float = Field(..., gt=0)
    term_months: int = Field(..., ge=12, le=84)
    int_rate: float = Field(..., ge=0, le=40)
    annual_income: float = Field(..., gt=0)
    age: int = Field(..., ge=18, le=100)
    employment_length_yrs: Optional[float] = Field(None, ge=0, le=60)
    dti: Optional[float] = Field(None, ge=0, le=100)
    credit_score: int = Field(..., ge=300, le=850)
    revol_util: Optional[float] = Field(None, ge=0, le=200)
    open_accounts: int = Field(..., ge=0)
    total_accounts: int = Field(..., ge=0)
    delinq_2yrs: int = Field(..., ge=0)
    inquiries_6m: int = Field(..., ge=0)
    pub_rec: int = Field(..., ge=0)
    grade: str
    loan_purpose: str
    home_ownership: str
    verification_status: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "member_id": 900001,
                "loan_amount": 15000, "term_months": 36, "int_rate": 9.5,
                "annual_income": 85000, "age": 41, "employment_length_yrs": 8,
                "dti": 14.2, "credit_score": 760, "revol_util": 22.0,
                "open_accounts": 9, "total_accounts": 22, "delinq_2yrs": 0,
                "inquiries_6m": 1, "pub_rec": 0, "grade": "A",
                "loan_purpose": "credit_card", "home_ownership": "MORTGAGE",
                "verification_status": "Verified",
            }
        }
    }


class ScoreResponse(BaseModel):
    probability_of_default: float
    decision: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: Optional[str] = None
