from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict


class TransactionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(..., min_length=1, description="Unique transaction ID")
    Time: float = Field(
        ..., ge=0, description="Seconds elapsed since first transaction"
    )
    Amount: float = Field(..., ge=0, description="Transaction amount")

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: bool
    model_version: str
    latency_ms: float = Field(..., ge=0)


class HealthResponse(BaseModel):
    status: str
    model_version: str | None = None
    model_stage: str | None = None
    uptime_seconds: float = Field(..., ge=0)
