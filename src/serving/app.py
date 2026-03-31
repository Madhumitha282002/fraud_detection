from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection Serving API")


class PredictRequest(BaseModel):
    transaction_id: int
    Amount: float
    Time: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest):
    # Day 11 stub. Real MLflow + Feast serving comes on Day 15.
    fraud_probability = 0.1
    return {
        "transaction_id": payload.transaction_id,
        "fraud_probability": fraud_probability,
        "is_fraud": fraud_probability >= 0.5,
    }