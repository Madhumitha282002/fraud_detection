from __future__ import annotations

import time

import structlog
from fastapi import FastAPI
from pydantic import BaseModel

from src.configs import load_yaml, settings
from src.core.logging_config import configure_logging

configure_logging()
logger = structlog.get_logger()

model_cfg = load_yaml("configs/model_config.yaml")

app = FastAPI(title="Fraud Detection Serving API")


class PredictRequest(BaseModel):
    transaction_id: int
    Amount: float
    Time: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment": settings.environment,
    }


@app.post("/predict")
def predict(payload: PredictRequest):
    start = time.time()

    # Day 11 stub. Real MLflow + Feast serving comes on Day 15.
    fraud_probability = 0.1
    threshold = model_cfg.get("prediction_threshold", settings.prediction_threshold)
    is_fraud = fraud_probability >= threshold
    latency_ms = round((time.time() - start) * 1000, 2)

    logger.info(
        "prediction",
        transaction_id=payload.transaction_id,
        prediction=is_fraud,
        confidence=fraud_probability,
        latency_ms=latency_ms,
    )

    return {
        "transaction_id": payload.transaction_id,
        "fraud_probability": fraud_probability,
        "is_fraud": is_fraud,
    }
