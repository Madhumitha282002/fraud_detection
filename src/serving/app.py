from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest

from src.configs import load_yaml, settings
from src.core.logging_config import configure_logging
from src.serving.schemas import HealthResponse, PredictionResponse, TransactionInput

configure_logging()
logger = structlog.get_logger()

model_cfg = load_yaml("configs/model_config.yaml")

app = FastAPI(title="Fraud Detection Serving API")
APP_START_TIME = time.time()
PREDICTIONS_DB_PATH = Path("data/predictions/predictions.db")

PREDICTION_COUNT = Counter(
    "prediction_count",
    "Total number of predictions served",
    ["label"],
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latency of prediction requests in seconds",
)

MODEL_VERSION_INFO = Gauge(
    "model_version_info",
    "Current loaded model version",
    ["model_version", "model_stage"],
)

DRIFT_SCORE_GAUGE = Gauge(
    "drift_score",
    "Latest computed drift score",
)


def ensure_prediction_log_table() -> None:
    PREDICTIONS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(PREDICTIONS_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                transaction_id TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                fraud_probability REAL NOT NULL,
                is_fraud INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                latency_ms REAL NOT NULL
            )
            """
        )

        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()
        }
        if "payload_json" not in columns:
            conn.execute(
                """
                ALTER TABLE predictions
                ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}'
                """
            )

        conn.commit()


def log_prediction(
    transaction_id: str,
    payload_dict: dict[str, Any],
    fraud_probability: float,
    is_fraud: bool,
    model_version: str,
    latency_ms: float,
) -> None:
    payload_json = json.dumps(payload_dict, sort_keys=True)
    input_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

    with sqlite3.connect(PREDICTIONS_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                timestamp,
                transaction_id,
                input_hash,
                payload_json,
                fraud_probability,
                is_fraud,
                model_version,
                latency_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                transaction_id,
                input_hash,
                payload_json,
                fraud_probability,
                int(is_fraud),
                model_version,
                latency_ms,
            ),
        )
        conn.commit()


def load_model_stub() -> str:
    """
    Temporary stub for Day 15/16.
    Replace this with MLflow Production model loading later.
    """
    return "stub_model"


@app.on_event("startup")
def startup_event() -> None:
    ensure_prediction_log_table()

    try:
        app.state.model = load_model_stub()
        app.state.model_version = "stub-v1"
        app.state.model_stage = "Production"

        MODEL_VERSION_INFO.labels(
            model_version=app.state.model_version,
            model_stage=app.state.model_stage,
        ).set(1)
        DRIFT_SCORE_GAUGE.set(0.0)

        logger.info(
            "app_started",
            environment=settings.environment,
            model_version=app.state.model_version,
            model_stage=app.state.model_stage,
        )
    except Exception as e:
        app.state.model = None
        app.state.model_version = None
        app.state.model_stage = None

        logger.exception("model_load_failed", error=str(e))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    uptime_seconds = round(time.time() - APP_START_TIME, 2)
    model_loaded = getattr(app.state, "model", None) is not None

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_version=getattr(app.state, "model_version", None),
        model_stage=getattr(app.state, "model_stage", None),
        uptime_seconds=uptime_seconds,
    )


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    return {
        "model_name": settings.mlflow_model_name,
        "model_version": getattr(app.state, "model_version", None),
        "model_stage": getattr(app.state, "model_stage", None),
        "environment": settings.environment,
        "prediction_threshold": model_cfg.get(
            "prediction_threshold",
            settings.prediction_threshold,
        ),
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type="text/plain")


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TransactionInput) -> PredictionResponse:
    start = time.time()

    model = getattr(app.state, "model", None)
    if model is None:
        logger.error("model_unavailable")
        raise HTTPException(status_code=503, detail="Model not available")

    payload_dict = payload.model_dump()

    # Temporary inference stub for Day 15/16.
    # Replace later with Feast feature fetch + MLflow model.predict().
    fraud_probability = 0.1
    threshold = model_cfg.get("prediction_threshold", settings.prediction_threshold)
    is_fraud = fraud_probability >= threshold
    latency_ms = round((time.time() - start) * 1000, 2)

    log_prediction(
        transaction_id=payload.transaction_id,
        payload_dict=payload_dict,
        fraud_probability=fraud_probability,
        is_fraud=is_fraud,
        model_version=app.state.model_version,
        latency_ms=latency_ms,
    )

    PREDICTION_COUNT.labels(label=str(is_fraud).lower()).inc()
    PREDICTION_LATENCY.observe(latency_ms / 1000.0)

    logger.info(
        "prediction",
        transaction_id=payload.transaction_id,
        prediction=is_fraud,
        confidence=fraud_probability,
        latency_ms=latency_ms,
        model_version=app.state.model_version,
    )

    return PredictionResponse(
        transaction_id=payload.transaction_id,
        fraud_probability=fraud_probability,
        is_fraud=is_fraud,
        model_version=app.state.model_version,
        latency_ms=latency_ms,
    )
