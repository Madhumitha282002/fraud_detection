from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def valid_payload() -> dict:
    payload = {
        "transaction_id": "tx_001",
        "Time": 1000.0,
        "Amount": 125.5,
    }
    for i in range(1, 29):
        payload[f"V{i}"] = float(i) / 10.0
    return payload


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert "status" in body
    assert "uptime_seconds" in body


def test_predict_valid_payload_returns_prediction(client: TestClient) -> None:
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200

    body = response.json()
    assert body["transaction_id"] == "tx_001"
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)
    assert "model_version" in body
    assert body["latency_ms"] >= 0


def test_predict_invalid_payload_returns_422(client: TestClient) -> None:
    bad_payload = {"transaction_id": "tx_001", "Amount": -5.0, "Time": 100.0}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_returns_503_when_model_missing(client: TestClient) -> None:
    original_model = getattr(app.state, "model", None)
    original_version = getattr(app.state, "model_version", None)
    original_stage = getattr(app.state, "model_stage", None)

    app.state.model = None
    app.state.model_version = None
    app.state.model_stage = None

    try:
        response = client.post("/predict", json=valid_payload())
        assert response.status_code == 503
    finally:
        app.state.model = original_model
        app.state.model_version = original_version
        app.state.model_stage = original_stage