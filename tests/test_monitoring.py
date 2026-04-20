from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.monitoring import alerting, drift_detection
from src.serving.app import app


def test_metrics_endpoint_returns_200() -> None:
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "prediction_latency_seconds" in response.text
        assert "model_version_info" in response.text


def test_drift_report_file_is_created() -> None:
    drift_score, report_path = drift_detection.compute_drift_report()

    assert isinstance(drift_score, float)
    assert 0.0 <= drift_score <= 1.0

    report_file = Path(report_path)
    assert report_file.exists()
    assert report_file.suffix == ".html"


def test_alerting_dry_run_does_not_crash() -> None:
    alerting.run_alert_check(dry_run=True)
