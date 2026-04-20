from __future__ import annotations

import requests
import structlog

from src.configs import load_yaml, settings
from src.monitoring.drift_detection import compute_drift_report

logger = structlog.get_logger()

monitoring_cfg = load_yaml("configs/monitoring_config.yaml")


def trigger_retraining(dry_run: bool = False) -> None:
    dag_id = monitoring_cfg["airflow_dag_id"]

    if dry_run:
        logger.info("retraining_triggered_mock", dag_id=dag_id)
        return

    url = f"{settings.airflow_base_url}/dags/{dag_id}/dagRuns"

    response = requests.post(
        url,
        auth=(settings.airflow_username, settings.airflow_password),
        json={"conf": {"triggered_by": "drift_alert"}},
        timeout=30,
    )
    response.raise_for_status()

    logger.info(
        "retraining_triggered",
        dag_id=dag_id,
        status_code=response.status_code,
    )


def run_alert_check(dry_run: bool = False) -> None:
    drift_threshold = monitoring_cfg["drift_threshold"]

    drift_score, report_path = compute_drift_report()

    logger.info(
        "drift_check_completed",
        drift_score=drift_score,
        drift_threshold=drift_threshold,
        report_path=report_path,
    )

    if drift_score > drift_threshold:
        logger.warning(
            "drift_threshold_exceeded",
            drift_score=drift_score,
            drift_threshold=drift_threshold,
        )
        trigger_retraining(dry_run=dry_run)
    else:
        logger.info(
            "drift_threshold_not_exceeded",
            drift_score=drift_score,
            drift_threshold=drift_threshold,
        )


if __name__ == "__main__":
    run_alert_check(dry_run=True)
