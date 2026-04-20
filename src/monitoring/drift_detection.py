from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
import structlog

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
except ImportError:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

from src.configs import load_yaml

logger = structlog.get_logger()

monitoring_cfg = load_yaml("configs/monitoring_config.yaml")


def load_reference_data() -> pd.DataFrame:
    reference_path = Path(monitoring_cfg["reference_data_path"])
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference dataset not found: {reference_path}")
    return pd.read_parquet(reference_path)


def load_recent_predictions() -> pd.DataFrame:
    db_path = monitoring_cfg["prediction_log_db_path"]
    limit = monitoring_cfg.get("recent_predictions_limit", 1000)

    with sqlite3.connect(db_path) as conn:
        query = f"""
            SELECT *
            FROM predictions
            WHERE payload_json IS NOT NULL
              AND payload_json != '{{}}'
            ORDER BY timestamp DESC
            LIMIT {limit}
        """
        df = pd.read_sql_query(query, conn)

    if df.empty:
        raise ValueError(
            "No recent predictions with usable payload_json found in the predictions table."
        )

    return df


def prepare_current_features(df: pd.DataFrame) -> pd.DataFrame:
    if "payload_json" not in df.columns:
        raise ValueError("payload_json column not found in prediction logs.")

    payload_rows: list[dict] = []

    for raw_payload in df["payload_json"]:
        if not raw_payload:
            continue

        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            continue

        if payload:
            payload_rows.append(payload)

    if not payload_rows:
        raise ValueError("No usable payload_json records found in prediction logs.")

    payload_df = pd.DataFrame(payload_rows)

    if payload_df.empty:
        raise ValueError("Parsed payload_json records produced an empty DataFrame.")

    return payload_df


def compute_drift_report() -> tuple[float, str]:
    reference_df = load_reference_data()
    current_logs = load_recent_predictions()
    current_df = prepare_current_features(current_logs)

    print(f"Loaded reference data: {reference_df.shape}")
    print(f"Loaded current logs: {current_logs.shape}")
    print(f"Prepared current features: {current_df.shape}")

    common_columns = [col for col in current_df.columns if col in reference_df.columns]
    if not common_columns:
        raise ValueError(
            "No overlapping columns between reference data and current data. "
            f"Current columns: {list(current_df.columns)}"
        )

    reference_subset = reference_df[common_columns].copy()
    current_subset = current_df[common_columns].copy()

    reference_subset = reference_subset.sample(
        n=min(len(reference_subset), 5000),
        random_state=42,
    )
    current_subset = current_subset.sample(
        n=min(len(current_subset), 1000),
        random_state=42,
    )

    print(f"Reference subset shape: {reference_subset.shape}")
    print(f"Current subset shape: {current_subset.shape}")
    print("Running Evidently drift report...")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_subset, current_data=current_subset)

    reports_dir = Path(monitoring_cfg["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    report_path = reports_dir / f"data_drift_report_{timestamp}.html"
    report.save_html(str(report_path))

    drift_score = 0.0
    report_dict = report.as_dict()
    metrics = report_dict.get("metrics", [])

    for metric in metrics:
        result = metric.get("result", {})
        if "number_of_drifted_columns" in result and "number_of_columns" in result:
            n_drifted = result.get("number_of_drifted_columns", 0)
            n_total = result.get("number_of_columns", 1)
            drift_score = n_drifted / max(n_total, 1)
            break

    logger.info(
        "drift_report_generated",
        drift_score=drift_score,
        report_path=str(report_path),
    )

    return drift_score, str(report_path)


if __name__ == "__main__":
    score, path = compute_drift_report()
    print(f"Drift score: {score}")
    print(f"Report saved to: {path}")
