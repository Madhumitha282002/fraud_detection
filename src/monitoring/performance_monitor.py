from __future__ import annotations

import pandas as pd
import structlog
from evidently import Report
from evidently.metric_preset import ClassificationPreset
from sklearn.metrics import f1_score, precision_score, recall_score

logger = structlog.get_logger()


def compute_live_metrics(df: pd.DataFrame) -> dict:
    """
    Expected columns:
    - y_true
    - y_pred
    - fraud_probability
    """
    precision = precision_score(df["y_true"], df["y_pred"], zero_division=0)
    recall = recall_score(df["y_true"], df["y_pred"], zero_division=0)
    f1 = f1_score(df["y_true"], df["y_pred"], zero_division=0)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    logger.info("live_performance_metrics_computed", **metrics)
    return metrics


def generate_classification_report(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, output_path: str
) -> None:
    report = Report(metrics=[ClassificationPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html(output_path)

    logger.info("classification_report_generated", output_path=output_path)
