from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = "/opt/project"
PYTHON_BIN = "python3"

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "creditcard.csv")
FEATURE_SCRIPT = "src.feature_engineering.build_features"
TRAIN_SCRIPT = "src.training.train"
PROMOTE_SCRIPT = "src.training.promote_model"

# Replace with your actual best/new run id flow later if needed
# For now, training logs to MLflow and registration/promotion can be split.
REGISTER_SCRIPT = "src.training.register_model"


def run_cmd(cmd: list[str]) -> None:
    result = subprocess.run(
        cmd, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def validate_new_data() -> None:
    run_cmd([PYTHON_BIN, "-m", "src.data_ingestion.validate", RAW_DATA_PATH])


def engineer_features() -> None:
    run_cmd([PYTHON_BIN, "-m", FEATURE_SCRIPT])
    run_cmd(
        [
            PYTHON_BIN,
            "-c",
            "from feast import FeatureStore; "
            "store=FeatureStore(repo_path='src/feature_engineering/feature_store'); "
            "print('Feast repo loaded successfully')",
        ]
    )
    run_cmd(
        [
            "bash",
            "-c",
            "cd src/feature_engineering/feature_store && "
            "feast apply && "
            "feast materialize 2020-01-01T00:00:00 2030-01-01T00:00:00",
        ]
    )


def train_model() -> None:
    run_cmd([PYTHON_BIN, "-m", TRAIN_SCRIPT])


def evaluate_model() -> None:
    # Day 9 wants staging vs production comparison on test set.
    # Your Day 7 script already does that logic.
    run_cmd([PYTHON_BIN, "-m", PROMOTE_SCRIPT])


def promote_or_reject() -> None:
    # For your current setup, promote_model.py already handles:
    # - compare staging vs production
    # - promote only if better on AUC-PR
    # So this task can call the same script or later be split more cleanly.
    run_cmd([PYTHON_BIN, "-m", PROMOTE_SCRIPT])


def notify() -> None:
    print("Retraining DAG completed. Check Airflow + MLflow for run details.")


default_args = {
    "owner": "madhumitha",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="fraud_detection_retraining",
    default_args=default_args,
    description="Daily retraining pipeline for fraud detection",
    start_date=datetime(2026, 3, 1),
    schedule="@daily",
    catchup=False,
    tags=["fraud", "retraining", "mlops"],
) as dag:

    task_validate_new_data = PythonOperator(
        task_id="validate_new_data",
        python_callable=validate_new_data,
    )

    task_engineer_features = PythonOperator(
        task_id="engineer_features",
        python_callable=engineer_features,
    )

    task_train_model = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    task_evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    task_promote_or_reject = PythonOperator(
        task_id="promote_or_reject",
        python_callable=promote_or_reject,
    )

    task_notify = PythonOperator(
        task_id="notify",
        python_callable=notify,
    )

    (
        task_validate_new_data
        >> task_engineer_features
        >> task_train_model
        >> task_evaluate_model
        >> task_promote_or_reject
        >> task_notify
    )
