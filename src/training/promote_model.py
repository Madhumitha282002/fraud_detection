from __future__ import annotations

import json

import mlflow
import numpy as np
from mlflow import MlflowClient

from src.training.utils import (
    compute_metrics,
    get_X_y,
    load_training_dataframe,
    stratified_split,
)

TRACKING_URI = "http://localhost:5001"
MODEL_NAME = "fraud-detection-model"


def load_model_from_stage(stage: str):
    uri = f"models:/{MODEL_NAME}/{stage}"
    return mlflow.pyfunc.load_model(uri)


def evaluate_model(model, X_test, y_test) -> dict:
    y_prob = np.asarray(model.predict(X_test), dtype=float)
    return compute_metrics(y_test, y_prob, threshold=0.5)


def get_latest_stage_version(client: MlflowClient, stage: str):
    latest = client.get_latest_versions(MODEL_NAME, stages=[stage])
    if not latest:
        return None
    return latest[0]


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    df = load_training_dataframe()
    splits = stratified_split(df, random_state=42)
    X_test, y_test = get_X_y(splits["test"])

    staging_version = get_latest_stage_version(client, "Staging")
    if staging_version is None:
        raise RuntimeError(f"No Staging version found for model '{MODEL_NAME}'")

    staging_model = load_model_from_stage("Staging")
    staging_metrics = evaluate_model(staging_model, X_test, y_test)

    production_version = get_latest_stage_version(client, "Production")
    production_metrics = None

    with mlflow.start_run(run_name="promotion_decision"):
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("staging_version", staging_version.version)

        for k, v in staging_metrics.items():
            mlflow.log_metric(f"staging_{k}", v)

        promote = False

        if production_version is None:
            promote = True
            mlflow.log_param("decision_reason", "no_existing_production_model")
        else:
            mlflow.log_param("production_version", production_version.version)
            production_model = load_model_from_stage("Production")
            production_metrics = evaluate_model(production_model, X_test, y_test)

            for k, v in production_metrics.items():
                mlflow.log_metric(f"production_{k}", v)

            promote = staging_metrics["auc_pr"] > production_metrics["auc_pr"]

        if promote:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=staging_version.version,
                stage="Production",
                archive_existing_versions=True,
            )
            mlflow.log_param("promotion_decision", "promoted")
            if production_version is not None:
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=production_version.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )
        else:
            mlflow.log_param("promotion_decision", "kept_existing_production")

        print("Staging metrics:")
        print(json.dumps(staging_metrics, indent=2))

        if production_metrics is not None:
            print("\nProduction metrics:")
            print(json.dumps(production_metrics, indent=2))

        print("\nDecision:")
        print(
            "Promoted Staging -> Production" if promote else "Kept current Production"
        )


if __name__ == "__main__":
    main()
