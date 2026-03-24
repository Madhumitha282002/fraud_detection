from __future__ import annotations

import argparse
import time

import mlflow
from mlflow import MlflowClient


MODEL_NAME = "fraud-detection-model"
TRACKING_URI = "http://localhost:5001"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-path", default="model")
    parser.add_argument("--dataset-version", default="creditcard.csv:v1")
    parser.add_argument("--feature-version", default="feature_set:v1")
    parser.add_argument("--algorithm", default="xgboost")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    model_uri = f"runs:/{args.run_id}/{args.artifact_path}"

    try:
        client.create_registered_model(MODEL_NAME)
    except Exception:
        pass

    mv = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    # Wait briefly until the version is ready
    for _ in range(20):
        current = client.get_model_version(MODEL_NAME, mv.version)
        if current.status == "READY":
            break
        time.sleep(1)

    client.update_registered_model(
        name=MODEL_NAME,
        description="Real-time fraud detection model trained with Feast features and tracked in MLflow.",
    )

    client.update_model_version(
        name=MODEL_NAME,
        version=mv.version,
        description="Candidate model registered from the best training/comparison run.",
    )

    client.set_registered_model_tag(MODEL_NAME, "task", "fraud-detection")
    client.set_registered_model_tag(MODEL_NAME, "dataset_version", args.dataset_version)
    client.set_registered_model_tag(MODEL_NAME, "feature_version", args.feature_version)

    client.set_model_version_tag(MODEL_NAME, mv.version, "algorithm", args.algorithm)
    client.set_model_version_tag(MODEL_NAME, mv.version, "dataset_version", args.dataset_version)
    client.set_model_version_tag(MODEL_NAME, mv.version, "feature_version", args.feature_version)

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    print(f"Registered {MODEL_NAME} version {mv.version} from {model_uri}")
    print(f"Transitioned version {mv.version} to Staging")


if __name__ == "__main__":
    main()