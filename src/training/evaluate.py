from __future__ import annotations

import argparse
import json

import mlflow
import numpy as np
from sklearn.metrics import classification_report

from src.training.utils import (
    compute_metrics,
    get_X_y,
    load_training_dataframe,
    stratified_split,
)


def load_model(model_uri: str):
    return mlflow.pyfunc.load_model(model_uri)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-uri",
        required=True,
        help="Examples: runs:/<run_id>/model or models:/fraud-detection-model/Production",
    )
    parser.add_argument("--tracking-uri", default="http://localhost:5001")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    df = load_training_dataframe()
    splits = stratified_split(df, random_state=42)
    test_df = splits["test"]

    X_test, y_test = get_X_y(test_df)
    model = load_model(args.model_uri)

    y_prob = np.asarray(model.predict(X_test), dtype=float)
    metrics = compute_metrics(y_test, y_prob, threshold=0.5)

    print("Test metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:")
    print(
        classification_report(
            y_test, (y_prob >= 0.5).astype(int), digits=4, zero_division=0
        )
    )

    with mlflow.start_run(run_name="model-evaluation"):
        mlflow.log_param("evaluated_model_uri", args.model_uri)
        for key, value in metrics.items():
            mlflow.log_metric(f"eval_{key}", value)


if __name__ == "__main__":
    main()
