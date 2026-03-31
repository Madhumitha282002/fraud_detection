from __future__ import annotations

import json

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.training.utils import (
    compute_metrics,
    get_X_y,
    load_training_dataframe,
    stratified_split,
)

EXPERIMENT_NAME = "fraud-detection"
TRACKING_URI = "http://localhost:5001"


def log_model_run(model_name: str, model, X_train, y_train, X_val, y_val, params: dict) -> dict:
    with mlflow.start_run(run_name=f"compare_{model_name}"):
        mlflow.log_param("model_type", model_name)
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        y_val_prob = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_val_prob, threshold=0.5)

        for k, v in metrics.items():
            mlflow.log_metric(f"val_{k}", v)

        mlflow.log_param("feature_list", json.dumps(list(X_train.columns)))

        if model_name == "lightgbm":
            mlflow.lightgbm.log_model(model, name="model")
        else:
            mlflow.sklearn.log_model(model, name="model")

        print(f"\n{model_name} validation metrics")
        print(json.dumps(metrics, indent=2))
        return metrics


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_training_dataframe()
    splits = stratified_split(df, random_state=42)

    X_train, y_train = get_X_y(splits["train"])
    X_val, y_val = get_X_y(splits["val"])

    # Replace these with tuned params from MLflow/Optuna once available
    xgb_best_note = "Use the best XGBoost params from the tuning parent run"

    lightgbm_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": 42,
    }
    lgb_model = lgb.LGBMClassifier(**lightgbm_params)
    log_model_run("lightgbm", lgb_model, X_train, y_train, X_val, y_val, lightgbm_params)

    rf_params = {
        "n_estimators": 300,
        "max_depth": 12,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    rf_model = RandomForestClassifier(**rf_params)
    log_model_run("random_forest", rf_model, X_train, y_train, X_val, y_val, rf_params)

    lr_params = {
        "C": 1.0,
        "class_weight": "balanced",
        "solver": "liblinear",
        "random_state": 42,
        "max_iter": 1000,
    }
    lr_model = LogisticRegression(**lr_params)
    log_model_run("logistic_regression", lr_model, X_train, y_train, X_val, y_val, lr_params)

    print("\nComparison runs logged to MLflow.")
    print(xgb_best_note)


if __name__ == "__main__":
    main()