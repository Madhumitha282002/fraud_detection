from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from mlflow.models import infer_signature
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
)

import mlflow
from src.training.utils import (
    FEATURE_NAMES,
    compute_metrics,
    compute_scale_pos_weight,
    get_project_root,
    get_X_y,
    load_training_dataframe,
    stratified_split,
)

EXPERIMENT_NAME = "fraud-detection"
RANDOM_STATE = 42
ARTIFACT_DIR_NAME = "artifacts"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix(metrics: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Fraud"])
    ax.set_yticklabels(["Legit", "Fraud"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_roc_curve(y_true, y_prob, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_pr_curve(y_true, y_prob, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_feature_importance(
    model: xgb.XGBClassifier, feature_names: list[str], out_path: Path
) -> None:
    booster = model.get_booster()
    scores = booster.get_score(importance_type="gain")
    ranked = [(name, float(scores.get(name, 0.0))) for name in feature_names]
    ranked.sort(key=lambda x: x[1], reverse=True)

    labels = [x[0] for x in ranked]
    values = [x[1] for x in ranked]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, values)
    ax.invert_yaxis()
    ax.set_title("Feature Importance (gain)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_training_dataframe()
    splits = stratified_split(df, random_state=RANDOM_STATE)

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)
    X_test, y_test = get_X_y(test_df)

    scale_pos_weight = compute_scale_pos_weight(y_train)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": RANDOM_STATE,
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "n_jobs": -1,
    }

    root = get_project_root()
    artifact_dir = root / ARTIFACT_DIR_NAME
    ensure_dir(artifact_dir)

    with mlflow.start_run(run_name="xgboost-baseline"):
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("val_size", len(val_df))
        mlflow.log_param("test_size", len(test_df))
        mlflow.log_param("num_features", len(FEATURE_NAMES))
        mlflow.log_param("feature_list", json.dumps(FEATURE_NAMES))

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, val_prob, threshold=0.5)

        for name, value in val_metrics.items():
            mlflow.log_metric(f"val_{name}", value)

        test_prob = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, test_prob, threshold=0.5)
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)

        report_text = classification_report(
            y_val,
            (val_prob >= 0.5).astype(int),
            digits=4,
            zero_division=0,
        )
        report_path = artifact_dir / "classification_report.txt"
        report_path.write_text(report_text)
        mlflow.log_artifact(str(report_path))

        cm_path = artifact_dir / "confusion_matrix.png"
        roc_path = artifact_dir / "roc_curve.png"
        pr_path = artifact_dir / "pr_curve.png"
        fi_path = artifact_dir / "feature_importance.png"

        save_confusion_matrix(val_metrics, cm_path)
        save_roc_curve(y_val, val_prob, roc_path)
        save_pr_curve(y_val, val_prob, pr_path)
        save_feature_importance(model, FEATURE_NAMES, fi_path)

        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(roc_path))
        mlflow.log_artifact(str(pr_path))
        mlflow.log_artifact(str(fi_path))

        signature = infer_signature(X_train, model.predict_proba(X_train)[:, 1])
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(5),
        )

        print("Validation metrics:")
        print(json.dumps(val_metrics, indent=2))
        print("\nTest metrics:")
        print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
