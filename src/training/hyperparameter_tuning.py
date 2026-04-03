from __future__ import annotations

import argparse
import json

import mlflow.xgboost
import optuna
import xgboost as xgb
from optuna.pruners import MedianPruner
from sklearn.metrics import average_precision_score

import mlflow
from src.training.utils import (
    FEATURE_NAMES,
    get_X_y,
    load_training_dataframe,
    stratified_split,
)

EXPERIMENT_NAME = "fraud-detection"
TRACKING_URI = "http://localhost:5001"


def build_objective(X_train, y_train, X_val, y_val):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": 42,
            "n_jobs": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 100, 600),
        }

        with mlflow.start_run(
            run_name=f"xgb_trial_{trial.number}",
            nested=True,
        ):
            mlflow.log_param("model_type", "xgboost")
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_params(params)

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_val_prob = model.predict_proba(X_val)[:, 1]
            auc_pr = float(average_precision_score(y_val, y_val_prob))

            mlflow.log_metric("val_auc_pr", auc_pr)
            mlflow.log_metric("val_num_rows", len(X_val))
            mlflow.log_param("feature_list", json.dumps(FEATURE_NAMES))

            trial.report(auc_pr, step=0)
            if trial.should_prune():
                mlflow.set_tag("trial_status", "pruned")
                raise optuna.TrialPruned()

            mlflow.set_tag("trial_status", "completed")
            return auc_pr

    return objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_training_dataframe()
    splits = stratified_split(df, random_state=42)

    X_train, y_train = get_X_y(splits["train"])
    X_val, y_val = get_X_y(splits["val"])

    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        study_name="xgboost_optuna_tuning",
    )

    with mlflow.start_run(run_name="tuning_xgboost_parent"):
        mlflow.set_tag("run_type", "hyperparameter_tuning")
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("n_trials", args.n_trials)

        objective = build_objective(X_train, y_train, X_val, y_val)
        study.optimize(objective, n_trials=args.n_trials)

        mlflow.log_metric("best_val_auc_pr", study.best_value)
        for key, value in study.best_params.items():
            mlflow.log_param(f"best_{key}", value)

        print("Best trial:")
        print(f"  value (val_auc_pr): {study.best_value:.6f}")
        print("  params:")
        for k, v in study.best_params.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
