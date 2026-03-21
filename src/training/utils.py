from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from feast import FeatureStore
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit


FEATURE_REFS = [
    "transaction_features:amount_log",
    "transaction_features:amount_zscore",
    "transaction_features:hour_of_day",
    "transaction_features:hour_sin",
    "transaction_features:hour_cos",
    "transaction_features:tx_frequency_1h",
    "transaction_features:tx_frequency_6h",
    "transaction_features:tx_frequency_24h",
    "transaction_features:amount_mean_1h",
    "transaction_features:amount_mean_24h",
    "transaction_features:time_since_last_tx",
]

FEATURE_NAMES = [f.split(":")[1] for f in FEATURE_REFS]


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_feast_store() -> FeatureStore:
    root = get_project_root()
    repo_path = root / "src" / "feature_engineering" / "feature_store"
    return FeatureStore(repo_path=str(repo_path))



def load_training_dataframe() -> pd.DataFrame:
    root = get_project_root()
    parquet_path = root / "data" / "processed" / "transactions_features.parquet"
    df = pd.read_parquet(parquet_path)

    entity_df = df[["transaction_id", "event_timestamp"]].copy()
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"], utc=True)

    store = get_feast_store()
    historical = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURE_REFS,
    ).to_df()

    merged = historical.merge(
        df[["transaction_id", "Class"]],
        on="transaction_id",
        how="left",
    )

    merged = merged.dropna(subset=["Class"]).reset_index(drop=True)
    merged["Class"] = merged["Class"].astype(int)
    return merged 


def stratified_split(
    df: pd.DataFrame,
    target_col: str = "Class",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    y = df[target_col]

    first_split = StratifiedShuffleSplit(
        n_splits=1, test_size=(1.0 - train_size), random_state=random_state
    )
    train_idx, temp_idx = next(first_split.split(df, y))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    temp_y = temp_df[target_col]
    relative_test_size = test_size / (val_size + test_size)

    second_split = StratifiedShuffleSplit(
        n_splits=1, test_size=relative_test_size, random_state=random_state
    )
    val_idx, test_idx = next(second_split.split(temp_df, temp_y))

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return {"train": train_df, "val": val_df, "test": test_df}


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_NAMES].copy()
    y = df["Class"].copy()
    return X, y


def compute_scale_pos_weight(y: pd.Series) -> float:
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0:
        raise ValueError("No positive samples found in target.")
    return negatives / positives


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "threshold": float(threshold),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update(
        {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    )
    return metrics