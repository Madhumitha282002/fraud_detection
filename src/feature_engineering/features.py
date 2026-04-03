from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

SECONDS_IN_HOUR = 3600
SECONDS_IN_DAY = 86400


@dataclass(frozen=True)
class AmountStats:
    mean: float
    std: float


def validate_input_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_transaction_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "transaction_id" not in out.columns:
        out["transaction_id"] = np.arange(len(out), dtype=np.int64)
    return out


def add_amount_log(df: pd.DataFrame, amount_col: str = "Amount") -> pd.DataFrame:
    validate_input_columns(df, [amount_col])
    out = df.copy()
    out["amount_log"] = np.log1p(out[amount_col].clip(lower=0))
    return out


def compute_amount_stats(df: pd.DataFrame, amount_col: str = "Amount") -> AmountStats:
    validate_input_columns(df, [amount_col])
    mean = float(df[amount_col].mean())
    std = float(df[amount_col].std(ddof=0))
    if std == 0:
        std = 1.0
    return AmountStats(mean=mean, std=std)


def add_amount_zscore(
    df: pd.DataFrame,
    stats: AmountStats | None = None,
    amount_col: str = "Amount",
) -> pd.DataFrame:
    validate_input_columns(df, [amount_col])
    out = df.copy()
    stats = stats or compute_amount_stats(out, amount_col=amount_col)
    out["amount_zscore"] = (out[amount_col] - stats.mean) / stats.std
    return out


def add_hour_features(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    validate_input_columns(df, [time_col])
    out = df.copy()

    hour_of_day = ((out[time_col] % SECONDS_IN_DAY) // SECONDS_IN_HOUR).astype(int)
    out["hour_of_day"] = hour_of_day

    angle = 2 * np.pi * out["hour_of_day"] / 24.0
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)
    return out


def add_time_since_last_tx(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    validate_input_columns(df, [time_col])
    out = df.copy().sort_values(time_col).reset_index(drop=True)
    out["time_since_last_tx"] = out[time_col].diff().fillna(0).clip(lower=0)
    return out


def _rolling_count(times: pd.Series, window_seconds: int) -> pd.Series:
    values = times.to_numpy()
    result = np.zeros(len(values), dtype=np.int64)

    left = 0
    for right in range(len(values)):
        while values[right] - values[left] > window_seconds:
            left += 1
        result[right] = right - left + 1
    return pd.Series(result, index=times.index)


def _rolling_mean(
    times: pd.Series, amounts: pd.Series, window_seconds: int
) -> pd.Series:
    t = times.to_numpy()
    a = amounts.to_numpy(dtype=float)
    result = np.zeros(len(t), dtype=float)

    left = 0
    running_sum = 0.0
    for right in range(len(t)):
        running_sum += a[right]
        while t[right] - t[left] > window_seconds:
            running_sum -= a[left]
            left += 1
        count = right - left + 1
        result[right] = running_sum / count
    return pd.Series(result, index=times.index)


def add_rolling_features(
    df: pd.DataFrame,
    time_col: str = "Time",
    amount_col: str = "Amount",
) -> pd.DataFrame:
    validate_input_columns(df, [time_col, amount_col])
    out = df.copy().sort_values(time_col).reset_index(drop=True)

    out["tx_frequency_1h"] = _rolling_count(out[time_col], 1 * SECONDS_IN_HOUR)
    out["tx_frequency_6h"] = _rolling_count(out[time_col], 6 * SECONDS_IN_HOUR)
    out["tx_frequency_24h"] = _rolling_count(out[time_col], 24 * SECONDS_IN_HOUR)

    out["amount_mean_1h"] = _rolling_mean(
        out[time_col], out[amount_col], 1 * SECONDS_IN_HOUR
    )
    out["amount_mean_24h"] = _rolling_mean(
        out[time_col], out[amount_col], 24 * SECONDS_IN_HOUR
    )
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end pure feature pipeline.
    Input: raw transaction dataframe
    Output: dataframe with engineered features added
    """
    validate_input_columns(df, ["Time", "Amount"])

    out = df.copy()
    out = add_transaction_id(out)
    out = add_amount_log(out)
    out = add_amount_zscore(out)
    out = add_hour_features(out)
    out = add_time_since_last_tx(out)
    out = add_rolling_features(out)

    feature_cols = [
        "transaction_id",
        "amount_log",
        "amount_zscore",
        "hour_of_day",
        "hour_sin",
        "hour_cos",
        "tx_frequency_1h",
        "tx_frequency_6h",
        "tx_frequency_24h",
        "amount_mean_1h",
        "amount_mean_24h",
        "time_since_last_tx",
    ]

    remaining = [c for c in out.columns if c not in feature_cols]
    return out[remaining + feature_cols]
