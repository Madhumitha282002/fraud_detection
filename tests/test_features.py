
import numpy as np
import pandas as pd

from src.feature_engineering.features import (
    add_amount_log,
    add_amount_zscore,
    add_hour_features,
    add_rolling_features,
    add_time_since_last_tx,
    build_features,
    compute_amount_stats,
)


def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": [0, 1800, 3600, 7200, 90000],
            "Amount": [0.0, 10.0, 20.0, 30.0, 40.0],
            "Class": [0, 0, 1, 0, 0],
        }
    )


def test_amount_log_handles_zero():
    df = pd.DataFrame({"Amount": [0.0, 9.0]})
    out = add_amount_log(df)
    assert out["amount_log"].iloc[0] == 0.0
    assert np.isclose(out["amount_log"].iloc[1], np.log1p(9.0))


def test_amount_zscore_uses_global_stats():
    df = pd.DataFrame({"Amount": [10.0, 20.0, 30.0]})
    stats = compute_amount_stats(df)
    out = add_amount_zscore(df, stats=stats)

    assert np.isclose(out["amount_zscore"].mean(), 0.0, atol=1e-7)


def test_hour_features():
    df = pd.DataFrame({"Time": [0, 3600, 86399]})
    out = add_hour_features(df)

    assert list(out["hour_of_day"]) == [0, 1, 23]
    assert np.isclose(out["hour_sin"].iloc[0], 0.0)
    assert np.isclose(out["hour_cos"].iloc[0], 1.0)


def test_time_since_last_tx_first_row_is_zero():
    df = sample_df()
    out = add_time_since_last_tx(df)

    assert out["time_since_last_tx"].iloc[0] == 0
    assert out["time_since_last_tx"].iloc[1] == 1800
    assert out["time_since_last_tx"].iloc[2] == 1800


def test_rolling_tx_counts():
    df = sample_df()
    out = add_rolling_features(df)

    assert list(out["tx_frequency_1h"]) == [1, 2, 3, 2, 1]
    assert list(out["tx_frequency_6h"]) == [1, 2, 3, 4, 1]


def test_rolling_amount_means():
    df = sample_df()
    out = add_rolling_features(df)

    assert np.isclose(out["amount_mean_1h"].iloc[0], 0.0)
    assert np.isclose(out["amount_mean_1h"].iloc[1], 5.0)
    assert np.isclose(out["amount_mean_1h"].iloc[2], 10.0)


def test_build_features_produces_expected_columns():
    df = sample_df()
    out = build_features(df)

    expected = {
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
    }
    assert expected.issubset(set(out.columns))