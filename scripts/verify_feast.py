from __future__ import annotations

import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path="src/feature_engineering/feature_store")

entity_df = pd.DataFrame(
    {
        "transaction_id": [0, 1, 2],
        "event_timestamp": pd.to_datetime(
            ["2020-01-01 00:00:00", "2020-01-01 00:30:00", "2020-01-01 01:00:00"]
        ),
    }
)

offline = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "transaction_features:amount_log",
        "transaction_features:amount_zscore",
        "transaction_features:hour_sin",
        "transaction_features:hour_cos",
        "transaction_features:tx_frequency_1h",
        "transaction_features:amount_mean_1h",
        "transaction_features:time_since_last_tx",
    ],
).to_df()

online = store.get_online_features(
    features=[
        "transaction_features:amount_log",
        "transaction_features:amount_zscore",
        "transaction_features:hour_sin",
        "transaction_features:hour_cos",
        "transaction_features:tx_frequency_1h",
        "transaction_features:amount_mean_1h",
        "transaction_features:time_since_last_tx",
    ],
    entity_rows=[
        {"transaction_id": 0},
        {"transaction_id": 1},
        {"transaction_id": 2},
    ],
).to_dict()

print("OFFLINE")
print(offline)
print("\nONLINE")
print(online)