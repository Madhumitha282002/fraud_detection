from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.feature_engineering.features import build_features


RAW_PATH = Path("data/raw/creditcard.csv")
OUT_PATH = Path("data/processed/transactions_features.parquet")


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    featured = build_features(df)

    # Feast likes an event timestamp for historical retrieval/materialization
    featured["event_timestamp"] = pd.to_datetime(
        featured["Time"], unit="s", origin=pd.Timestamp("2020-01-01")
    )
    featured["created_timestamp"] = featured["event_timestamp"]

    featured.to_parquet(OUT_PATH, index=False)
    print(f"Saved features to {OUT_PATH} with shape={featured.shape}")


if __name__ == "__main__":
    main()