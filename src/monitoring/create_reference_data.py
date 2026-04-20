from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    input_path = Path("data/raw/creditcard.csv")
    output_path = Path("data/processed/training_reference.parquet")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Create a transaction_id if one does not already exist
    if "transaction_id" not in df.columns:
        df.insert(0, "transaction_id", [f"tx_{i}" for i in range(len(df))])

    keep_columns = ["transaction_id", "Time", "Amount"] + [
        f"V{i}" for i in range(1, 29)
    ]
    reference_df = df[keep_columns].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_df.to_parquet(output_path, index=False)

    print(f"Saved reference dataset to: {output_path}")
    print(f"Shape: {reference_df.shape}")


if __name__ == "__main__":
    main()
