from __future__ import annotations

from pathlib import Path

import great_expectations as gx
import pandas as pd

SUITE_NAME = "transaction_data_quality"
DATASOURCE_NAME = "fraud_datasource"
ASSET_NAME = "creditcard_asset"
BATCH_DEF_NAME = "whole_batch"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_context():
    """
    Create or open the same persistent File Data Context used by build_ge_suite.py
    """
    project_root = get_project_root()
    return gx.get_context(mode="file", project_root_dir=str(project_root))


def load_dataframe(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.is_absolute():
        csv_path = get_project_root() / csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    return pd.read_csv(csv_path)


def get_or_create_datasource_and_asset(context):
    try:
        datasource = context.data_sources.get(DATASOURCE_NAME)
    except Exception:
        datasource = context.data_sources.add_pandas(name=DATASOURCE_NAME)

    try:
        asset = datasource.get_asset(ASSET_NAME)
    except Exception:
        asset = datasource.add_dataframe_asset(name=ASSET_NAME)

    try:
        batch_definition = asset.get_batch_definition(BATCH_DEF_NAME)
    except Exception:
        batch_definition = asset.add_batch_definition_whole_dataframe(BATCH_DEF_NAME)

    return datasource, asset, batch_definition


def get_suite(context):
    return context.suites.get(SUITE_NAME)


def validate_csv(csv_path: str | Path):
    context = get_context()
    df = load_dataframe(csv_path)

    _, _, batch_definition = get_or_create_datasource_and_asset(context)
    suite = get_suite(context)

    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    results = batch.validate(suite)

    fraud_rate = float(df["Class"].mean()) if "Class" in df.columns else float("nan")
    fraud_rate_ok = "Class" in df.columns and 0.001 <= fraud_rate <= 0.05

    success = bool(results.success) and fraud_rate_ok
    return success, results, fraud_rate