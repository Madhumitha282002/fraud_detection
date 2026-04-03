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
    Create or open a persistent File Data Context in the project root.
    """
    project_root = get_project_root()
    context = gx.get_context(mode="file", project_root_dir=str(project_root))
    return context


def get_or_create_datasource_and_asset(context):
    """
    Reuse existing datasource/asset if present, otherwise create them.
    """
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


def get_or_create_suite(context):
    """
    Load the suite if it exists; otherwise create it through the active context.
    """
    try:
        suite = context.suites.get(SUITE_NAME)

        # remove old expectations so reruns don't duplicate or leave stale rules
        for exp in list(suite.expectations):
            suite.delete_expectation(exp)

        suite.save()
    except Exception:
        suite = gx.ExpectationSuite(name=SUITE_NAME)
        suite = context.suites.add(suite)

    return suite


def add_expectations_to_suite(suite):
    expected_columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]

    # 1) Column existence
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(column_set=expected_columns)
    )

    # 2) Column types
    for col in [f"V{i}" for i in range(1, 29)] + ["Amount"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeOfType(
                column=col,
                type_="float64",
            )
        )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInTypeList(
            column="Time",
            type_list=["int64", "int32", "float64", "float32"],
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInTypeList(
            column="Class",
            type_list=["int64", "int32"],
        )
    )

    # 3) Null checks
    for col in expected_columns:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    # 4) Value ranges
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Amount",
            min_value=0,
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Class",
            value_set=[0, 1],
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Time",
            min_value=0,
        )
    )

    # 5) Row count
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=1000,
            max_value=500000,
        )
    )

    suite.save()


def main() -> None:
    project_root = get_project_root()
    csv_path = project_root / "data" / "raw" / "creditcard.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset at: {csv_path}")

    df = pd.read_csv(csv_path)
    context = get_context()

    _, _, batch_definition = get_or_create_datasource_and_asset(context)
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = get_or_create_suite(context)
    add_expectations_to_suite(suite)

    saved_suite = context.suites.get(SUITE_NAME)
    results = batch.validate(saved_suite)

    fraud_rate = float(df["Class"].mean()) if "Class" in df.columns else float("nan")
    fraud_rate_ok = "Class" in df.columns and 0.001 <= fraud_rate <= 0.05

    print(f"Project root: {project_root}")
    print(f"Suite saved: {saved_suite.name}")
    print(f"Initial GX validation success: {results.success}")
    print(f"Fraud rate: {fraud_rate:.6f}")
    print(f"Fraud-rate check passed: {fraud_rate_ok}")


if __name__ == "__main__":
    main()
