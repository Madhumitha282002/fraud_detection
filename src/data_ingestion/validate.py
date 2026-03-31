from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.data_ingestion.gx_setup import validate_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_csv")
    parser.add_argument("--ci", action="store_true", help="Use CI-friendly validation thresholds")
    args = parser.parse_args()

    csv_path = Path(args.path_to_csv)

    try:
        success, results, fraud_rate = validate_csv(csv_path, ci_mode=args.ci)
    except Exception as exc:
        msg = str(exc)
        if "ExpectationSuite with name transaction_data_quality was not found" in msg:
            print("Validation execution error: suite 'transaction_data_quality' does not exist yet.")
            print("Run: python3 -m src.data_ingestion.build_ge_suite")
            return 1

        print(f"Validation execution error: {exc}")
        return 1

    print(f"File: {csv_path}")
    if not math.isnan(fraud_rate):
        print(f"Fraud rate: {fraud_rate:.6f}")

    try:
        print(json.dumps(results.to_json_dict(), indent=2))
    except Exception:
        print(results)

    if success:
        print("VALIDATION PASSED")
        return 0

    print("VALIDATION FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())