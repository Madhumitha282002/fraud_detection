from __future__ import annotations

from typing import Any


REQUIRED_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class", "transaction_id"]


def validate_transaction_message(payload: dict[str, Any]) -> tuple[bool, str | None]:
    for col in REQUIRED_COLUMNS:
        if col not in payload:
            return False, f"Missing required field: {col}"

    try:
        for i in range(1, 29):
            float(payload[f"V{i}"])
        float(payload["Time"])
        float(payload["Amount"])
        int(payload["Class"])
        int(payload["transaction_id"])
    except (TypeError, ValueError) as e:
        return False, f"Type conversion failed: {e}"

    if float(payload["Time"]) < 0:
        return False, "Time must be >= 0"

    if float(payload["Amount"]) < 0:
        return False, "Amount must be >= 0"

    if int(payload["Class"]) not in (0, 1):
        return False, "Class must be 0 or 1"

    return True, None