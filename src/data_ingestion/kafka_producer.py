from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
from confluent_kafka import Producer

DEFAULT_TOPIC = "transactions"
DEFAULT_BROKERS = "localhost:9092"
DEFAULT_RATE = 10.0


def delivery_report(err, msg) -> None:
    if err is not None:
        print(f"Delivery failed for key={msg.key()}: {err}")
    else:
        print(
            f"Delivered to topic={msg.topic()} partition={msg.partition()} offset={msg.offset()}"
        )


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "transaction_id" not in df.columns:
        df = df.copy()
        df["transaction_id"] = range(len(df))
    return df


def build_message(row: pd.Series, max_jitter_seconds: int) -> dict:
    payload = row.to_dict()

    original_time = float(payload["Time"])
    jitter = random.uniform(-max_jitter_seconds, max_jitter_seconds)
    payload["Time"] = max(0.0, original_time + jitter)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="data/raw/creditcard.csv")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--brokers", default=DEFAULT_BROKERS)
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE)
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument("--jitter-seconds", type=int, default=5)
    args = parser.parse_args()

    if args.rate <= 0:
        raise ValueError("--rate must be > 0")

    df = load_data(Path(args.csv_path))
    producer = Producer({"bootstrap.servers": args.brokers})

    sleep_seconds = 1.0 / args.rate
    sent = 0

    try:
        for _, row in df.iterrows():
            if args.max_messages is not None and sent >= args.max_messages:
                break

            payload = build_message(row, args.jitter_seconds)
            key = str(payload["transaction_id"])

            producer.produce(
                topic=args.topic,
                key=key.encode("utf-8"),
                value=json.dumps(payload).encode("utf-8"),
                callback=delivery_report,
            )
            producer.poll(0)

            sent += 1
            time.sleep(sleep_seconds)

    finally:
        producer.flush()
        print(f"Finished sending {sent} messages to topic '{args.topic}'")


if __name__ == "__main__":
    main()
