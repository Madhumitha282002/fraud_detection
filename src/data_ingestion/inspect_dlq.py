from __future__ import annotations

import json
from typing import Any

import structlog
from confluent_kafka import Consumer

from src.configs import settings
from src.core.logging_config import configure_logging

configure_logging()
logger = structlog.get_logger()

DLQ_TOPIC = settings.kafka_dlq_topic
BROKERS = settings.kafka_brokers


def make_consumer() -> Consumer:
    return Consumer(
        {
            "bootstrap.servers": BROKERS,
            "group.id": "fraud-dlq-inspector",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )


def decode_message(raw_value: bytes | None) -> dict[str, Any]:
    if raw_value is None:
        return {"raw_value": None}

    try:
        decoded = raw_value.decode("utf-8")
    except Exception:
        return {"raw_value": "<failed to decode bytes as utf-8>"}

    try:
        return json.loads(decoded)
    except json.JSONDecodeError:
        return {"raw_value": decoded}


def print_dlq_record(payload: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("DLQ MESSAGE")
    print("=" * 80)
    print(f"transaction_id   : {payload.get('transaction_id')}")
    print(f"error            : {payload.get('error')}")
    print(f"timestamp        : {payload.get('timestamp')}")
    print(f"source_topic     : {payload.get('source_topic')}")
    print(f"source_partition : {payload.get('source_partition')}")
    print(f"source_offset    : {payload.get('source_offset')}")
    print(f"raw_message      : {payload.get('raw_message')}")
    print("=" * 80)


def main(max_messages: int = 10) -> None:
    consumer = make_consumer()
    consumer.subscribe([DLQ_TOPIC])

    logger.info("dlq_inspector_started", topic=DLQ_TOPIC, max_messages=max_messages)

    count = 0

    try:
        while count < max_messages:
            msg = consumer.poll(1.0)

            if msg is None:
                continue

            if msg.error():
                logger.error("dlq_consumer_error", error=str(msg.error()))
                continue

            payload = decode_message(msg.value())

            logger.info(
                "dlq_message_received",
                topic=msg.topic(),
                partition=msg.partition(),
                offset=msg.offset(),
            )

            if isinstance(payload, dict):
                print_dlq_record(payload)
            else:
                print(payload)

            count += 1
            consumer.commit(message=msg)

    except KeyboardInterrupt:
        logger.info("dlq_inspector_interrupted")
    finally:
        consumer.close()
        logger.info("dlq_inspector_stopped", messages_read=count)


if __name__ == "__main__":
    main()