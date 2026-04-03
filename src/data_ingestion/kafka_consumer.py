from __future__ import annotations

import json
import time

import pandas as pd
import structlog
from confluent_kafka import Consumer, Producer
import logging
from src.data_ingestion.message_validation import validate_transaction_message
from src.feature_engineering.features import build_features

RAW_TOPIC = "transactions"
PROCESSED_TOPIC = "processed-transactions"
DLQ_TOPIC = "transactions-dlq"
BROKERS = "localhost:9092"
GROUP_ID = "fraud-feature-consumer"


logging.basicConfig(format="%(message)s", level=logging.INFO)
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()


def make_consumer() -> Consumer:
    return Consumer(
        {
            "bootstrap.servers": BROKERS,
            "group.id": GROUP_ID,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )


def make_producer() -> Producer:
    return Producer({"bootstrap.servers": BROKERS})


def send_to_dlq(
    producer: Producer, original_value: bytes | None, error_message: str
) -> None:
    payload = {
        "error": error_message,
        "raw_message": original_value.decode("utf-8") if original_value else None,
        "timestamp": time.time(),
    }
    producer.produce(DLQ_TOPIC, value=json.dumps(payload).encode("utf-8"))
    producer.flush()


def process_payload(payload: dict) -> dict:
    df = pd.DataFrame([payload])
    featured = build_features(df)
    return featured.iloc[0].to_dict()


def main() -> None:
    consumer = make_consumer()
    producer = make_producer()

    consumer.subscribe([RAW_TOPIC])
    logger.info("consumer_started", topic=RAW_TOPIC, group_id=GROUP_ID)

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue

            if msg.error():
                logger.error("consumer_error", error=str(msg.error()))
                continue

            start = time.time()
            try:
                payload = json.loads(msg.value().decode("utf-8"))

                is_valid, error = validate_transaction_message(payload)
                if not is_valid:
                    send_to_dlq(producer, msg.value(), error or "validation_failed")
                    logger.warning(
                        "message_sent_to_dlq",
                        topic=RAW_TOPIC,
                        offset=msg.offset(),
                        reason=error,
                    )
                    consumer.commit(message=msg)
                    continue

                enriched = process_payload(payload)

                print(json.dumps(enriched, default=str))

                producer.produce(
                    PROCESSED_TOPIC,
                    key=str(payload["transaction_id"]).encode("utf-8"),
                    value=json.dumps(enriched, default=str).encode("utf-8"),
                )
                producer.flush()

                latency_ms = round((time.time() - start) * 1000, 2)
                logger.info(
                    "message_processed",
                    topic=msg.topic(),
                    partition=msg.partition(),
                    offset=msg.offset(),
                    processing_time_ms=latency_ms,
                    transaction_id=payload["transaction_id"],
                )

                consumer.commit(message=msg)

            except Exception as e:
                send_to_dlq(producer, msg.value(), str(e))
                logger.exception(
                    "message_processing_failed",
                    topic=msg.topic(),
                    offset=msg.offset(),
                    error=str(e),
                )
                consumer.commit(message=msg)

    except KeyboardInterrupt:
        logger.info("consumer_shutdown")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
