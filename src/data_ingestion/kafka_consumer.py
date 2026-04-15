from __future__ import annotations

import json
import signal
import time

import pandas as pd
import structlog
from confluent_kafka import Consumer, Producer

from src.configs import settings
from src.core.logging_config import configure_logging
from src.data_ingestion.message_validation import validate_transaction_message
from src.feature_engineering.features import build_features

configure_logging()
logger = structlog.get_logger()

RAW_TOPIC = settings.kafka_raw_topic
PROCESSED_TOPIC = settings.kafka_processed_topic
DLQ_TOPIC = settings.kafka_dlq_topic
BROKERS = settings.kafka_brokers
GROUP_ID = settings.kafka_group_id

# This flag controls whether the consumer loop keeps running.
RUNNING = True


def handle_shutdown(signum, frame) -> None:
    # When SIGINT or SIGTERM arrives, stop the loop gracefully.
    global RUNNING
    RUNNING = False
    logger.info("shutdown_signal_received", signal=signum)


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
    producer: Producer,
    original_value: bytes | None,
    error_message: str,
    *,
    topic: str | None = None,
    partition: int | None = None,
    offset: int | None = None,
) -> None:
    raw_message = original_value.decode("utf-8") if original_value else None

    payload = {
        "error": error_message,
        "raw_message": raw_message,
        "timestamp": time.time(),
        "source_topic": topic,
        "source_partition": partition,
        "source_offset": offset,
    }

    try:
        decoded = json.loads(raw_message) if raw_message else {}
        payload["transaction_id"] = decoded.get("transaction_id")
    except Exception:
        payload["transaction_id"] = None

    producer.produce(DLQ_TOPIC, value=json.dumps(payload).encode("utf-8"))
    producer.flush()


def process_payload(payload: dict) -> dict:
    df = pd.DataFrame([payload])
    featured = build_features(df)
    return featured.iloc[0].to_dict()


def main() -> None:
    global RUNNING

    # Register graceful shutdown handlers.
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    consumer = make_consumer()
    producer = make_producer()

    consumer.subscribe([RAW_TOPIC])
    logger.info("consumer_started", topic=RAW_TOPIC, group_id=GROUP_ID)

    try:
        while RUNNING:
            producer.poll(0)

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
                    send_to_dlq(
                        producer,
                        msg.value(),
                        error or "validation_failed",
                        topic=msg.topic(),
                        partition=msg.partition(),
                        offset=msg.offset(),
                    )
                    logger.warning(
                        "message_sent_to_dlq",
                        topic=RAW_TOPIC,
                        partition=msg.partition(),
                        offset=msg.offset(),
                        reason=error,
                        transaction_id=payload.get("transaction_id"),
                    )
                    consumer.commit(message=msg)
                    continue

                enriched = process_payload(payload)

                logger.info(
                    "features_computed",
                    transaction_id=payload.get("transaction_id"),
                    enriched=enriched,
                )

                producer.produce(
                    PROCESSED_TOPIC,
                    key=str(payload.get("transaction_id", "unknown")).encode("utf-8"),
                    value=json.dumps(enriched, default=str).encode("utf-8"),
                )
                producer.flush()

                logger.info(
                    "processed_message_published",
                    output_topic=PROCESSED_TOPIC,
                    transaction_id=payload.get("transaction_id"),
                )

                latency_ms = round((time.time() - start) * 1000, 2)
                logger.info(
                    "message_processed",
                    topic=msg.topic(),
                    partition=msg.partition(),
                    offset=msg.offset(),
                    processing_time_ms=latency_ms,
                    transaction_id=payload.get("transaction_id"),
                )

                consumer.commit(message=msg)

            except Exception as e:
                send_to_dlq(
                    producer,
                    msg.value(),
                    str(e),
                    topic=msg.topic(),
                    partition=msg.partition(),
                    offset=msg.offset(),
                )
                logger.exception(
                    "message_processing_failed",
                    topic=msg.topic(),
                    partition=msg.partition(),
                    offset=msg.offset(),
                    error=str(e),
                )
                consumer.commit(message=msg)

    finally:
        try:
            logger.info("flushing_producer")
            producer.flush(timeout=10)
        except Exception as e:
            logger.warning("producer_flush_failed", error=str(e))

        consumer.close()
        logger.info("consumer_closed")
