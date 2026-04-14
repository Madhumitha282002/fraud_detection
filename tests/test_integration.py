from __future__ import annotations

import json
import time

import mlflow
import pandas as pd
import pytest
import requests
from confluent_kafka import Consumer, Producer

KAFKA_BROKERS = "localhost:9092"
RAW_TOPIC = "transactions"
PROCESSED_TOPIC = "processed-transactions"
AIRFLOW_BASE_URL = "http://localhost:8080/api/v1"
MLFLOW_TRACKING_URI = "http://localhost:5001"
MODEL_NAME = "fraud-detection-model"


def _load_sample_transaction() -> dict:
    df = pd.read_csv("tests/fixtures/sample_transactions.csv")
    row = df.iloc[0].to_dict()
    row["transaction_id"] = int(row.get("transaction_id", 999001))
    return row


def _make_consumer(topic: str, group_id: str) -> Consumer:
    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_BROKERS,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([topic])
    return consumer


@pytest.mark.integration
def test_kafka_round_trip():
    producer = Producer({"bootstrap.servers": KAFKA_BROKERS})
    consumer = _make_consumer(RAW_TOPIC, "integration-kafka-roundtrip")

    payload = _load_sample_transaction()
    payload["transaction_id"] = 991001
    value = json.dumps(payload).encode("utf-8")

    producer.produce(RAW_TOPIC, key=b"991001", value=value)
    producer.flush()

    deadline = time.time() + 10
    received = None
    while time.time() < deadline:
        msg = consumer.poll(1.0)
        if msg is None or msg.error():
            continue
        candidate = json.loads(msg.value().decode("utf-8"))
        if int(candidate["transaction_id"]) == 991001:
            received = candidate
            break

    consumer.close()

    assert received is not None
    assert int(received["transaction_id"]) == 991001


@pytest.mark.integration
def test_feature_pipeline_publishes_enriched_message():
    producer = Producer({"bootstrap.servers": KAFKA_BROKERS})
    consumer = _make_consumer(PROCESSED_TOPIC, "integration-processed-topic")

    payload = _load_sample_transaction()
    payload["transaction_id"] = 991002

    producer.produce(
        RAW_TOPIC,
        key=b"991002",
        value=json.dumps(payload).encode("utf-8"),
    )
    producer.flush()

    deadline = time.time() + 15
    received = None
    while time.time() < deadline:
        msg = consumer.poll(1.0)
        if msg is None or msg.error():
            continue

        candidate = json.loads(msg.value().decode("utf-8"))
        candidate_tx_id = candidate.get("transaction_id")
        if candidate_tx_id is not None and int(float(candidate_tx_id)) == 991002:
            received = candidate
            break

    consumer.close()

    assert received is not None
    assert "amount_log" in received
    assert "amount_zscore" in received
    assert "hour_sin" in received
    assert "hour_cos" in received
    assert "tx_frequency_1h" in received
    assert "amount_mean_1h" in received
    assert "time_since_last_tx" in received


@pytest.mark.integration
def test_inference_round_trip():
    payload = {
        "transaction_id": 123,
        "Amount": 100.0,
        "Time": 12345.0,
    }

    response = requests.post("http://localhost:8000/predict", json=payload, timeout=10)
    assert response.status_code == 200

    body = response.json()
    assert "fraud_probability" in body
    assert "is_fraud" in body
    assert 0.0 <= float(body["fraud_probability"]) <= 1.0


@pytest.mark.integration
def test_model_registry_production_model_loads():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")

    sample = pd.DataFrame(
        [
            {
                "amount_log": 1.0,
                "amount_zscore": 0.0,
                "hour_of_day": 12,
                "hour_sin": 0.0,
                "hour_cos": -1.0,
                "tx_frequency_1h": 2,
                "tx_frequency_6h": 3,
                "tx_frequency_24h": 5,
                "amount_mean_1h": 50.0,
                "amount_mean_24h": 60.0,
                "time_since_last_tx": 120.0,
            }
        ]
    )

    preds = model.predict(sample)
    assert len(preds) == 1
    assert 0.0 <= float(preds[0]) <= 1.0


@pytest.mark.integration
def test_retraining_pipeline_triggers_new_run():
    auth = ("admin", "admin")

    dag_run_id = f"integration-{int(time.time())}"
    trigger_resp = requests.post(
        f"{AIRFLOW_BASE_URL}/dags/fraud_detection_retraining/dagRuns",
        auth=auth,
        json={"dag_run_id": dag_run_id},
        timeout=15,
    )
    assert trigger_resp.status_code in (200, 201)

    deadline = time.time() + 900
    state = None
    while time.time() < deadline:
        resp = requests.get(
            f"{AIRFLOW_BASE_URL}/dags/fraud_detection_retraining/dagRuns/{dag_run_id}",
            auth=auth,
            timeout=15,
        )
        assert resp.status_code == 200
        state = resp.json()["state"]
        if state in ("success", "failed"):
            break
        time.sleep(10)

    assert state == "success", f"DAG final state was: {state}"
