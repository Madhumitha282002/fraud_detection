from __future__ import annotations

import random

from locust import HttpUser, between, task


def build_payload(tx_id: int) -> dict:
    payload = {
        "transaction_id": f"load_test_{tx_id}",
        "Time": float(random.randint(0, 86400)),
        "Amount": round(random.uniform(1.0, 1000.0), 2),
    }
    for i in range(1, 29):
        payload[f"V{i}"] = round(random.uniform(-3.0, 3.0), 4)
    return payload


class FraudApiUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def predict(self) -> None:
        tx_id = random.randint(1, 1_000_000)
        payload = build_payload(tx_id)
        self.client.post("/predict", json=payload)

    @task(1)
    def health(self) -> None:
        self.client.get("/health")
