from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"
AIRFLOW_BASE_URL = "http://127.0.0.1:8080/api/v1"
AIRFLOW_USERNAME = "admin"
AIRFLOW_PASSWORD = "admin"

PREDICTIONS_DB_PATH = Path("data/predictions/predictions.db")
DRIFT_REPORTS_DIR = Path("dashboards/reports")
MODEL_CONFIG_PATH = Path("configs/model_config.yaml")

st.set_page_config(
    page_title="Fraud Detection MLOps Dashboard",
    layout="wide",
)


def load_recent_predictions(limit: int = 50) -> pd.DataFrame:
    if not PREDICTIONS_DB_PATH.exists():
        return pd.DataFrame()

    with sqlite3.connect(PREDICTIONS_DB_PATH) as conn:
        query = f"""
            SELECT timestamp, transaction_id, fraud_probability, is_fraud, model_version, latency_ms
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT {limit}
        """
        return pd.read_sql_query(query, conn)


def get_prediction_summary() -> dict:
    if not PREDICTIONS_DB_PATH.exists():
        return {
            "total_predictions": 0,
            "fraud_count": 0,
            "avg_latency_ms": 0.0,
            "current_model_version": "N/A",
        }

    with sqlite3.connect(PREDICTIONS_DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_predictions,
                COALESCE(SUM(is_fraud), 0) AS fraud_count,
                COALESCE(AVG(latency_ms), 0.0) AS avg_latency_ms,
                COALESCE(MAX(model_version), 'N/A') AS current_model_version
            FROM predictions
            """
        ).fetchone()

    return {
        "total_predictions": int(row[0] or 0),
        "fraud_count": int(row[1] or 0),
        "avg_latency_ms": round(float(row[2] or 0.0), 2),
        "current_model_version": row[3] or "N/A",
    }


def get_health() -> dict:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def get_model_info() -> dict:
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def send_prediction(payload: dict) -> dict:
    response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


def latest_drift_report_path() -> str | None:
    if not DRIFT_REPORTS_DIR.exists():
        return None

    reports = sorted(DRIFT_REPORTS_DIR.glob("data_drift_report_*.html"))
    if not reports:
        return None

    return str(reports[-1])


def build_default_payload() -> dict:
    payload = {
        "transaction_id": "streamlit_tx_001",
        "Time": 1000.0,
        "Amount": 125.5,
    }
    for i in range(1, 29):
        payload[f"V{i}"] = round(0.1 * i, 4)
    return payload


def get_model_metrics() -> dict:
    """
    Placeholder metrics panel.
    Replace with MLflow lookup later if you want live registry metrics.
    """
    return {
        "AUC-PR": 0.50,
        "Precision": 0.50,
        "Recall": 0.50,
        "F1": 0.50,
    }


def get_pipeline_status() -> dict:
    dag_id = "fraud_detection_retraining"
    try:
        response = requests.get(
            f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns?limit=1&order_by=-start_date",
            auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        dag_runs = data.get("dag_runs", [])

        if not dag_runs:
            return {
                "last_status": "No runs found",
                "last_duration": "N/A",
                "next_scheduled_run": "N/A",
            }

        latest = dag_runs[0]
        start_date = latest.get("start_date")
        end_date = latest.get("end_date")
        state = latest.get("state", "unknown")

        duration = "N/A"
        if start_date and end_date:
            duration = f"{start_date} -> {end_date}"

        return {
            "last_status": state,
            "last_duration": duration,
            "next_scheduled_run": "Check Airflow UI / scheduler",
        }
    except Exception as e:
        return {
            "last_status": "Unavailable",
            "last_duration": "Unavailable",
            "next_scheduled_run": str(e),
        }


st.title("Real-Time Fraud Detection — MLOps Dashboard")
st.caption("Streamlit demo for serving, monitoring, and pipeline visibility")

with st.sidebar:
    st.header("Navigation")
    section = st.radio(
        "Go to",
        [
            "Overview",
            "Submit Transaction",
            "Live Predictions",
            "Monitoring",
            "Pipeline & Model",
        ],
    )

health = get_health()
model_info = get_model_info()
recent_predictions = load_recent_predictions()
summary = get_prediction_summary()
model_metrics = get_model_metrics()
pipeline_status = get_pipeline_status()

top1, top2, top3, top4 = st.columns(4)
top1.metric("Total Predictions", summary["total_predictions"])
top2.metric("Fraud Count", summary["fraud_count"])
top3.metric("Avg Latency (ms)", summary["avg_latency_ms"])
top4.metric("Current Model", summary["current_model_version"])

if section == "Overview":
    st.subheader("System Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("API Status", health.get("status", "unknown"))
    col2.metric("Model Version", health.get("model_version", "N/A"))
    col3.metric("Model Stage", health.get("model_stage", "N/A"))

    st.write("### Recent Predictions")
    if recent_predictions.empty:
        st.info("No predictions logged yet.")
    else:
        st.dataframe(recent_predictions, use_container_width=True)

    st.write("### Model Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC-PR", model_metrics["AUC-PR"])
    m2.metric("Precision", model_metrics["Precision"])
    m3.metric("Recall", model_metrics["Recall"])
    m4.metric("F1", model_metrics["F1"])

elif section == "Submit Transaction":
    st.subheader("Submit a Transaction")

    default_payload = build_default_payload()

    with st.form("predict_form"):
        transaction_id = st.text_input(
            "Transaction ID", value=default_payload["transaction_id"]
        )
        time_value = st.number_input(
            "Time", min_value=0.0, value=float(default_payload["Time"])
        )
        amount_value = st.number_input(
            "Amount", min_value=0.0, value=float(default_payload["Amount"])
        )

        feature_values = {}
        cols = st.columns(4)
        for idx, i in enumerate(range(1, 29)):
            with cols[idx % 4]:
                feature_values[f"V{i}"] = st.number_input(
                    f"V{i}",
                    value=float(default_payload[f"V{i}"]),
                    format="%.4f",
                )

        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "transaction_id": transaction_id,
            "Time": time_value,
            "Amount": amount_value,
            **feature_values,
        }

        try:
            result = send_prediction(payload)
            st.success("Prediction completed")
            st.json(result)

            if result["is_fraud"]:
                st.error(
                    f"Fraud detected with probability {result['fraud_probability']:.4f}"
                )
            else:
                st.success(
                    f"Legit transaction with probability {result['fraud_probability']:.4f}"
                )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif section == "Live Predictions":
    st.subheader("Live Prediction Feed")

    if st.button("Refresh"):
        recent_predictions = load_recent_predictions()

    if recent_predictions.empty:
        st.info("No recent predictions available.")
    else:
        display_df = recent_predictions.copy()
        display_df["is_fraud"] = display_df["is_fraud"].astype(bool)
        st.dataframe(display_df, use_container_width=True)

elif section == "Monitoring":
    st.subheader("Monitoring")

    report_path = latest_drift_report_path()
    if report_path:
        st.success(f"Latest drift report found: {report_path}")
    else:
        st.warning("No drift report found yet.")

    st.write("### API Health")
    st.json(health)

elif section == "Pipeline & Model":
    st.subheader("Pipeline & Model Summary")

    st.write("### Model Info")
    st.json(model_info)

    st.write("### Pipeline Status")
    p1, p2, p3 = st.columns(3)
    p1.metric("Last DAG Status", pipeline_status["last_status"])
    p2.metric("Last Run Duration", pipeline_status["last_duration"])
    p3.metric("Next Scheduled Run", pipeline_status["next_scheduled_run"])

    st.write("### Model Metrics")
    st.json(model_metrics)
