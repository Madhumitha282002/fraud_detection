# Real-Time Fraud Detection — End-to-End MLOps System

An end-to-end MLOps project for real-time fraud detection using streaming ingestion, feature engineering, experiment tracking, model serving, monitoring, drift detection, retraining orchestration, and a user-facing demo dashboard. The project demonstrates the complete ML lifecycle from data validation and training to deployment, observability, and continuous improvement. :contentReference[oaicite:1]{index=1}

## Overview

This project implements a production-style machine learning system for fraud detection on credit card transactions. It combines model development and operational tooling into a unified workflow: incoming transactions are validated and processed, predictions are served through an API, system and model metrics are monitored, drift can be detected automatically, and retraining workflows can be triggered through orchestration. :contentReference[oaicite:2]{index=2}

The project was built to demonstrate practical MLOps capabilities, including MLflow experiment tracking and registry, FastAPI model serving, Kafka-based streaming, Airflow orchestration, Prometheus and Grafana monitoring, Evidently drift detection, and a Streamlit dashboard for interactive system visibility. :contentReference[oaicite:3]{index=3}

## Key Highlights

- Real-time fraud prediction service built with FastAPI
- MLflow experiment tracking and model metadata management
- Kafka-based streaming pipeline for transaction ingestion
- Great Expectations data validation and quality checks
- Feature engineering and feature-serving workflow
- Evidently drift detection with HTML drift reports
- Prometheus metrics and Grafana observability dashboard
- Airflow retraining DAG and alert-triggered retraining flow
- Streamlit demo dashboard for prediction and system visibility
- Load testing and lifecycle validation documentation :contentReference[oaicite:4]{index=4}

## Architecture

flowchart LR
    A[Raw Transactions / Dataset] --> B[Kafka Producer]
    B --> C[Kafka Topic]
    C --> D[Kafka Consumer]
    D --> E[Validation + Feature Engineering]
    E --> F[FastAPI Inference API]
    F --> G[Prediction Log SQLite]
    F --> H[Prometheus Metrics]
    H --> I[Grafana Dashboard]
    G --> J[Evidently Drift Detection]
    J --> K[Alerting]
    K --> L[Airflow Retraining DAG]
    L --> M[MLflow Tracking / Registry]
    M --> F
    N[Streamlit Dashboard] --> F
    N --> G
    N --> L

A more detailed explanation of the system design is available in docs/architecture.md

## Tech Stack
| Tool | Role |
|---|---|
| FastAPI | Real-time inference API |
| MLflow | Experiment tracking and model registry |
| Feast | Feature store |
| Kafka | Streaming transactions |
| Great Expectations | Data validation |
| Airflow | Retraining orchestration |
| Prometheus | Metrics collection |
| Grafana | Observability dashboards |
| Streamlit | Demo dashboard |    


## Quick Start
```bash
docker compose up -d
python -m uvicorn src.serving.app:app --reload
python -m streamlit run dashboards/streamlit_app.py