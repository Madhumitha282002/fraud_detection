from __future__ import annotations

import os

import numpy as np
import pytest

import mlflow
from src.training.utils import (
    compute_metrics,
    get_X_y,
    load_training_dataframe,
    stratified_split,
)

MODEL_URI_ENV = "MLFLOW_MODEL_URI"


@pytest.fixture(scope="session")
def test_data():
    df = load_training_dataframe()
    splits = stratified_split(df, random_state=42)
    X_test, y_test = get_X_y(splits["test"])
    return X_test, y_test


@pytest.fixture(scope="session")
def loaded_model():
    model_uri = os.getenv(MODEL_URI_ENV)
    if not model_uri:
        pytest.skip(f"Set {MODEL_URI_ENV} to run MLflow model tests.")
    mlflow.set_tracking_uri("http://localhost:5001")
    return mlflow.pyfunc.load_model(model_uri)


def test_model_loads_successfully(loaded_model):
    assert loaded_model is not None


def test_prediction_output_shape_and_type(loaded_model, test_data):
    X_test, _ = test_data
    preds = loaded_model.predict(X_test.head(32))
    preds = np.asarray(preds)

    assert preds.shape == (32,)
    assert np.issubdtype(preds.dtype, np.floating)


def test_probabilities_in_range(loaded_model, test_data):
    X_test, _ = test_data
    preds = np.asarray(loaded_model.predict(X_test.head(128)))

    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)


def test_auc_pr_exceeds_baseline_threshold(loaded_model, test_data):
    X_test, y_test = test_data
    preds = np.asarray(loaded_model.predict(X_test))
    metrics = compute_metrics(y_test, preds)

    assert metrics["auc_pr"] >= 0.50
