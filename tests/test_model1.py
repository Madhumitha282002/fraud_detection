from __future__ import annotations

import mlflow
import numpy as np

from src.training.utils import compute_metrics, get_X_y, load_training_dataframe, stratified_split


TRACKING_URI = "http://localhost:5001"
MODEL_NAME = "fraud-detection-model"
MODEL_STAGE = "Production"
MIN_AUC_PR = 0.07


def load_registry_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")


def test_model_loads_successfully_from_registry():
    model = load_registry_model()
    assert model is not None


def test_prediction_output_shape_and_type_from_registry():
    model = load_registry_model()

    df = load_training_dataframe()
    splits = stratified_split(df, random_state=42)
    X_test, _ = get_X_y(splits["test"])

    preds = np.asarray(model.predict(X_test.head(32)))
    assert preds.shape == (32,)
    assert np.issubdtype(preds.dtype, np.floating)


def test_probabilities_in_range_from_registry():
    model = load_registry_model()

    df = load_training_dataframe()
    splits = stratified_split(df, random_state=42)
    X_test, _ = get_X_y(splits["test"])

    preds = np.asarray(model.predict(X_test.head(128)))
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)


def test_auc_pr_threshold_from_registry():
    model = load_registry_model()

    df = load_training_dataframe()
    splits = stratified_split(df, random_state=42)
    X_test, y_test = get_X_y(splits["test"])

    preds = np.asarray(model.predict(X_test))
    metrics = compute_metrics(y_test, preds)

    assert metrics["auc_pr"] >= MIN_AUC_PR