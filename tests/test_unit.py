'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unit tests for core utilities: postprocess, metrics and edge cases."
'''

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from src.nlp.preprocess import preprocess_text
from src.nlp.postprocess import select_top_k
from src.nlp.vectorizers import build_tfidf_vectorizer, fit_transform_tfidf
from src.model.train import TrainingConfig, train_model
from src.model.predict import predict_with_probabilities
from src.model.evaluate import compute_basic_metrics

## ============================================================
## TEST: TOP-K SELECTION
## ============================================================
def test_select_top_k_basic() -> None:
    """
        Ensure top-k returns correct ordered labels
    """

    probabilities = np.array([
        [0.1, 0.7, 0.2],
    ])

    labels = ["A", "B", "C"]

    results = select_top_k(probabilities, labels, top_k=2)

    assert len(results) == 1
    assert results[0][0][0] == "B"
    assert results[0][1][0] == "C"
    assert results[0][0][1] >= results[0][1][1]

def test_select_top_k_full_length() -> None:
    """
        If top_k equals number of classes, all labels must be returned
    """

    probabilities = np.array([
        [0.3, 0.2, 0.5],
    ])

    labels = ["A", "B", "C"]

    results = select_top_k(probabilities, labels, top_k=3)

    returned_labels = {item[0] for item in results[0]}

    assert returned_labels == set(labels)

## ============================================================
## TEST: BASIC METRICS
## ============================================================
def test_compute_basic_metrics_perfect() -> None:
    """
        Perfect prediction should give accuracy=1 and f1=1
    """

    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])

    metrics = compute_basic_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1_micro"] == 1.0
    assert metrics["f1_macro"] == 1.0

def test_compute_basic_metrics_partial() -> None:
    """
        Partial prediction should give accuracy < 1
    """

    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 2, 2])

    metrics = compute_basic_metrics(y_true, y_pred)

    assert metrics["accuracy"] < 1.0
    assert 0.0 <= metrics["f1_micro"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0

## ============================================================
## TEST: EDGE CASES
## ============================================================
def test_select_top_k_empty_probabilities() -> None:
    """
        Empty probability matrix should raise error
    """

    probabilities = np.empty((0, 3))
    labels = ["A", "B", "C"]

    with pytest.raises(Exception):
        select_top_k(probabilities, labels, top_k=2)

## ============================================================
## E2E SMOKE TEST
## ============================================================
def test_e2e_tiny_train_predict_metrics() -> None:
    """
        Tiny end-to-end workflow test

        Steps:
            - Create small synthetic dataset
            - Preprocess texts
            - Vectorize with TF-IDF
            - Train simple model
            - Predict on same data
            - Compute metrics
    """

    ## Synthetic dataset (3 classes)
    texts = [
        "Patient has fever and cough, influenza suspected",
        "Fracture of the tibia after fall, orthopedic evaluation",
        "Diabetes follow-up with elevated glucose and HbA1c",
        "Severe cough and viral infection symptoms",
        "Bone fracture pain and swelling in leg",
        "Hyperglycemia and insulin therapy adjustment",
    ]

    labels = np.array([0, 1, 2, 0, 1, 2])

    ## Preprocess texts
    processed = [preprocess_text(t) for t in texts]

    ## Vectorize
    vectorizer = build_tfidf_vectorizer(max_features=5000)
    X = fit_transform_tfidf(vectorizer, processed)

    ## Train baseline model
    config = TrainingConfig(model_type="logreg", random_state=42, n_jobs=1)
    model = train_model(X, labels, config)

    ## Predict and compute metrics
    y_pred, probs = predict_with_probabilities(model, X)
    metrics = compute_basic_metrics(labels, y_pred)

    ## Assertions (smoke-level)
    assert probs.shape[0] == len(texts)
    assert probs.shape[1] == len(set(labels))

    assert "accuracy" in metrics
    assert "f1_micro" in metrics
    assert "f1_macro" in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_micro"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0