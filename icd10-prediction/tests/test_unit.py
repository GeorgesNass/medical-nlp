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
import pandas as pd
import pytest

from src.core.data_consistency import run_data_consistency
from src.core.data_quality import run_data_quality
from src.core.data_drift import run_data_drift
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

def test_select_top_k_greater_than_classes() -> None:
    """
        top_k greater than number of classes should not crash
    """

    probabilities = np.array([[0.2, 0.8]])
    labels = ["A", "B"]

    results = select_top_k(probabilities, labels, top_k=10)

    assert len(results[0]) == 2
  
def test_select_top_k_mismatch_labels() -> None:
    """
        Mismatch between labels and probabilities should fail
    """

    probabilities = np.array([[0.2, 0.8]])
    labels = ["A"]  # mismatch

    with pytest.raises(Exception):
        select_top_k(probabilities, labels, top_k=1)
        
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

def test_compute_basic_metrics_empty() -> None:
    """
        Empty inputs should raise error
    """

    y_true = np.array([])
    y_pred = np.array([])

    with pytest.raises(Exception):
        compute_basic_metrics(y_true, y_pred)
        
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
    assert model is not None 
    
    assert probs.shape[0] == len(texts)
    assert probs.shape[1] == len(set(labels))

    assert "accuracy" in metrics
    assert "f1_micro" in metrics
    assert "f1_macro" in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_micro"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    
## ============================================================
## DATA CONSISTENCY TESTS (ICD10)
## ============================================================
def test_data_consistency_valid_icd10() -> None:
    """
        Validate correct ICD10 payload

        Returns:
            None
    """

    data = {
        "text": "patient infection intestinale",
        "labels": ["A00"],
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is True

def test_data_consistency_invalid_label_format() -> None:
    """
        Detect invalid ICD10 label format

        Returns:
            None
    """

    data = {
        "text": "test",
        "labels": ["INVALID"],
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is False

def test_data_consistency_empty_text() -> None:
    """
        Detect empty text

        Returns:
            None
    """

    data = {
        "text": "",
        "labels": ["A00"],
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is False

def test_data_consistency_missing_labels() -> None:
    """
        Detect missing labels

        Returns:
            None
    """

    data = {
        "text": "test",
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is False
    
## ============================================================
## DATA QUALITY TESTS
## ============================================================
def test_data_quality_valid_dataset() -> None:
    """
        Validate correct dataset (no anomalies)

        Returns:
            None
    """

    texts = [
        "patient fever infection",
        "fracture tibia trauma",
        "diabetes glucose high",
    ]

    labels = ["A00", "S82", "E11"]

    result = run_data_quality(texts=texts, labels=labels)

    assert result["score"] > 0.8
    assert result["errors"] == 0

def test_data_quality_detect_empty_text() -> None:
    """
        Detect empty text anomaly

        Returns:
            None
    """

    texts = ["", "valid text"]
    labels = ["A00", "B00"]

    result = run_data_quality(texts=texts, labels=labels)

    assert result["errors"] > 0

def test_data_quality_detect_label_mismatch() -> None:
    """
        Detect mismatch between texts and labels

        Returns:
            None
    """

    texts = ["text1", "text2"]
    labels = ["A00"]

    result = run_data_quality(texts=texts, labels=labels)

    assert result["errors"] > 0

def test_data_quality_strict_mode() -> None:
    """
        Strict mode should raise error on anomalies

        Returns:
            None
    """

    texts = ["", "bad data"]
    labels = ["A00", "B00"]

    with pytest.raises(Exception):
        run_data_quality(
            texts=texts,
            labels=labels,
            strict=True,
        )
        
## ============================================================
## DATA DRIFT TESTS (ICD10)
## ============================================================
def test_data_drift_no_drift_icd10() -> None:
    """
        Validate no drift scenario on ICD10 dataset
    """

    df_ref = pd.DataFrame({
        "text": ["a", "b", "c"],
        "label": ["A00", "B00", "C00"],
        "prediction": ["A00", "B00", "C00"],
    })

    df_cur = pd.DataFrame({
        "text": ["a", "b", "c"],
        "label": ["A00", "B00", "C00"],
        "prediction": ["A00", "B00", "C00"],
    })

    result = run_data_drift(df_ref=df_ref, df_current=df_cur)

    assert result["drift_score"] >= 0.9
    assert result["errors"] == 0

def test_data_drift_detected_labels_icd10() -> None:
    """
        Detect drift on ICD10 labels
    """

    df_ref = pd.DataFrame({
        "text": ["short", "short"],
        "label": ["A00", "A00"],
        "prediction": ["A00", "A00"],
    })

    df_cur = pd.DataFrame({
        "text": ["long text", "long text"],
        "label": ["B00", "B00"],
        "prediction": ["B00", "B00"],
    })

    result = run_data_drift(df_ref=df_ref, df_current=df_cur)

    assert result["drift_score"] < 1.0
    assert result["warnings"] > 0

def test_data_drift_empty_icd10() -> None:
    """
        Validate empty dataset handling
    """

    df_ref = pd.DataFrame()
    df_cur = pd.DataFrame()

    with pytest.raises(Exception):
        run_data_drift(df_ref=df_ref, df_current=df_cur)

def test_data_drift_strict_icd10() -> None:
    """
        Validate strict mode behavior
    """

    df_ref = pd.DataFrame({"text": ["a"], "label": ["A00"]})
    df_cur = pd.DataFrame({"text": ["very long text"], "label": ["B00"]})

    with pytest.raises(Exception):
        run_data_drift(df_ref=df_ref, df_current=df_cur, strict=True)