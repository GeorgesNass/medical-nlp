'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Evaluation utilities for ICD10 prediction: classification metrics, top-k accuracy and confusion export."
'''

from __future__ import annotations

## Standard library
from pathlib import Path
from typing import Dict, List

## Third-party
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

## Internal
from src.utils.logging_utils import get_logger
from src.utils.io_utils import ensure_parent_dir, write_json

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("evaluate", log_file="evaluate.log")

## ============================================================
## BASIC METRICS
## ============================================================
def compute_basic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
        Compute core classification metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary with accuracy and F1 scores
    """

    logger.info("Computing basic metrics")

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }

    return metrics

def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
) -> Dict:
    """
        Compute full classification report

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names ordered by class index

        Returns:
            Classification report dict
    """

    logger.info("Computing classification report")

    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    return report

## ============================================================
## TOP-K METRICS
## ============================================================
def compute_top_k_accuracy(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    k: int = 5,
) -> float:
    """
        Compute top-k accuracy

        Definition:
            True label appears within top-k highest probabilities

        Args:
            probabilities: Probability matrix (n_samples, n_classes)
            y_true: True label indices
            k: Top-k value

        Returns:
            Top-k accuracy
    """

    logger.info("Computing top-%d accuracy", k)

    top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]

    correct = 0

    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1

    return correct / len(y_true)

## ============================================================
## CONFUSION MATRIX
## ============================================================
def compute_confusion_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
) -> pd.DataFrame:
    """
        Compute confusion matrix as DataFrame

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names

        Returns:
            Confusion matrix DataFrame
    """

    logger.info("Computing confusion matrix")

    cm = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    return df_cm

## ============================================================
## EXPORT UTILITIES
## ============================================================
def export_metrics(
    metrics: Dict,
    output_path: str | Path,
) -> Path:
    """
        Export metrics dictionary to JSON

        Args:
            metrics: Metrics dictionary
            output_path: Output JSON path

        Returns:
            Saved path
    """

    path = ensure_parent_dir(output_path)

    write_json(metrics, path)

    logger.info("Metrics exported to %s", path)

    return path

def export_confusion_csv(
    df_confusion: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
        Export confusion matrix to CSV

        Args:
            df_confusion: Confusion matrix DataFrame
            output_path: Output CSV path

        Returns:
            Saved path
    """

    path = ensure_parent_dir(output_path)

    df_confusion.to_csv(path, index=True)

    logger.info("Confusion matrix exported to %s", path)

    return path