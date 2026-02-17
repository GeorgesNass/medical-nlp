'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Model inference utilities for ICD10 prediction: single and batch prediction with probability handling."
'''

from __future__ import annotations

## Standard library
from pathlib import Path
from typing import Any, List, Tuple

## Third-party
import numpy as np
from scipy import sparse

## Internal
from src.utils.logging_utils import get_logger
from src.core.errors import ModelError
from src.model.train import load_model

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("predict", log_file="predict.log")

## ============================================================
## INTERNAL HELPERS
## ============================================================
def _get_probabilities(model: Any, X: sparse.spmatrix | np.ndarray) -> np.ndarray:
    """
        Extract probability matrix from model

        Supports:
            - Models implementing predict_proba
            - Fallback to decision_function + softmax if needed

        Args:
            model: Trained model
            X: Feature matrix

        Returns:
            Probability matrix (n_samples, n_classes)
    """

    ## Standard scikit-learn API
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    ## Fallback: decision_function -> softmax
    if hasattr(model, "decision_function"):
        logits = model.decision_function(X)

        ## Convert 1D to 2D if binary
        if logits.ndim == 1:
            logits = np.vstack([-logits, logits]).T

        return _softmax(logits)

    raise ModelError("Model does not support probability extraction")

def _softmax(logits: np.ndarray) -> np.ndarray:
    """
        Compute softmax probabilities

        Args:
            logits: Raw logits

        Returns:
            Probability matrix
    """

    logits_stable = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(logits_stable)
    
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

## ============================================================
## PUBLIC API
## ============================================================
def predict_labels(
    model: Any,
    X: sparse.spmatrix | np.ndarray,
) -> np.ndarray:
    """
        Predict class labels

        Args:
            model: Trained model
            X: Feature matrix

        Returns:
            Predicted label indices
    """

    logger.info("Predicting labels | samples=%d", X.shape[0])

    return model.predict(X)

def predict_with_probabilities(
    model: Any,
    X: sparse.spmatrix | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
        Predict labels and return probabilities

        Args:
            model: Trained model
            X: Feature matrix

        Returns:
            Tuple (predicted_labels, probability_matrix)
    """

    logger.info("Predicting labels with probabilities")

    probs = _get_probabilities(model, X)
    preds = np.argmax(probs, axis=1)

    return preds, probs

def load_and_predict(
    model_path: str | Path,
    X: sparse.spmatrix | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
        Load model from disk and run prediction

        Args:
            model_path: Path to serialized model
            X: Feature matrix

        Returns:
            Tuple (predicted_labels, probability_matrix)
    """

    logger.info("Loading model for inference")

    model = load_model(model_path)

    return predict_with_probabilities(model, X)