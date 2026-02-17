'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Post-processing utilities for ICD10 predictions: probability thresholding, top-k selection and basic hierarchy constraints."
'''

from __future__ import annotations

## Standard library imports
from typing import Dict, List, Tuple

## Third-party imports
import numpy as np

## Internal imports
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("postprocess", log_file="postprocess.log")

## ============================================================
## PROBABILITY UTILITIES
## ============================================================
def softmax(logits: np.ndarray) -> np.ndarray:
    """
        Compute softmax probabilities from logits

        Args:
            logits: Raw model outputs (n_samples, n_classes)

        Returns:
            Probability matrix
    """

    ## Numerical stability trick
    logits_stable = logits - np.max(logits, axis=1, keepdims=True)

    exp_scores = np.exp(logits_stable)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return probs

## ============================================================
## TOP-K SELECTION
## ============================================================
def select_top_k(
    probabilities: np.ndarray,
    labels: List[str],
    top_k: int = 5,
) -> List[List[Tuple[str, float]]]:
    """
        Select top-k predictions per sample

        Args:
            probabilities: Probability matrix (n_samples, n_classes)
            labels: List of label names aligned with columns
            top_k: Number of predictions to keep

        Returns:
            List of list of (label, confidence)
    """

    logger.debug("Selecting top_k=%d predictions", top_k)

    ## ================= VALIDATION =================
    if probabilities is None:
        raise ValueError("probabilities is None")

    if not isinstance(probabilities, np.ndarray):
        raise TypeError("probabilities must be a numpy.ndarray")

    if probabilities.ndim != 2:
        raise ValueError(f"probabilities must be 2D, got ndim={probabilities.ndim}")

    if probabilities.shape[0] == 0:
        raise ValueError("probabilities has 0 rows")

    if probabilities.shape[1] == 0:
        raise ValueError("probabilities has 0 columns")

    if len(labels) != probabilities.shape[1]:
        raise ValueError(
            f"labels length mismatch: labels={len(labels)} vs probs_cols={probabilities.shape[1]}"
        )

    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    ## =================================================

    results: List[List[Tuple[str, float]]] = []

    for row in probabilities:
        ## Get indices sorted by descending probability
        top_indices = np.argsort(row)[::-1][:top_k]

        sample_preds: List[Tuple[str, float]] = []

        for idx in top_indices:
            sample_preds.append((labels[idx], float(row[idx])))

        results.append(sample_preds)

    return results
    

## ============================================================
## THRESHOLDING
## ============================================================
def apply_thresholds(
    probabilities: np.ndarray,
    labels: List[str],
    thresholds: Dict[str, float],
) -> List[List[Tuple[str, float]]]:
    """
        Apply per-label probability thresholds

        Args:
            probabilities: Probability matrix (n_samples, n_classes)
            labels: List of label names
            thresholds: Dict mapping label -> threshold

        Returns:
            Filtered predictions per sample
    """

    logger.debug("Applying per-label thresholds")

    results: List[List[Tuple[str, float]]] = []

    for row in probabilities:
        sample_preds: List[Tuple[str, float]] = []

        for idx, prob in enumerate(row):
            label = labels[idx]
            threshold = thresholds.get(label, 0.5)

            if prob >= threshold:
                sample_preds.append((label, float(prob)))

        ## Sort remaining predictions by descending confidence
        sample_preds.sort(key=lambda x: x[1], reverse=True)

        results.append(sample_preds)

    return results

## ============================================================
## HIERARCHY CONSTRAINTS (OPTIONAL)
## ============================================================
def enforce_parent_presence(
    predictions: List[List[Tuple[str, float]]],
    taxonomy,
) -> List[List[Tuple[str, float]]]:
    """
        Ensure parent ICD10 codes are present if child codes predicted

        Args:
            predictions: List of predictions per sample
            taxonomy: ICD10Taxonomy instance

        Returns:
            Updated predictions
    """

    updated: List[List[Tuple[str, float]]] = []

    for sample in predictions:
        labels_present = {label for label, _ in sample}
        augmented = list(sample)

        for label, score in sample:
            parent = taxonomy.get_parent(label) if taxonomy else None

            ## Add parent if missing
            if parent and parent not in labels_present:
                augmented.append((parent, score * 0.9))

        ## Deduplicate and sort
        augmented = list({lbl: sc for lbl, sc in augmented}.items())
        augmented.sort(key=lambda x: x[1], reverse=True)

        updated.append(augmented)

    return updated

## ============================================================
## HIGH-LEVEL POSTPROCESS
## ============================================================
def postprocess_predictions(
    logits_or_probs: np.ndarray,
    labels: List[str],
    top_k: int = 5,
    thresholds: Dict[str, float] | None = None,
    taxonomy=None,
    apply_softmax: bool = False,
) -> List[List[Tuple[str, float]]]:
    """
        Complete post-processing pipeline

        Steps:
            1) Optional softmax
            2) Threshold filtering OR top-k selection
            3) Optional hierarchy enforcement

        Args:
            logits_or_probs: Model output matrix
            labels: Label list
            top_k: Number of predictions if thresholds not used
            thresholds: Optional per-label thresholds
            taxonomy: Optional ICD10Taxonomy
            apply_softmax: Whether input is raw logits

        Returns:
            Final predictions per sample
    """

    logger.info("Postprocessing predictions")

    ## Convert logits to probabilities if required
    if apply_softmax:
        probabilities = softmax(logits_or_probs)
    else:
        probabilities = logits_or_probs

    ## Apply thresholding or top-k
    if thresholds:
        predictions = apply_thresholds(
            probabilities=probabilities,
            labels=labels,
            thresholds=thresholds,
        )
    else:
        predictions = select_top_k(
            probabilities=probabilities,
            labels=labels,
            top_k=top_k,
        )

    ## Apply hierarchy constraints if provided
    if taxonomy is not None:
        predictions = enforce_parent_presence(
            predictions=predictions,
            taxonomy=taxonomy,
        )

    return predictions