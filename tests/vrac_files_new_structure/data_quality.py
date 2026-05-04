'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data quality checks for ICD10 prediction: text validation, label distribution, z-score, IQR, anomaly scoring."
'''

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np

from src.utils.logging_utils import get_logger
from src.utils.stats_utils import compute_mean_std, compute_iqr_bounds

try:
    from src.core.errors import ValidationError, DataError
except Exception:
    ValidationError = ValueError
    DataError = RuntimeError

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("data_quality")

## ============================================================
## HELPERS
## ============================================================
def _add_issue(
    issues: List[Dict[str, Any]],
    rule: str,
    level: str,
    message: str,
    details: Dict[str, Any] | None = None,
) -> None:
    """
        Append issue and log it

        Args:
            issues: Issue container
            rule: Rule identifier
            level: error or warning
            message: Issue message
            details: Optional metadata

        Returns:
            None
    """

    issue = {
        "rule": rule,
        "level": level,
        "message": message,
        "details": details or {},
    }

    issues.append(issue)

    ## log
    if level == "error":
        logger.error(f"{rule} - {message}")
    else:
        logger.warning(f"{rule} - {message}")

## ============================================================
## DETECTION METHODS
## ============================================================
def _detect_zscore(arr: np.ndarray, threshold: float) -> np.ndarray:
    """
        Detect anomalies using z-score

        Args:
            arr: Input array
            threshold: Z-score threshold

        Returns:
            Boolean mask
    """

    mean, std = compute_mean_std(arr)

    if std == 0:
        return np.zeros_like(arr, dtype=bool)

    z = (arr - mean) / std

    return np.abs(z) > threshold

def _detect_iqr(arr: np.ndarray, multiplier: float) -> np.ndarray:
    """
        Detect anomalies using IQR

        Args:
            arr: Input array
            multiplier: IQR multiplier

        Returns:
            Boolean mask
    """

    lower, upper = compute_iqr_bounds(arr, multiplier)

    return (arr < lower) | (arr > upper)

## ============================================================
## MAIN FUNCTION
## ============================================================
def run_data_quality(
    texts: List[str],
    labels: List[str],
    method: str = "zscore",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    strict: bool = False,
) -> Dict[str, Any]:
    """
        Run data quality checks for ICD10 dataset

        High-level workflow:
            1) Validate text inputs
            2) Detect empty / NaN values
            3) Analyze text length distribution
            4) Analyze label distribution
            5) Detect statistical anomalies
            6) Compute global score

        Args:
            texts: List of input texts
            labels: List of ICD10 labels
            method: Detection method (zscore or iqr)
            z_threshold: Z-score threshold
            iqr_multiplier: IQR multiplier
            strict: Raise error if invalid

        Returns:
            Result dictionary
    """

    issues: List[Dict[str, Any]] = []

    try:
        ## BASIC VALIDATION
        if not texts or not labels:
            raise ValidationError("Empty dataset")

        if len(texts) != len(labels):
            raise ValidationError("Texts and labels size mismatch")

        ## TEXT VALIDATION
        empty_mask = [t is None or str(t).strip() == "" for t in texts]

        if any(empty_mask):
            _add_issue(
                issues,
                "empty_text",
                "error",
                "Empty or missing text detected",
                {"count": int(sum(empty_mask))},
            )

        ## TEXT LENGTH ANALYSIS
        lengths = np.array([len(str(t)) for t in texts], dtype=float)

        if method == "zscore":
            anomaly_mask = _detect_zscore(lengths, z_threshold)
        elif method == "iqr":
            anomaly_mask = _detect_iqr(lengths, iqr_multiplier)
        else:
            raise ValidationError("Invalid anomaly method")

        if anomaly_mask.any():
            _add_issue(
                issues,
                "text_length_anomaly",
                "warning",
                "Abnormal text length detected",
                {"count": int(anomaly_mask.sum())},
            )

        ## LABEL DISTRIBUTION CHECK
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))

        ## detect imbalance (very simple heuristic)
        max_count = max(counts)
        min_count = min(counts)

        if min_count == 0 or max_count / max(min_count, 1) > 50:
            _add_issue(
                issues,
                "label_imbalance",
                "warning",
                "Severe label imbalance detected",
                {"distribution": distribution},
            )

        ## GLOBAL SCORE
        error_count = len([i for i in issues if i["level"] == "error"])
        total_checks = len(texts)

        score = 1.0 - (error_count / max(total_checks, 1))

        result = {
            "is_valid": error_count == 0,
            "errors": error_count,
            "warnings": len(issues) - error_count,
            "score": score,
            "issues": issues,
        }

        logger.info(f"Data quality score: {score}")

        ## STRICT MODE
        if strict and error_count > 0:
            raise ValidationError("Data quality failed")

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Data quality failure: {exc}")
        raise DataError("Data quality pipeline failed") from exc