'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data quality checks for clustering: embeddings validation, z-score, IQR and anomaly scoring."
'''

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.utils.logging_utils import get_logger
from src.utils.stats_utils import compute_mean_std, compute_iqr_bounds
from src.utils.utils import normalize_clinical_text

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

    if level == "error":
        logger.error(f"{rule} - {message}")
    else:
        logger.warning(f"{rule} - {message}")

def _compute_norms(vectors: List[np.ndarray]) -> np.ndarray:
    """
        Compute vector norms

        Args:
            vectors: List of embedding vectors

        Returns:
            Norm array
    """

    return np.array([float(np.linalg.norm(v)) for v in vectors])

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
    embeddings: List[np.ndarray],
    method: str = "zscore",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    strict: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
        Run data quality checks for clustering embeddings

        High-level workflow:
            1) Validate embeddings list
            2) Optionally validate associated text (feature engineering)
            3) Check empty / invalid vectors
            4) Check NaN / inf values
            5) Analyze embedding norms
            6) Detect statistical anomalies
            7) Compute global score

        Args:
            embeddings: List of embedding vectors
            method: Detection method (zscore or iqr)
            z_threshold: Z-score threshold
            iqr_multiplier: IQR multiplier
            strict: Raise error if invalid
            **kwargs: Optional parameters (e.g., text for FE validation)

        Returns:
            Result dictionary
    """

    issues: List[Dict[str, Any]] = []

    try:

        ## Feature engineering - optional text validation
        text = kwargs.get("text", None)

        if isinstance(text, str) and text:
            normalized_text = normalize_clinical_text(text)

            if len(normalized_text) < 3:
                _add_issue(
                    issues,
                    "text_too_short",
                    "warning",
                    "Normalized text too short",
                )

            if len(normalized_text.split()) < 2:
                _add_issue(
                    issues,
                    "low_token_count",
                    "warning",
                    "Very low token count",
                )

        ## BASIC VALIDATION
        if not embeddings:
            raise ValidationError("Empty embeddings dataset")

        ## EMPTY / SHAPE CHECK
        for idx, vec in enumerate(embeddings):
            if vec is None or not isinstance(vec, np.ndarray):
                _add_issue(
                    issues,
                    "invalid_vector",
                    "error",
                    f"Invalid embedding at index {idx}",
                )
                continue

            if vec.size == 0:
                _add_issue(
                    issues,
                    "empty_vector",
                    "error",
                    f"Empty embedding at index {idx}",
                )

        ## NaN / INF CHECK
        for idx, vec in enumerate(embeddings):
            if isinstance(vec, np.ndarray):
                if np.isnan(vec).any() or np.isinf(vec).any():
                    _add_issue(
                        issues,
                        "nan_inf_vector",
                        "error",
                        f"NaN or Inf detected at index {idx}",
                    )

        ## NORM ANALYSIS
        norms = _compute_norms(embeddings)

        if method == "zscore":
            anomaly_mask = _detect_zscore(norms, z_threshold)
        elif method == "iqr":
            anomaly_mask = _detect_iqr(norms, iqr_multiplier)
        else:
            raise ValidationError("Invalid anomaly method")

        if anomaly_mask.any():
            _add_issue(
                issues,
                "embedding_norm_anomaly",
                "warning",
                "Abnormal embedding norm detected",
                {"count": int(anomaly_mask.sum())},
            )

        ## GLOBAL SCORE
        error_count = len([i for i in issues if i["level"] == "error"])
        total = len(embeddings)

        score = 1.0 - (error_count / max(total, 1))

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