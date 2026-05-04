'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data quality checks for semantic expansion: term validation, similarity scores, z-score, IQR and anomaly scoring."
'''

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.core.config import get_config
from src.nlp.preprocess import normalize_medical_text
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

    if level == "error":
        logger.error(f"{rule} - {message}")
    else:
        logger.warning(f"{rule} - {message}")

def _safe_len(text: str) -> int:
    """
        Safe length computation

        Args:
            text: Input string

        Returns:
            Length
    """

    return len(str(text))

def _compute_array(arr: List[float]) -> np.ndarray:
    """
        Convert list to numpy array safely

        Args:
            arr: Input list

        Returns:
            numpy array
    """

    return np.array(arr, dtype=float)

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
    terms: List[str],
    scores: List[float] | None = None,
    method: str = "zscore",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    strict: bool = False,
) -> Dict[str, Any]:
    """
        Run data quality checks for semantic expansion

        High-level workflow:
            1) Validate terms list
            2) Detect empty / duplicate terms
            3) Validate similarity scores
            4) Detect statistical anomalies
            5) Compute global score

        Args:
            terms: List of expanded terms
            scores: Optional similarity scores
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
        if not terms:
            raise ValidationError("Empty terms list")


        ## Feature engineering preprocessing
        config = get_config()

        if config.feature_engineering.enabled:
            terms = [normalize_medical_text(t or "") for t in terms]
            
        ## EMPTY TERMS
        empty_mask = [t is None or str(t).strip() == "" for t in terms]

        if any(empty_mask):
            _add_issue(
                issues,
                "empty_term",
                "error",
                "Empty or missing term detected",
                {"count": int(sum(empty_mask))},
            )

        ## DUPLICATES
        if len(set(terms)) != len(terms):
            _add_issue(
                issues,
                "duplicate_terms",
                "warning",
                "Duplicate terms detected",
            )

        ## SCORE CHECK
        if scores is not None:
            arr = _compute_array(scores)

            if np.isnan(arr).any() or np.isinf(arr).any():
                _add_issue(
                    issues,
                    "invalid_scores",
                    "error",
                    "NaN or Inf detected in scores",
                )

            ## ANOMALY DETECTION
            if method == "zscore":
                mask = _detect_zscore(arr, z_threshold)
            elif method == "iqr":
                mask = _detect_iqr(arr, iqr_multiplier)
            else:
                raise ValidationError("Invalid anomaly method")

            if mask.any():
                _add_issue(
                    issues,
                    "score_anomaly",
                    "warning",
                    "Abnormal similarity scores detected",
                    {"count": int(mask.sum())},
                )

        ## TOKEN LENGTH ANALYSIS
        lengths = _compute_array([_safe_len(t) for t in terms])

        if method == "zscore":
            token_mask = _detect_zscore(lengths, z_threshold)
        elif method == "iqr":
            token_mask = _detect_iqr(lengths, iqr_multiplier)
        else:
            token_mask = np.zeros_like(lengths, dtype=bool)

        if token_mask.any():
            _add_issue(
                issues,
                "term_length_anomaly",
                "warning",
                "Abnormal term length detected",
                {"count": int(token_mask.sum())},
            )
            
        ## GLOBAL SCORE
        error_count = len([i for i in issues if i["level"] == "error"])
        total = len(terms)

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

        logger.debug("Feature engineering quality checks applied | terms=%d", len(terms))

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Data quality failure: {exc}")
        raise DataError("Data quality pipeline failed") from exc