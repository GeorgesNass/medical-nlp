'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data quality checks for clinical NER: sequence validation, BIO tag consistency, z-score, IQR and anomaly scoring."
'''

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.core.config import get_config
from src.nlp.normalization import normalize_clinical_text
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

def _is_valid_bio_tag(tag: str) -> bool:
    """
        Validate a BIO tag format

        Args:
            tag: Raw tag value

        Returns:
            True if tag format is valid
    """

    ## normalize tag
    normalized = str(tag).strip()

    ## accept O directly
    if normalized == "O":
        return True

    ## validate BIO prefixed tags
    if normalized.startswith("B-") or normalized.startswith("I-"):
        return len(normalized.split("-", maxsplit=1)) == 2 and normalized.split("-", maxsplit=1)[1].strip() != ""

    return False

def _has_invalid_bio_transition(tags: List[str]) -> bool:
    """
        Detect invalid BIO transitions

        Args:
            tags: Sequence labels

        Returns:
            True if a transition is invalid
    """

    ## iterate over tags
    previous_type = None
    previous_prefix = "O"

    for raw_tag in tags:
        tag = str(raw_tag).strip()

        ## skip O tags
        if tag == "O":
            previous_type = None
            previous_prefix = "O"
            continue

        ## reject malformed tags
        if not _is_valid_bio_tag(tag):
            return True

        prefix, entity_type = tag.split("-", maxsplit=1)

        ## I- cannot start an entity from O or another entity type
        if prefix == "I":
            if previous_prefix == "O":
                return True
            if previous_type != entity_type:
                return True

        previous_type = entity_type
        previous_prefix = prefix

    return False

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
    labels: List[List[str]],
    method: str = "zscore",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    strict: bool = False,
) -> Dict[str, Any]:
    """
        Run data quality checks for clinical NER dataset

        High-level workflow:
            1) Validate sequence inputs
            2) Detect empty / missing values
            3) Analyze sequence length distribution
            4) Validate BIO label consistency
            5) Detect statistical anomalies
            6) Compute global score

        Args:
            texts: List of input texts
            labels: List of BIO tag sequences
            method: Detection method (zscore or iqr)
            z_threshold: Z-score threshold
            iqr_multiplier: IQR multiplier
            strict: Raise error if invalid

        Returns:
            Result dictionary
    """

    issues: List[Dict[str, Any]] = []

    config = get_config()

    if config.feature_engineering.enabled:
        texts = [normalize_clinical_text(t) for t in texts]
        
    try:
        ## BASIC VALIDATION
        if not texts or not labels:
            raise ValidationError("Empty dataset")

        if len(texts) != len(labels):
            raise ValidationError("Texts and labels size mismatch")

        ## TEXT VALIDATION
        empty_mask = [text is None or str(text).strip() == "" for text in texts]

        if any(empty_mask):
            _add_issue(
                issues,
                "empty_text",
                "error",
                "Empty or missing text detected",
                {"count": int(sum(empty_mask))},
            )

        ## SEQUENCE LENGTH ANALYSIS
        lengths = np.array([len(str(text).split()) for text in texts], dtype=float)

        if config.feature_engineering.enabled:
            avg_token_lengths = np.array(
                [
                    np.mean([len(w) for w in str(text).split()]) if str(text).split() else 0
                    for text in texts
                ],
                dtype=float,
            )
    
        if method == "zscore":
            anomaly_mask = _detect_zscore(lengths, z_threshold)
        elif method == "iqr":
            anomaly_mask = _detect_iqr(lengths, iqr_multiplier)
        else:
            raise ValidationError("Invalid anomaly method")

        if anomaly_mask.any():
            _add_issue(
                issues,
                "sequence_length_anomaly",
                "warning",
                "Abnormal sequence length detected",
                {"count": int(anomaly_mask.sum())},
            )

        ## FEATURE ENGINEERING: TOKEN LENGTH ANOMALY
        if config.feature_engineering.enabled:
            if method == "zscore":
                token_anomaly = _detect_zscore(avg_token_lengths, z_threshold)
            else:
                token_anomaly = _detect_iqr(avg_token_lengths, iqr_multiplier)

            if token_anomaly.any():
                _add_issue(
                    issues,
                    "token_length_anomaly",
                    "warning",
                    "Abnormal token length detected",
                    {"count": int(token_anomaly.sum())},
                )
                
        ## LABEL CONSISTENCY CHECK
        invalid_label_count = 0
        invalid_transition_count = 0

        for index, tag_sequence in enumerate(labels):
            ## validate labels container
            if not isinstance(tag_sequence, list):
                invalid_label_count += 1
                continue

            ## validate raw tag format
            if any(not _is_valid_bio_tag(tag) for tag in tag_sequence):
                invalid_label_count += 1

            ## validate transitions
            if _has_invalid_bio_transition(tag_sequence):
                invalid_transition_count += 1

            ## optional sequence/token alignment heuristic
            token_count = len(str(texts[index]).split())

            if config.feature_engineering.enabled:
                token_count = len(str(texts[index]).split())
                logger.debug("Feature engineering applied in data_quality")

            if token_count != len(tag_sequence):
                _add_issue(
                    issues,
                    "sequence_alignment_warning",
                    "warning",
                    "Token count and label count mismatch detected",
                    {
                        "index": index,
                        "token_count": token_count,
                        "label_count": len(tag_sequence),
                    },
                )

        if invalid_label_count > 0:
            _add_issue(
                issues,
                "invalid_bio_tag",
                "error",
                "Invalid BIO tag detected",
                {"count": invalid_label_count},
            )

        if invalid_transition_count > 0:
            _add_issue(
                issues,
                "invalid_bio_transition",
                "warning",
                "Invalid BIO transition detected",
                {"count": invalid_transition_count},
            )

        ## GLOBAL SCORE
        error_count = len([issue for issue in issues if issue["level"] == "error"])
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