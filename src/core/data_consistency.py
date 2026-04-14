'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data consistency for document classification: dataset, text, labels, balance, schema, config"
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional
from collections import Counter

from src.utils.logging_utils import get_logger
from src.utils.data_utils import (
    normalize_data,
    validate_schema,
    validate_types,
    compute_quality_score,
)

try:
    from src.core.errors import ValidationError, DataError
except Exception:
    ValidationError = ValueError
    DataError = RuntimeError

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("data_consistency")

## ============================================================
## ISSUE HANDLING
## ============================================================
def _add_issue(
    issues: List[Dict[str, Any]],
    rule: str,
    level: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
        Append issue and log it

        Args:
            issues: Issue list
            rule: Rule name
            level: Severity
            message: Description
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

    ## Log
    if level == "error":
        logger.error(f"{rule} - {message}")
    else:
        logger.warning(f"{rule} - {message}")

## ============================================================
## VALIDATIONS
## ============================================================
def _validate_dataset(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate dataset structure

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    records = data.get("records")

    if not records:
        _add_issue(issues, "dataset_missing", "error", "Dataset is required")
        return

    if not isinstance(records, list):
        _add_issue(issues, "dataset_type", "error", "Dataset must be list")
        return

    if len(records) == 0:
        _add_issue(issues, "dataset_empty", "error", "Dataset is empty")
        return

    for idx, record in enumerate(records):

        if not isinstance(record, dict):
            _add_issue(issues, "record_format", "error", "Record must be dict", {"index": idx})
            continue

        ## Text validation
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            _add_issue(issues, "text_invalid", "error", "Invalid text", {"index": idx})

        ## Label validation
        label = record.get("label")
        if label is None or not isinstance(label, str) or not label.strip():
            _add_issue(issues, "label_invalid", "error", "Invalid label", {"index": idx})

def _validate_labels(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate label distribution

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    records = data.get("records", [])

    labels = [
        r.get("label")
        for r in records
        if isinstance(r, dict) and isinstance(r.get("label"), str)
    ]

    if not labels:
        return

    counter = Counter(labels)

    ## Too many classes
    if len(counter) > 100:
        _add_issue(issues, "too_many_classes", "warning", "Too many classes")

    ## Imbalance detection
    most_common = counter.most_common(1)[0][1]
    total = sum(counter.values())

    if total > 0 and most_common / total > 0.9:
        _add_issue(
            issues,
            "label_imbalance",
            "warning",
            "Severe class imbalance",
        )

def _validate_config(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate optional config rules

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    min_length = data.get("min_text_length", 3)

    if not isinstance(min_length, int) or min_length < 1:
        _add_issue(
            issues,
            "min_length_invalid",
            "error",
            "min_text_length must be positive int",
        )

def _validate_structure(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate schema and types

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    for s in validate_schema(data):
        _add_issue(issues, s["rule"], s["level"], s["message"])

    for t in validate_types(data):
        _add_issue(issues, t["rule"], t["level"], t["message"])

## ============================================================
## QUALITY
## ============================================================
def _compute_quality(
    data: Dict[str, Any],
) -> float:
    """
        Compute dataset quality score

        Args:
            data: Input data

        Returns:
            float
    """

    return compute_quality_score(data)

## ============================================================
## MAIN ENTRYPOINT
## ============================================================
def run_data_consistency(
    data: Dict[str, Any],
    strict: bool = False,
) -> Dict[str, Any]:
    """
        Run data consistency for classification pipeline

        Args:
            data: Input data
            strict: Raise error if inconsistency

        Returns:
            Dict[str, Any]
    """

    issues: List[Dict[str, Any]] = []

    try:
        ## Normalize input
        data = normalize_data(data)

        ## Dataset validation
        _validate_dataset(data, issues)

        ## Labels validation
        _validate_labels(data, issues)

        ## Config validation
        _validate_config(data, issues)

        ## Schema / types
        _validate_structure(data, issues)

        ## Quality score
        quality_score = _compute_quality(data)

        errors = [i for i in issues if i["level"] == "error"]

        result = {
            "is_consistent": len(errors) == 0,
            "errors": len(errors),
            "warnings": len(issues) - len(errors),
            "quality_score": quality_score,
            "issues": issues,
        }

        ## Strict mode
        if strict and errors:
            raise ValidationError("Data consistency failed")

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        raise DataError("Consistency pipeline failed") from exc