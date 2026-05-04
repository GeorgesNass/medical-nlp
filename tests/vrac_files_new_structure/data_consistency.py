'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data consistency for ICD10 classification: text, labels, format, duplicates and coherence"
'''

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

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
            level: Severity level
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

    ## Log issue
    if level == "error":
        logger.error(f"{rule} - {message}")
    else:
        logger.warning(f"{rule} - {message}")

## ============================================================
## VALIDATIONS
## ============================================================
def _validate_text(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate text field

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    text = data.get("text", "")

    ## Normalize text
    normalized = normalize_data({"text": text}).get("text", "")
    data["text"] = normalized

    ## Check empty
    if not normalized:
        _add_issue(issues, "text_empty", "error", "Text is empty")

    ## Check length
    if len(normalized) < 3:
        _add_issue(issues, "text_short", "warning", "Text too short")

def _validate_labels(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate ICD10 labels

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    labels = data.get("labels")

    ## Missing labels
    if labels is None:
        _add_issue(issues, "labels_missing", "error", "Labels are required")
        return

    ## Type check
    if not isinstance(labels, list):
        _add_issue(issues, "labels_type", "error", "Labels must be a list")
        return

    ## Empty list
    if len(labels) == 0:
        _add_issue(issues, "labels_empty", "error", "Labels list is empty")
        return

    ## ICD10 pattern
    pattern = re.compile(r"^[A-Z][0-9]{2}(\.[0-9])?$")

    for idx, label in enumerate(labels):

        ## Type check
        if not isinstance(label, str):
            _add_issue(
                issues,
                "label_type",
                "error",
                "Label must be string",
                {"index": idx},
            )
            continue

        ## Format check
        if not pattern.match(label):
            _add_issue(
                issues,
                "label_format",
                "error",
                "Invalid ICD10 format",
                {"label": label},
            )

def _validate_duplicates(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Detect duplicate labels

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    labels = data.get("labels", [])

    duplicates = len(labels) - len(set(labels))

    if duplicates > 0:
        _add_issue(
            issues,
            "label_duplicates",
            "warning",
            "Duplicate labels detected",
            {"count": duplicates},
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

    ## Validate schema
    for s in validate_schema(data):
        _add_issue(issues, s["rule"], s["level"], s["message"])

    ## Validate types
    for t in validate_types(data):
        _add_issue(issues, t["rule"], t["level"], t["message"])

## ============================================================
## QUALITY
## ============================================================
def _compute_quality(
    data: Dict[str, Any],
) -> float:
    """
        Compute quality score

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
        Run data consistency pipeline for ICD10 classification

        Args:
            data: Input data
            strict: Raise error if inconsistency

        Returns:
            Dict[str, Any]
    """

    issues: List[Dict[str, Any]] = []

    try:
        ## Normalize data
        data = normalize_data(data)

        ## Validate text
        _validate_text(data, issues)

        ## Validate labels
        _validate_labels(data, issues)

        ## Validate duplicates
        _validate_duplicates(data, issues)

        ## Validate structure
        _validate_structure(data, issues)

        ## Compute quality score
        quality_score = _compute_quality(data)

        ## Count errors
        errors = [i for i in issues if i["level"] == "error"]

        ## Build result
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