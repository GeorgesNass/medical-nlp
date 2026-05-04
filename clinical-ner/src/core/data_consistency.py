'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data consistency for clinical NER: text, entities, spans, overlaps, BIO labels and coherence"
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.config import get_config
from src.nlp.normalization import normalize_clinical_text
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

    ## Normalize
    config = get_config()

    if config.feature_engineering.enabled:
        normalized = normalize_clinical_text(text)
    else:
        normalized = normalize_data({"text": text}).get("text", "")

    data["text"] = normalized
    
    if config.feature_engineering.enabled:
        data["text_length"] = len(normalized)
        data["token_count"] = len(normalized.split())

    ## Empty check
    if not normalized:
        _add_issue(issues, "text_empty", "error", "Text is empty")

    ## Length check
    if len(normalized) < 3:
        _add_issue(issues, "text_short", "warning", "Text too short")

def _validate_entities(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate entities structure and spans

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    entities = data.get("entities")

    if entities is None:
        _add_issue(issues, "entities_missing", "error", "Entities are required")
        return

    if not isinstance(entities, list):
        _add_issue(issues, "entities_type", "error", "Entities must be a list")
        return

    text = data.get("text", "")

    spans = []

    for idx, entity in enumerate(entities):

        if not isinstance(entity, dict):
            _add_issue(
                issues,
                "entity_format",
                "error",
                "Entity must be dict",
                {"index": idx},
            )
            continue

        start = entity.get("start")
        end = entity.get("end")
        label = entity.get("label")

        ## Check presence
        if start is None or end is None:
            _add_issue(
                issues,
                "entity_span_missing",
                "error",
                "Missing span",
                {"index": idx},
            )
            continue

        ## Check type
        if not isinstance(start, int) or not isinstance(end, int):
            _add_issue(
                issues,
                "entity_span_type",
                "error",
                "Span must be int",
                {"index": idx},
            )
            continue

        ## Bounds check
        if start < 0 or end > len(text) or start >= end:
            _add_issue(
                issues,
                "entity_span_invalid",
                "error",
                "Invalid span bounds",
                {"index": idx},
            )
            continue

        ## Label check
        if not isinstance(label, str):
            _add_issue(
                issues,
                "entity_label_type",
                "error",
                "Label must be string",
                {"index": idx},
            )
        
        config = get_config()

        if config.feature_engineering.enabled and isinstance(entity, dict):
            entity_text = text[start:end]
            entity["normalized_text"] = normalize_clinical_text(entity_text)
            entity["token_count"] = len(entity["normalized_text"].split())
            logger.debug("Feature engineering applied in data_consistency")            

        spans.append((start, end))

    ## Overlap detection
    spans_sorted = sorted(spans, key=lambda x: x[0])

    for i in range(1, len(spans_sorted)):
        prev = spans_sorted[i - 1]
        curr = spans_sorted[i]

        if curr[0] < prev[1]:
            _add_issue(
                issues,
                "entity_overlap",
                "warning",
                "Overlapping entities detected",
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
        Run data consistency pipeline for NER

        Args:
            data: Input data
            strict: Raise error if inconsistency

        Returns:
            Dict[str, Any]
    """

    issues: List[Dict[str, Any]] = []

    try:
        ## Normalize
        data = normalize_data(data)

        ## Validate text
        _validate_text(data, issues)

        ## Validate entities
        _validate_entities(data, issues)

        ## Validate structure
        _validate_structure(data, issues)

        ## Compute quality
        quality_score = _compute_quality(data)

        errors = [i for i in issues if i["level"] == "error"]

        result = {
            "is_consistent": len(errors) == 0,
            "errors": len(errors),
            "warnings": len(issues) - len(errors),
            "quality_score": quality_score,
            "issues": issues,
        }

        if strict and errors:
            raise ValidationError("Data consistency failed")

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        raise DataError("Consistency pipeline failed") from exc