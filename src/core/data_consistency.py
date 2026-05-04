'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data consistency for clustering: embeddings, text, dimensions, NaN, duplicates and coherence."
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional

import math

from src.utils.logging_utils import get_logger
from src.utils.data_utils import (
    normalize_data,
    validate_schema,
    validate_types,
    compute_quality_score,
)
from src.utils.utils import normalize_clinical_text

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
            details: Metadata
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
        Validate text coherence

        Args:
            data: Input data
            issues: Issue list
    """

    text = data.get("text", "")

    ## Normalize text
    normalized = normalize_data({"text": text}).get("text", "")

    ## Feature engineering normalization
    fe_text = normalize_clinical_text(normalized)
    data["text"] = fe_text

    data["char_length"] = len(fe_text)
    data["token_count"] = len(fe_text.split())

    ## Empty check
    if not normalized:
        _add_issue(issues, "text_empty", "error", "Text is empty")

    ## Length check
    if len(normalized) < 3:
        _add_issue(issues, "text_short", "warning", "Text too short")

def _validate_embeddings(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate embeddings structure and values

        Args:
            data: Input data
            issues: Issue list
    """

    embeddings = data.get("embeddings")

    ## Missing embeddings
    if embeddings is None:
        _add_issue(issues, "embeddings_missing", "error", "Embeddings are required")
        return

    ## Type check
    if not isinstance(embeddings, list):
        _add_issue(issues, "embeddings_type", "error", "Embeddings must be a list")
        return

    ## Empty
    if len(embeddings) == 0:
        _add_issue(issues, "embeddings_empty", "error", "Embeddings list is empty")
        return

    ## Validate first vector
    first = embeddings[0]

    if not isinstance(first, list):
        _add_issue(issues, "embedding_vector_type", "error", "Each embedding must be a list")
        return

    dim = len(first)

    ## Check dimension consistency
    for idx, vec in enumerate(embeddings):

        ## Type
        if not isinstance(vec, list):
            _add_issue(issues, "embedding_vector_type", "error", "Invalid embedding type", {"index": idx})
            continue

        ## Dimension
        if len(vec) != dim:
            _add_issue(issues, "embedding_dim", "error", "Inconsistent embedding dimension", {"index": idx})
            continue

        ## NaN / invalid values
        for v in vec:
            if not isinstance(v, (int, float)):
                _add_issue(issues, "embedding_value_type", "error", "Embedding value must be numeric")
                break

            if isinstance(v, float) and math.isnan(v):
                _add_issue(issues, "embedding_nan", "error", "NaN detected in embedding")
                break

def _validate_duplicates(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Detect duplicate embeddings

        Args:
            data: Input data
            issues: Issue list
    """

    embeddings = data.get("embeddings", [])

    seen = set()
    duplicates = 0

    for vec in embeddings:

        ## Convert to tuple for hashing
        try:
            t = tuple(vec)
        except Exception:
            continue

        if t in seen:
            duplicates += 1
        else:
            seen.add(t)

    if duplicates > 0:
        _add_issue(
            issues,
            "embedding_duplicates",
            "warning",
            "Duplicate embeddings detected",
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
    """

    for s in validate_schema(data):
        _add_issue(issues, s["rule"], s["level"], s["message"])

    for t in validate_types(data):
        _add_issue(issues, t["rule"], t["level"], t["message"])

## ============================================================
## QUALITY
## ============================================================
def _compute_quality(data: Dict[str, Any]) -> float:
    """
        Compute quality score

        Args:
            data: Input data

        Returns:
            Score
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
        Run full consistency pipeline for clustering

        Args:
            data: Input data
            strict: Raise error if inconsistency

        Returns:
            Result dictionary
    """

    issues: List[Dict[str, Any]] = []

    try:
        ## Normalize
        data = normalize_data(data)

        ## Validate text
        _validate_text(data, issues)

        ## Validate embeddings
        _validate_embeddings(data, issues)

        ## Validate duplicates
        _validate_duplicates(data, issues)

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