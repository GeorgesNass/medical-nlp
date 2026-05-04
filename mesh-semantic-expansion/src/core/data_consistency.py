'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized data consistency checks for NLP pipelines: text, embeddings, metadata, cross-source, business rules and quality."
'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.config import get_config
from src.nlp.preprocess import normalize_medical_text
from src.utils.logging_utils import get_logger
from src.utils.data_utils import (
    normalize_data,
    validate_schema,
    validate_types,
    compare_sources,
    check_business_rules,
    compute_quality_score,
    detect_duplicates,
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
def _create_issue(
    rule: str,
    level: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
        Create standardized issue object

        Args:
            rule: Rule name
            level: Severity level
            message: Description
            details: Optional metadata

        Returns:
            Issue dictionary
    """

    ## Build issue object
    issue = {
        "rule": rule,
        "level": level,
        "message": message,
        "details": details or {},
    }

    logger.debug(f"Issue created: {rule}")

    return issue

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

    ## Create issue
    issue = _create_issue(rule, level, message, details)

    ## Append to list
    issues.append(issue)

    ## Log issue
    if level == "error":
        logger.error(f"{rule} - {message}")
    else:
        logger.warning(f"{rule} - {message}")

## ============================================================
## VALIDATIONS
## ============================================================
def _validate_file(
    file_path: Optional[str | Path],
    issues: List[Dict[str, Any]],
) -> Optional[Path]:
    """
        Validate file input

        Args:
            file_path: Path input
            issues: Issue list

        Returns:
            Path or None
    """

    ## Skip if no file provided
    if file_path is None:
        return None

    path = Path(file_path)

    ## Check existence
    if not path.exists():
        logger.error(f"File not found: {path}")
        _add_issue(issues, "file_exists", "error", "File does not exist", {"file": str(path)})
        return None

    ## Check file type
    if not path.is_file():
        logger.error(f"Invalid file path: {path}")
        _add_issue(issues, "file_type", "error", "Path is not a file")
        return None

    return path

def _validate_text(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate text content

        Args:
            data: Input data
            issues: Issue list
    """

    ## Extract text
    text = data.get("text", "")

    ## Normalize
    config = get_config()

    if config.feature_engineering.enabled:
        normalized = normalize_medical_text(text)
    else:
        normalized = normalize_data({"text": text}).get("text", "")
        
    data["text"] = normalized

    ## Check empty
    if not normalized:
        logger.error("Empty text detected")
        _add_issue(issues, "text_empty", "error", "Text is empty")

    ## Check minimal length
    if len(normalized) < 3:
        logger.warning("Text too short")
        _add_issue(issues, "text_short", "warning", "Text too short")

def _validate_embeddings(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate embeddings consistency

        Args:
            data: Input data
            issues: Issue list
    """

    embeddings = data.get("embeddings")

    ## Skip if not present
    if embeddings is None:
        return

    ## Check type
    if not isinstance(embeddings, list):
        logger.error("Embeddings must be list")
        _add_issue(issues, "embedding_type", "error", "Embeddings must be a list")
        return

    ## Check empty
    if len(embeddings) == 0:
        logger.error("Empty embeddings")
        _add_issue(issues, "embedding_empty", "error", "Embeddings cannot be empty")
        return

    ## Check numeric values
    if not all(isinstance(v, (int, float)) for v in embeddings):
        logger.error("Invalid embedding values")
        _add_issue(issues, "embedding_values", "error", "Embeddings must be numeric")

    ## Check dimension consistency (optional standard size)
    if len(embeddings) < 10:
        logger.warning("Embedding dimension too small")
        _add_issue(issues, "embedding_dim", "warning", "Embedding dimension suspiciously small")

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

    ## Schema validation
    schema_issues = validate_schema(data)
    for s in schema_issues:
        _add_issue(issues, s["rule"], s["level"], s["message"])

    ## Type validation
    type_issues = validate_types(data)
    for t in type_issues:
        _add_issue(issues, t["rule"], t["level"], t["message"])

def _validate_cross_source(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate cross-source consistency

        Args:
            data: Input data
            issues: Issue list
    """

    results = compare_sources(data)

    for r in results:
        _add_issue(issues, r["rule"], r["level"], r["message"], r.get("details"))

def _validate_business(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Apply business rules

        Args:
            data: Input data
            issues: Issue list
    """

    results = check_business_rules(data)

    for r in results:
        _add_issue(issues, r["rule"], r["level"], r["message"], r.get("details"))

def _validate_duplicates(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Detect duplicates

        Args:
            data: Input data
            issues: Issue list
    """

    duplicates = detect_duplicates(data)

    if duplicates:
        logger.warning("Duplicates detected")
        _add_issue(issues, "duplicates", "warning", "Duplicate values detected", {"values": duplicates})

def _validate_feature_quality(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate feature-related properties

        Args:
            data: Input data
            issues: Issue list
    """

    text = data.get("text", "")

    ## Long text
    if len(text) > 10000:
        _add_issue(
            issues,
            "text_too_long",
            "warning",
            "Text unusually long",
            {"length": len(text)},
        )

    ## Token check
    tokens = text.split()

    if len(tokens) == 0:
        _add_issue(
            issues,
            "token_empty",
            "error",
            "No tokens after preprocessing",
        )

    logger.debug("Feature engineering consistency checks applied")
        
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
            Score
    """

    score = compute_quality_score(data)
    logger.debug(f"Quality score: {score}")
    return score

## ============================================================
## MAIN ENTRYPOINT
## ============================================================
def run_data_consistency(
    data: Dict[str, Any],
    file_path: Optional[str | Path] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
        Run full consistency pipeline

        High-level workflow:
            1) Normalize data
            2) Validate file
            3) Validate text
            4) Validate embeddings
            5) Validate schema and types
            6) Validate cross-source consistency
            7) Apply business rules
            8) Detect duplicates
            9) Compute quality score

        Args:
            data: Input data
            file_path: Optional file path
            strict: Raise error if inconsistency

        Returns:
            Result dictionary
    """

    ## Initialize issues
    issues: List[Dict[str, Any]] = []

    try:
        ## Normalize data
        data = normalize_data(data)

        ## Validate file
        path = _validate_file(file_path, issues)

        ## Validate text
        _validate_text(data, issues)

        ## Validate embeddings
        _validate_embeddings(data, issues)

        ## Validate structure
        _validate_structure(data, issues)

        ## Validate cross-source
        _validate_cross_source(data, issues)

        ## Apply business rules
        _validate_business(data, issues)

        ## Detect duplicates
        _validate_duplicates(data, issues)
        
        _validate_feature_quality(data, issues)

        ## Compute quality
        quality_score = _compute_quality(data)

        ## Extract errors
        errors = [i for i in issues if i["level"] == "error"]

        ## Build result
        result = {
            "is_consistent": len(errors) == 0,
            "errors": len(errors),
            "warnings": len(issues) - len(errors),
            "quality_score": quality_score,
            "issues": issues,
            "file": str(path) if path else None,
        }

        logger.info(f"Consistency result: {result['is_consistent']}")

        ## Strict mode
        if strict and errors:
            logger.error("Strict mode failure")
            raise ValidationError("Data consistency failed")

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        raise DataError("Consistency pipeline failed") from exc