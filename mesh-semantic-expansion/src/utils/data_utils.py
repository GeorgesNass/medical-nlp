'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Utility functions for NLP data consistency: normalization, schema, types, cross-source, business rules and quality."
'''

from __future__ import annotations

import re
from typing import Any, Dict, List

from pathlib import Path
from src.core.config import get_config
from src.nlp.preprocess import normalize_medical_text
from src.utils import get_logger

## ============================================================
## LOGGER INITIALIZATION
## ============================================================
logger = get_logger("data_utils")

def normalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
        Normalize data fields

        Args:
            data: Input dictionary

        Returns:
            Normalized dictionary
    """

    ## Initialize container
    normalized = {}

    for key, value in data.items():

        ## Normalize string
        if isinstance(value, str):
            config = get_config()

            if config.feature_engineering.enabled:
                value = normalize_medical_text(value)
                logger.debug("Feature engineering applied in normalize_data")
            else:
                value = value.strip().lower()
                value = re.sub(r"\s+", " ", value)
                logger.debug(f"Normalizing string: {key}")

        ## Normalize list
        if isinstance(value, list):
            logger.debug(f"Normalizing list: {key}")
            config = get_config()

            if config.feature_engineering.enabled:
                value = [
                    normalize_medical_text(v) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                value = [
                    v.strip().lower() if isinstance(v, str) else v
                    for v in value
                ]

        ## Store
        normalized[key] = value

    return normalized

def validate_schema(data: Dict[str, Any]) -> List[Dict]:
    """
        Validate required NLP fields

        Args:
            data: Input dictionary

        Returns:
            List of issues
    """

    issues = []

    ## Required NLP fields
    required_fields = ["text"]

    for field in required_fields:

        ## Check missing
        if field not in data:
            logger.error(f"Missing field: {field}")
            issues.append({
                "rule": "schema",
                "level": "error",
                "message": f"Missing field: {field}",
            })

    return issues

def validate_types(data: Dict[str, Any]) -> List[Dict]:
    """
        Validate field types

        Args:
            data: Input dictionary

        Returns:
            List of issues
    """

    issues = []

    ## Text type
    if "text" in data and not isinstance(data["text"], str):
        logger.error("Invalid text type")
        issues.append({
            "rule": "type_text",
            "level": "error",
            "message": "text must be string",
        })

    ## Embeddings type
    if "embeddings" in data and not isinstance(data["embeddings"], list):
        logger.error("Invalid embeddings type")
        issues.append({
            "rule": "type_embeddings",
            "level": "error",
            "message": "embeddings must be list",
        })

    return issues

def compare_sources(data: Dict[str, Any]) -> List[Dict]:
    """
        Compare multiple sources (text vs metadata)

        Args:
            data: Input dictionary

        Returns:
            List of issues
    """

    issues = []

    ## Compare text sources
    if "text" in data and "metadata_text" in data:

        if data["text"] != data["metadata_text"]:
            logger.warning("Mismatch text vs metadata")
            issues.append({
                "rule": "cross_text",
                "level": "warning",
                "message": "Mismatch between text and metadata_text",
            })

    return issues

def check_business_rules(data: Dict[str, Any]) -> List[Dict]:
    """
        Apply NLP business rules

        Args:
            data: Input dictionary

        Returns:
            List of issues
    """

    issues = []

    ## Text length rule
    if "text" in data:
        if len(data["text"]) < 3:
            logger.warning("Text too short")
            issues.append({
                "rule": "business_text_length",
                "level": "warning",
                "message": "Text too short",
            })

    ## Embedding length rule
    if "embeddings" in data:
        if isinstance(data["embeddings"], list) and len(data["embeddings"]) < 10:
            logger.warning("Embedding too small")
            issues.append({
                "rule": "business_embedding_dim",
                "level": "warning",
                "message": "Embedding dimension too small",
            })

    return issues

def compute_quality_score(data: Dict[str, Any]) -> float:
    """
        Compute quality score

        Args:
            data: Input dictionary

        Returns:
            Score
    """

    text = data.get("text", "")

    ## Empty case
    if not text:
        logger.warning("Empty text for scoring")
        return 0.0

    ## Ratio alphanumeric
    valid_chars = sum(c.isalnum() for c in text)
    score = valid_chars / len(text)

    logger.debug(f"Quality score: {score}")

    return score

def detect_duplicates(data: Dict[str, Any]) -> List[Any]:
    """
        Detect duplicate values

        Args:
            data: Input dictionary

        Returns:
            List of duplicates
    """

    seen = set()
    duplicates = []

    for value in data.values():

        ## Skip complex types
        if isinstance(value, (list, dict)):
            continue

        if value in seen:
            logger.warning(f"Duplicate detected: {value}")
            duplicates.append(value)
        else:
            seen.add(value)

    return duplicates
    
## ============================================================
## FEATURE ENGINEERING IO HELPERS
## ============================================================
def load_and_normalize_text_from_path(
    file_path: str | Path,
    encoding: str = "utf-8",
) -> str:
    """
        Load and normalize text using feature engineering pipeline

        Args:
            file_path: Input file path
            encoding: File encoding

        Returns:
            Normalized text
    """

    ## Read file
    path = Path(file_path)
    text = path.read_text(encoding=encoding)

    config = get_config()

    ## Apply feature engineering if enabled
    if config.feature_engineering.enabled:
        logger.debug("Feature engineering applied in load_and_normalize_text_from_path")
        return normalize_medical_text(text)

    return text.strip()
    
## ============================================================
## FEATURE EXPORT
## ============================================================
def export_feature_summary(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
        Export basic feature summary

        Args:
            data: Input dictionary

        Returns:
            Feature summary
    """

    text = data.get("text", "")

    return {
        "char_length": len(text),
        "token_count": len(text.split()),
        "has_embeddings": "embeddings" in data,
    }