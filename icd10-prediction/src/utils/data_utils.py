'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Utility functions for ICD10 classification: normalization, label validation, types and quality"
'''

from __future__ import annotations

import re
from typing import Any, Dict, List

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
            Dict[str, Any]
    """

    normalized = {}

    for key, value in data.items():

        ## Normalize string
        if isinstance(value, str):
            logger.debug(f"Normalizing string: {key}")
            value = value.strip().upper()
            value = re.sub(r"\s+", " ", value)

        ## Normalize list
        if isinstance(value, list):
            logger.debug(f"Normalizing list: {key}")
            value = [
                v.strip().upper() if isinstance(v, str) else v
                for v in value
            ]

        normalized[key] = value

    return normalized

def validate_schema(data: Dict[str, Any]) -> List[Dict]:
    """
        Validate required fields for ICD10 classification

        Args:
            data: Input dictionary

        Returns:
            List[Dict]
    """

    issues = []

    ## Required text
    if "text" not in data:
        logger.error("Missing text field")
        issues.append({
            "rule": "schema_text",
            "level": "error",
            "message": "text is required",
        })

    ## Required labels
    if "labels" not in data:
        logger.error("Missing labels field")
        issues.append({
            "rule": "schema_labels",
            "level": "error",
            "message": "labels are required",
        })

    return issues

def validate_types(data: Dict[str, Any]) -> List[Dict]:
    """
        Validate field types

        Args:
            data: Input dictionary

        Returns:
            List[Dict]
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

    ## Labels type
    if "labels" in data and not isinstance(data["labels"], list):
        logger.error("Invalid labels type")
        issues.append({
            "rule": "type_labels",
            "level": "error",
            "message": "labels must be list",
        })

    return issues

def validate_labels_format(labels: List[Any]) -> List[Dict]:
    """
        Validate ICD10 label format

        Args:
            labels: List of labels

        Returns:
            List[Dict]
    """

    issues = []

    pattern = re.compile(r"^[A-Z][0-9]{2}(\.[0-9])?$")

    for idx, label in enumerate(labels):

        ## Type check
        if not isinstance(label, str):
            logger.error(f"Invalid label type at index {idx}")
            issues.append({
                "rule": "label_type",
                "level": "error",
                "message": "Label must be string",
            })
            continue

        ## Format check
        if not pattern.match(label):
            logger.error(f"Invalid ICD10 format: {label}")
            issues.append({
                "rule": "label_format",
                "level": "error",
                "message": "Invalid ICD10 format",
            })

    return issues

def check_business_rules(data: Dict[str, Any]) -> List[Dict]:
    """
        Apply business rules

        Args:
            data: Input dictionary

        Returns:
            List[Dict]
    """

    issues = []

    ## Text length
    if "text" in data and len(data["text"]) < 3:
        logger.warning("Text too short")
        issues.append({
            "rule": "business_text_length",
            "level": "warning",
            "message": "Text too short",
        })

    ## Labels size
    if "labels" in data:
        if isinstance(data["labels"], list) and len(data["labels"]) == 0:
            logger.warning("Labels empty")
            issues.append({
                "rule": "business_labels_empty",
                "level": "warning",
                "message": "Labels list is empty",
            })

    return issues

def compute_quality_score(data: Dict[str, Any]) -> float:
    """
        Compute quality score

        Args:
            data: Input dictionary

        Returns:
            float
    """

    text = data.get("text", "")

    if not text:
        logger.warning("Empty text for scoring")
        return 0.0

    valid_chars = sum(c.isalnum() for c in text)
    score = valid_chars / len(text)

    logger.debug(f"Quality score: {score}")

    return score