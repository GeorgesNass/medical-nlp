'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Utility functions for document classification: normalization, validation, quality scoring"
'''

from __future__ import annotations

from typing import Any, Dict, List

from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER INITIALIZATION
## ============================================================
logger = get_logger("data_utils")

def normalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
        Normalize input payload

        Args:
            data: Input dictionary

        Returns:
            Dict[str, Any]
    """

    normalized = {}

    for key, value in data.items():

        ## Normalize strings
        if isinstance(value, str):
            logger.debug(f"Normalizing string: {key}")
            value = value.strip().lower()

        ## Normalize dataset
        if key == "records" and isinstance(value, list):
            logger.debug("Normalizing records")

            normalized_records = []

            for record in value:
                if isinstance(record, dict):
                    normalized_records.append({
                        "text": str(record.get("text", "")).strip().lower(),
                        "label": str(record.get("label", "")).strip(),
                        **record,
                    })

            value = normalized_records

        normalized[key] = value

    return normalized

def validate_schema(data: Dict[str, Any]) -> List[Dict]:
    """
        Validate required fields

        Args:
            data: Input dictionary

        Returns:
            List[Dict]
    """

    issues = []

    ## Records required
    if "records" not in data:
        logger.error("Missing records field")
        issues.append({
            "rule": "schema_records",
            "level": "error",
            "message": "records are required",
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

    ## Records type
    if "records" in data and not isinstance(data["records"], list):
        logger.error("Invalid records type")
        issues.append({
            "rule": "type_records",
            "level": "error",
            "message": "records must be list",
        })

    return issues

def validate_business_rules(data: Dict[str, Any]) -> List[Dict]:
    """
        Validate classification rules

        Args:
            data: Input dictionary

        Returns:
            List[Dict]
    """

    issues = []

    records = data.get("records", [])

    ## Warn if too few records
    if isinstance(records, list) and len(records) < 2:
        logger.warning("Too few records")
        issues.append({
            "rule": "too_few_records",
            "level": "warning",
            "message": "At least 2 records recommended",
        })

    ## Warn if only one class
    labels = [
        r.get("label")
        for r in records
        if isinstance(r, dict)
    ]

    if len(set(labels)) <= 1:
        logger.warning("Single class dataset")
        issues.append({
            "rule": "single_class",
            "level": "warning",
            "message": "Only one class detected",
        })

    return issues

def compute_quality_score(data: Dict[str, Any]) -> float:
    """
        Compute quality score based on text length

        Args:
            data: Input dictionary

        Returns:
            float
    """

    records = data.get("records", [])

    if not records:
        logger.warning("No records for scoring")
        return 0.0

    total_length = 0

    for record in records:
        if isinstance(record, dict):
            text = record.get("text", "")
            if isinstance(text, str):
                total_length += len(text)

    if total_length == 0:
        return 0.0

    ## Normalize score
    score = min(total_length / 1000.0, 1.0)

    logger.debug(f"Quality score: {score}")

    return score