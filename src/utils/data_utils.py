'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Utility functions for clustering: normalization, embeddings validation, types and quality."
'''

from __future__ import annotations

import re
import math
from typing import Any, Dict, List

from src.utils.utils import normalize_clinical_text
from src.utils.logging_utils import get_logger

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

    normalized = {}
    use_fe = data.get("feature_engineering", False)

    for key, value in data.items():

        ## Normalize string
        if isinstance(value, str):
            logger.debug(f"Normalizing string: {key}")
            if use_fe:
                value = normalize_clinical_text(value)
            else:
                value = value.strip().lower()
                value = re.sub(r"\s+", " ", value)
                
        ## Normalize list
        if isinstance(value, list):
            logger.debug(f"Normalizing list: {key}")
            value = [
                v.strip().lower() if isinstance(v, str) else v
                for v in value
            ]

        normalized[key] = value
        if use_fe and key == "text":
            normalized["normalized_text"] = value
            normalized["char_length"] = len(value)
            normalized["token_count"] = len(value.split())
            
    return normalized

def validate_schema(data: Dict[str, Any]) -> List[Dict]:
    """
        Validate required fields for clustering

        Args:
            data: Input dictionary

        Returns:
            List of issues
    """

    issues = []

    ## Required fields
    if "text" not in data:
        logger.error("Missing text field")
        issues.append({
            "rule": "schema_text",
            "level": "error",
            "message": "text is required",
        })

    if "embeddings" not in data:
        logger.error("Missing embeddings field")
        issues.append({
            "rule": "schema_embeddings",
            "level": "error",
            "message": "embeddings are required",
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

def validate_embeddings_basic(embeddings: List[Any]) -> List[Dict]:
    """
        Basic embeddings validation

        Args:
            embeddings: Embedding list

        Returns:
            List of issues
    """

    issues = []

    if not embeddings:
        logger.error("Embeddings empty")
        issues.append({
            "rule": "embedding_empty",
            "level": "error",
            "message": "Embeddings list is empty",
        })
        return issues

    first = embeddings[0]

    if not isinstance(first, list):
        logger.error("Embedding vector type invalid")
        issues.append({
            "rule": "embedding_vector_type",
            "level": "error",
            "message": "Embedding must be list of list",
        })
        return issues

    dim = len(first)

    for i, vec in enumerate(embeddings):

        ## Dimension check
        if isinstance(vec, list) and len(vec) != dim:
            logger.error(f"Inconsistent dimension at {i}")
            issues.append({
                "rule": "embedding_dim",
                "level": "error",
                "message": "Inconsistent embedding dimension",
            })

        ## NaN check
        for v in vec:
            if isinstance(v, float) and math.isnan(v):
                logger.error("NaN detected")
                issues.append({
                    "rule": "embedding_nan",
                    "level": "error",
                    "message": "NaN in embeddings",
                })
                break

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

    if not text:
        logger.warning("Empty text for scoring")
        return 0.0

    valid_chars = sum(c.isalnum() for c in text)
    score = valid_chars / len(text)

    logger.debug(f"Quality score: {score}")

    return score