'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Utility functions for clinical NER: normalization, entities validation, types and quality"
'''

from __future__ import annotations

from typing import Any, Dict, List

from src.core.config import get_config
from src.nlp.normalization import normalize_clinical_text
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

            config = get_config()

            if config.feature_engineering.enabled:
                value = normalize_clinical_text(value)
            else:
                value = value.strip().lower()
                
        ## Normalize entities
        if key == "entities" and isinstance(value, list):
            logger.debug("Normalizing entities")

            normalized_entities = []

            for entity in value:
                if isinstance(entity, dict):
                    normalized_entities.append({
                        "start": int(entity.get("start", 0)),
                        "end": int(entity.get("end", 0)),
                        "label": entity.get("label", "").upper(),
                    })

            value = normalized_entities

        normalized[key] = value

    return normalized

def validate_schema(data: Dict[str, Any]) -> List[Dict]:
    """
        Validate required fields for NER

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

    ## Required entities
    if "entities" not in data:
        logger.error("Missing entities field")
        issues.append({
            "rule": "schema_entities",
            "level": "error",
            "message": "entities are required",
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

    ## Entities type
    if "entities" in data and not isinstance(data["entities"], list):
        logger.error("Invalid entities type")
        issues.append({
            "rule": "type_entities",
            "level": "error",
            "message": "entities must be list",
        })

    return issues

def validate_entities_basic(entities: List[Any]) -> List[Dict]:
    """
        Validate entities basic structure

        Args:
            entities: List of entities

        Returns:
            List[Dict]
    """

    issues = []

    for idx, entity in enumerate(entities):

        if not isinstance(entity, dict):
            logger.error(f"Invalid entity at index {idx}")
            issues.append({
                "rule": "entity_format",
                "level": "error",
                "message": "Entity must be dict",
            })
            continue

        ## Required keys
        for key in ["start", "end", "label"]:
            if key not in entity:
                logger.error(f"Missing {key} in entity {idx}")
                issues.append({
                    "rule": "entity_missing_key",
                    "level": "error",
                    "message": f"Missing {key}",
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
    
## ============================================================
## FEATURE ENGINEERING EXPORT
## ============================================================
def load_and_normalize_text_from_path(data: Dict[str, Any]) -> str:
    """
        Load and normalize text from input dictionary

        High-level workflow:
            1) Extract text field
            2) Apply feature engineering normalization if enabled

        Args:
            data: Input dictionary

        Returns:
            Normalized text
    """

    config = get_config()

    text = data.get("text", "")

    if config.feature_engineering.enabled:
        return normalize_clinical_text(text)

    return str(text).strip()

def export_feature_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """
        Export feature summary for a record

        High-level workflow:
            1) Extract text
            2) Compute basic statistics

        Args:
            data: Input dictionary

        Returns:
            Dict with feature stats
    """

    text = data.get("text", "")

    return {
        "char_length": len(text),
        "token_count": len(str(text).split()),
        "quality_score": compute_quality_score(data),
    }