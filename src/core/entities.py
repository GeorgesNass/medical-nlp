'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Core enums and constants for Clinical NER: entity labels, sources, negation and temporality."
'''

from __future__ import annotations

## Standard library imports
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

## Centralized errors and logging
from src.core.config import get_config
from src.core.errors import DataError
from src.nlp.normalization import normalize_clinical_text
from src.utils.logging_utils import get_logger

## Module-level logger
logger = get_logger(name="clinical_ner.entities")

class EntityLabel(str, Enum):
    """
        Canonical entity labels for the Clinical NER project

        Notes:
            - Keep this list stable because it impacts training labels
            - Labels are shared across rule-based and ML pipelines
    """

    MEDICATION = "MEDICATION"
    DISEASE = "DISEASE"
    ALLERGY = "ALLERGY"
    PROCEDURE = "PROCEDURE"
    TEST = "TEST"
    ANATOMY = "ANATOMY"
    SYMPTOM = "SYMPTOM"
    FAMILY_HISTORY = "FAMILY_HISTORY"
    SOCIAL_HISTORY = "SOCIAL_HISTORY"
    DEVICE = "DEVICE"
    MICROORGANISM = "MICROORGANISM"
    CHEMICAL = "CHEMICAL"
    DOSAGE = "DOSAGE"
    FREQUENCY = "FREQUENCY"
    DURATION = "DURATION"
    ROUTE = "ROUTE"

class DictionarySource(str, Enum):
    """
        Dictionary or ontology source identifiers

        Notes:
            - MESH expects a MeSH concept_id
            - CUSTOM follows project-specific conventions
            - UMLS follows project-specific conventions
    """

    MESH = "mesh"
    CUSTOM = "custom"
    UMLS = "umls"

class EntityProvenance(str, Enum):
    """
        Provenance describing how an entity annotation was obtained
    """

    MANUAL = "manual"
    MESH_AUTO = "mesh_auto"
    DICT_AUTO = "dict_auto"
    MODEL = "model"

class TemporalityMedication(str, Enum):
    """
        Temporality values for MEDICATION entities
    """

    PAST = "past"
    CURRENT = "current"
    CHRONIC = "chronic"
    FUTURE = "future"

class TemporalityPathology(str, Enum):
    """
        Temporality values for DISEASE / PATHOLOGY entities

        Notes:
            - FUTURE is intentionally excluded
    """

    PAST = "past"
    CURRENT = "current"
    CHRONIC = "chronic"

class NegationStatus(str, Enum):
    """
        Negation status for an entity mention
    """

    NEGATED = "negated"
    NOT_NEGATED = "not_negated"
    UNKNOWN = "unknown"

## Default NER labels
DEFAULT_NER_LABELS: tuple[EntityLabel, ...] = (
    EntityLabel.MEDICATION,
    EntityLabel.DISEASE,
    EntityLabel.ALLERGY,
    EntityLabel.PROCEDURE,
    EntityLabel.TEST,
    EntityLabel.ANATOMY,
)

## Labels supporting temporality
TEMPORALITY_LABELS_MEDICATION: tuple[EntityLabel, ...] = (EntityLabel.MEDICATION,)
TEMPORALITY_LABELS_PATHOLOGY: tuple[EntityLabel, ...] = (EntityLabel.DISEASE,)

@dataclass(frozen=True, slots=True)
class TemporalityPolicy:
    """
        Allowed temporality values per entity category

        Attributes:
            medication_values: Allowed temporality values for MEDICATION
            pathology_values: Allowed temporality values for DISEASE
    """

    medication_values: tuple[TemporalityMedication, ...]
    pathology_values: tuple[TemporalityPathology, ...]

## Global temporality policy
TEMPORALITY_POLICY = TemporalityPolicy(
    medication_values=(
        TemporalityMedication.PAST,
        TemporalityMedication.CURRENT,
        TemporalityMedication.CHRONIC,
        TemporalityMedication.FUTURE,
    ),
    pathology_values=(
        TemporalityPathology.PAST,
        TemporalityPathology.CURRENT,
        TemporalityPathology.CHRONIC,
    ),
)

## ============================================================
## FEATURE ENGINEERING ENTITY STRUCTURE
## ============================================================
@dataclass(frozen=True, slots=True)
class EntityFeatures:
    """
        Feature representation for an extracted entity

        Args:
            text: Raw entity text
            normalized_text: Normalized entity text
            token_count: Number of tokens
            char_length: Character length
    """

    text: str
    normalized_text: str
    token_count: int
    char_length: int

def build_entity_features(text: str) -> EntityFeatures:
    """
        Build feature representation for an entity

        Args:
            text: Raw entity text

        Returns:
            EntityFeatures
    """

    config = get_config()

    if config.feature_engineering.enabled:
        normalized = normalize_clinical_text(text)
    else:
        normalized = text.strip()

    tokens = normalized.split()

    return EntityFeatures(
        text=text,
        normalized_text=normalized,
        token_count=len(tokens),
        char_length=len(normalized),
    )
    
def is_temporality_applicable(label: EntityLabel) -> bool:
    """
        Check whether temporality applies to an entity label

        Args:
            label: Entity label

        Returns:
            True if temporality applies, otherwise False
    """
    
    return (
        label in TEMPORALITY_LABELS_MEDICATION
        or label in TEMPORALITY_LABELS_PATHOLOGY
    )

def validate_temporality(label: EntityLabel, temporality: str | None) -> bool:
    """
        Validate temporality value according to entity label constraints

        Rules:
            - If temporality is None:
                - Valid if label does NOT support temporality
            - If label supports temporality:
                - temporality must be in the allowed set
            - If label does NOT support temporality:
                - temporality must be None

        Args:
            label: Entity label
            temporality: Temporal value or None

        Returns:
            True if temporality is valid, otherwise False
    """

    ## Temporality not provided
    if temporality is None:
        return not is_temporality_applicable(label)

    ## Normalize temporality
    value = temporality.strip().lower()

    ## Medication temporality
    if label in TEMPORALITY_LABELS_MEDICATION:
        allowed = {v.value for v in TEMPORALITY_POLICY.medication_values}
        return value in allowed

    ## Pathology temporality
    if label in TEMPORALITY_LABELS_PATHOLOGY:
        allowed = {v.value for v in TEMPORALITY_POLICY.pathology_values}
        return value in allowed

    ## No temporality expected for other labels
    return False

def normalize_label(raw_label: str) -> EntityLabel:
    """
        Normalize a raw string into an EntityLabel enum

        Args:
            raw_label: Raw label string (case-insensitive)

        Returns:
            Normalized EntityLabel

        Raises:
            DataError: If the label is unknown
    """

    ## Normalize input
    raw = raw_label.strip().upper()

    for lbl in EntityLabel:
        if lbl.value == raw:
            return lbl

    ## Feature engineering hook (no mutation, just trace)
    if get_config().feature_engineering.enabled:
        logger.debug(f"Label normalized with FE: {raw}")
        
    msg = f"Unknown entity label: {raw_label}"
    logger.error(msg)
    raise DataError(msg)

def normalize_dictionary_source(raw_source: str) -> DictionarySource:
    """
        Normalize a raw string into a DictionarySource enum

        Args:
            raw_source: Raw dictionary source string

        Returns:
            Normalized DictionarySource

        Raises:
            DataError: If the source is unknown
    """

    ## Normalize input
    raw = raw_source.strip().lower()

    for src in DictionarySource:
        if src.value == raw:
            return src

    msg = f"Unknown dictionary source: {raw_source}"
    logger.error(msg)
    raise DataError(msg)

def ensure_unique_entity_ids(entity_ids: Iterable[str]) -> bool:
    """
        Check uniqueness of entity identifiers

        Args:
            entity_ids: Iterable of entity ids

        Returns:
            True if all ids are unique, otherwise False
    """

    ## Convert to list to preserve iteration
    values = list(entity_ids)
    return len(values) == len(set(values))
