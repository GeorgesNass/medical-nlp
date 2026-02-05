'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Rule-based NLP components: dictionary matching, negation detection, and temporality inference."
'''

from __future__ import annotations

## Standard library imports
import re
from pathlib import Path

## Core domain imports
from src.core.entities import (
    DictionarySource,
    EntityLabel,
    EntityProvenance,
    NegationStatus,
    is_temporality_applicable,
)
from src.core.schema import Entity, Record

## Centralized errors and logging
from src.core.errors import DataError
from src.utils.logging_utils import get_logger

## Generic utilities
from src.utils.utils import ensure_str


## Module-level logger
logger = get_logger(name="clinical_ner.rules")


def load_dictionary_terms(dictionaries_root: str | Path) -> dict[str, dict[str, str]]:
    """
        Load dictionary terms used for auto-labeling

        Expected format (in-memory):
            {
                "aspirin": {"concept_id": "...", "concept_name": "...", "dictionary": "mesh"},
                ...
            }

        Args:
            dictionaries_root: Root directory containing dictionaries

        Returns:
            Mapping of surface form to dictionary metadata

        Raises:
            DataError: If dictionary folder is missing
    """
    ## Resolve dictionaries root
    root = Path(dictionaries_root).resolve()

    if not root.exists():
        msg = f"Dictionaries root not found: {root}"
        logger.error(msg)
        raise DataError(msg)

    ## Placeholder for real loaders (MeSH, custom, UMLS)
    ## This must be implemented later based on your dictionary file formats
    return {}


def dictionary_match(
    text: str,
    dictionary: dict[str, dict[str, str]],
    label: EntityLabel,
    max_entities: int = 300,
    provenance: EntityProvenance = EntityProvenance.DICT_AUTO,
) -> list[Entity]:
    """
        Perform dictionary-based entity matching on raw text

        Args:
            text: Input document text
            dictionary: Loaded dictionary mapping term -> metadata
            label: Entity label assigned to matches
            max_entities: Safety cap on number of entities
            provenance: Provenance assigned to created entities

        Returns:
            List of detected Entity objects
    """
    ## Initialize output list
    entities: list[Entity] = []

    ## Short-circuit on empty input
    raw_text = ensure_str(text)
    if raw_text.strip() == "" or not dictionary:
        return entities

    ## Regex-based matching (baseline)
    for term, meta in dictionary.items():
        if len(entities) >= max_entities:
            break

        ## Skip invalid terms
        term_clean = ensure_str(term).strip()
        if term_clean == "":
            continue

        ## Match term as a token-like unit
        pattern = rf"\b{re.escape(term_clean)}\b"
        for match in re.finditer(pattern, raw_text, flags=re.IGNORECASE):
            if len(entities) >= max_entities:
                break

            ## Read dictionary metadata
            concept_id = meta.get("concept_id")
            concept_name = meta.get("concept_name")
            dictionary_src = meta.get("dictionary", DictionarySource.MESH.value)

            ## Build entity object
            ent = Entity(
                id=f"ent_{len(entities):06d}",
                text=match.group(0),
                start=match.start(),
                end=match.end(),
                label=label,
                concept_id=concept_id,
                concept_name=concept_name,
                dictionary=DictionarySource(dictionary_src)
                if dictionary_src in {d.value for d in DictionarySource}
                else DictionarySource.MESH,
                source=provenance,
                confidence=1.0,
            )

            ## Validate entity object (will enforce concept_id/name for auto provenance)
            ent.validate()

            entities.append(ent)

    return entities


def detect_negation_for_entity(record: Record, ent: Entity, window: int = 12) -> NegationStatus:
    """
        Detect negation status for a single entity using a simple context window

        Args:
            record: Parent record
            ent: Entity to classify
            window: Number of tokens before the entity to inspect

        Returns:
            NegationStatus
    """
    ## Negation cues baseline
    negation_cues = {"no", "not", "without", "denies", "deny", "absence", "absent"}

    tokens = ensure_str(record.text).lower().split()

    ## Compute token index approximation
    token_index = len(record.text[: ent.start].split())
    start = max(0, token_index - window)
    context = set(tokens[start:token_index])

    return NegationStatus.NEGATED if context & negation_cues else NegationStatus.NOT_NEGATED


def apply_negation_rules(record: Record, window: int = 12) -> None:
    """
        Apply rule-based negation detection to entities in a record

        Args:
            record: Record containing entities
            window: Number of tokens before entity to inspect
    """
    for ent in record.entities:
        ## Assign negation based on local context
        ent.negation = detect_negation_for_entity(record=record, ent=ent, window=window)


def infer_temporality_for_entity(record: Record, ent: Entity) -> str | None:
    """
        Infer temporality for a single entity using simple keyword cues

        Args:
            record: Parent record
            ent: Entity to enrich

        Returns:
            Temporality string or None
    """
    ## Keywords baseline
    past_cues = {"history", "previously", "formerly", "past"}
    chronic_cues = {"chronic", "long-term", "longterm", "longstanding"}
    current_cues = {"currently", "ongoing", "today", "active", "now"}
    future_cues = {"will", "planned", "schedule", "tomorrow", "next"}

    tokens = ensure_str(record.text).lower().split()

    ## Default temporality is None and validated later in Entity.validate()
    if any(cue in tokens for cue in past_cues):
        return "past"
    if any(cue in tokens for cue in chronic_cues):
        return "chronic"
    if any(cue in tokens for cue in current_cues):
        return "current"

    ## Only MEDICATION should accept "future" (validated in schema/entities policy)
    if any(cue in tokens for cue in future_cues):
        return "future"

    return None


def apply_temporality_rules(record: Record) -> None:
    """
        Apply rule-based temporality inference to entities in a record

        Args:
            record: Record containing entities

        Raises:
            DataError: If inferred temporality violates business rules
    """
    for ent in record.entities:
        ## Skip labels without temporality
        if not is_temporality_applicable(ent.label):
            ent.temporality = None
            continue

        ## Infer temporality and validate through Entity.validate()
        ent.temporality = infer_temporality_for_entity(record=record, ent=ent)

        try:
            ent.validate()
        except DataError as exc:
            msg = (
                f"Temporality validation failed for record_id {record.record_id} "
                f"(entity id {ent.id})"
            )
            logger.error(msg)
            raise DataError(msg) from exc
