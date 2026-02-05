'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Model inference layer for Clinical NER: Hugging Face and spaCy wrappers producing Entity objects."
'''

from __future__ import annotations

## Standard library imports
from pathlib import Path
from typing import Iterable

## Core domain imports
from src.core.entities import (
    EntityLabel,
    EntityProvenance,
)
from src.core.schema import Entity, Record

## Centralized errors and logging
from src.core.errors import ConfigurationError
from src.utils.logging_utils import get_logger

## Generic utilities
from src.utils.utils import ensure_str


## Module-level logger
logger = get_logger(name="clinical_ner.inference")


def load_hf_pipeline(model_name: str, device: str = "cpu"):
    """
        Load a Hugging Face NER pipeline

        Args:
            model_name: Hugging Face model identifier
            device: Execution device (cpu or cuda)

        Returns:
            Hugging Face pipeline instance

        Raises:
            ConfigurationError: If transformers is not installed
    """
    try:
        from transformers import pipeline  # type: ignore
    except Exception as exc:
        msg = "transformers library is required for Hugging Face inference"
        logger.error(msg)
        raise ConfigurationError(msg) from exc

    logger.info("Loading Hugging Face NER model: %s", model_name)
    return pipeline(
        task="token-classification",
        model=model_name,
        aggregation_strategy="simple",
        device=0 if device == "cuda" else -1,
    )


def infer_entities_hf(
    records: Iterable[Record],
    model_name: str,
    device: str = "cpu",
    label_mapping: dict[str, EntityLabel] | None = None,
    confidence_threshold: float = 0.5,
) -> list[Record]:
    """
        Run Hugging Face NER inference on records

        Args:
            records: Iterable of Record objects
            model_name: Hugging Face model identifier
            device: Execution device
            label_mapping: Optional mapping from model labels to EntityLabel
            confidence_threshold: Minimum score to keep entity

        Returns:
            Updated list of Record objects with inferred entities
    """
    ## Load HF pipeline
    nlp = load_hf_pipeline(model_name=model_name, device=device)

    updated_records: list[Record] = []

    for record in records:
        ## Run model on full text
        outputs = nlp(record.text)

        for out in outputs:
            score = float(out.get("score", 0.0))
            if score < confidence_threshold:
                continue

            ## Map model label
            raw_label = ensure_str(out.get("entity_group"))
            label = (
                label_mapping.get(raw_label)
                if label_mapping and raw_label in label_mapping
                else None
            )

            if label is None:
                continue

            ## Build entity
            entity = Entity(
                id=f"ent_{len(record.entities):06d}",
                text=out.get("word"),
                start=int(out.get("start")),
                end=int(out.get("end")),
                label=label,
                confidence=score,
                source=EntityProvenance.MODEL,
            )

            record.entities.append(entity)

        updated_records.append(record)

    return updated_records


def infer_entities_spacy(
    records: Iterable[Record],
    model_path: str | Path,
    label_mapping: dict[str, EntityLabel] | None = None,
) -> list[Record]:
    """
        Run spaCy NER inference on records

        Args:
            records: Iterable of Record objects
            model_path: Path to spaCy model directory
            label_mapping: Optional mapping from spaCy labels to EntityLabel

        Returns:
            Updated list of Record objects with inferred entities

        Raises:
            ConfigurationError: If spaCy is not installed or model is missing
    """
    try:
        import spacy  # type: ignore
    except Exception as exc:
        msg = "spaCy library is required for spaCy inference"
        logger.error(msg)
        raise ConfigurationError(msg) from exc

    model_path = Path(model_path).resolve()

    if not model_path.exists():
        msg = f"spaCy model not found: {model_path}"
        logger.error(msg)
        raise ConfigurationError(msg)

    logger.info("Loading spaCy model from: %s", model_path)
    nlp = spacy.load(model_path)

    updated_records: list[Record] = []

    for record in records:
        doc = nlp(record.text)

        for ent in doc.ents:
            ## Map spaCy label
            label = (
                label_mapping.get(ent.label_)
                if label_mapping and ent.label_ in label_mapping
                else None
            )

            if label is None:
                continue

            ## Build entity
            entity = Entity(
                id=f"ent_{len(record.entities):06d}",
                text=ent.text,
                start=ent.start_char,
                end=ent.end_char,
                label=label,
                source=EntityProvenance.MODEL,
            )

            record.entities.append(entity)

        updated_records.append(record)

    return updated_records
