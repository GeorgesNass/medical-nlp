'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Hugging Face NER training utilities for Clinical NER."
'''

from __future__ import annotations

## Standard library imports
from pathlib import Path
from typing import Iterable

## Centralized errors and logging
from src.core.errors import ConfigurationError, DataError
from src.utils.logging_utils import get_logger

## Core domain imports
from src.core.schema import Record
from src.core.entities import EntityLabel


## Module-level logger
logger = get_logger(name="clinical_ner.hf_train")


def _ensure_transformers_available() -> None:
    """
        Ensure transformers and torch are available

        Raises:
            ConfigurationError: If required libraries are missing
    """
    try:
        import transformers  # noqa: F401 # type: ignore
        import torch  # noqa: F401 # type: ignore
    except Exception as exc:
        msg = "transformers and torch are required for Hugging Face training"
        logger.error(msg)
        raise ConfigurationError(msg) from exc


def _build_hf_dataset(records: Iterable[Record]) -> list[dict]:
    """
        Convert records into a minimal list-of-dicts dataset

        Args:
            records: Iterable of Record objects

        Returns:
            List of dict records with text and entity spans

        Raises:
            DataError: If no entities are found
    """
    dataset: list[dict] = []

    for rec in records:
        if not rec.entities:
            continue

        dataset.append(
            {
                "text": rec.text,
                "entities": [
                    {"start": e.start, "end": e.end, "label": e.label.value}
                    for e in rec.entities
                ],
            }
        )

    if not dataset:
        msg = "No entities found in records for Hugging Face dataset building"
        logger.error(msg)
        raise DataError(msg)

    return dataset


def train_hf_ner(
    records: Iterable[Record],
    output_dir: str | Path,
    base_model_name: str,
    labels: Iterable[EntityLabel],
) -> Path:
    """
        Train a Hugging Face NER model from labeled records

        Notes:
            - This file provides a minimal training scaffold
            - Tokenization/label alignment is typically dataset-specific
            - You can later replace this with a full Trainer implementation

        Args:
            records: Labeled Record objects
            output_dir: Output directory for the trained model
            base_model_name: Base model identifier
            labels: Entity labels list

        Returns:
            Path to output directory

        Raises:
            ConfigurationError: If transformers/torch are missing
            DataError: If no training data is available
    """
    _ensure_transformers_available()

    ## Build minimal dataset structure
    _ = _build_hf_dataset(records)

    ## Create output directory
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    ## Log scaffold warning
    logger.warning("HF training scaffold created, full Trainer pipeline not implemented yet")
    logger.info("Base model: %s", base_model_name)
    logger.info("Labels: %s", [lbl.value for lbl in labels])
    logger.info("Output dir: %s", out)

    return out
