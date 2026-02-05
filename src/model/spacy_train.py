'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "spaCy NER training utilities for Clinical NER."
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
logger = get_logger(name="clinical_ner.spacy_train")


def _build_spacy_training_data(
    records: Iterable[Record],
    labels: Iterable[EntityLabel],
) -> list[tuple[str, dict]]:
    """
        Convert Record objects into spaCy training format

        Args:
            records: Iterable of Record objects
            labels: Entity labels to include in training

        Returns:
            List of (text, annotations) tuples
    """

    ## Normalize label set
    allowed_labels = {lbl.value for lbl in labels}

    training_data: list[tuple[str, dict]] = []

    for rec in records:
        entities = []

        for ent in rec.entities:
            if ent.label.value not in allowed_labels:
                continue

            entities.append((ent.start, ent.end, ent.label.value))

        if entities:
            training_data.append((rec.text, {"entities": entities}))

    return training_data


def train_spacy_ner(
    records: Iterable[Record],
    output_dir: str | Path,
    labels: Iterable[EntityLabel],
    n_iter: int = 30,
) -> Path:
    """
        Train a spaCy NER model from labeled records

        Args:
            records: Labeled Record objects
            output_dir: Directory where the model will be saved
            labels: Entity labels to train
            n_iter: Number of training iterations

        Returns:
            Path to the trained spaCy model directory

        Raises:
            ConfigurationError: If spaCy is not installed
            DataError: If no training data is available
    """
    try:
        import spacy  # type: ignore
        from spacy.util import minibatch  # type: ignore
    except Exception as exc:
        msg = "spaCy library is required for training"
        logger.error(msg)
        raise ConfigurationError(msg) from exc

    ## Build training data
    training_data = _build_spacy_training_data(records, labels)

    if not training_data:
        msg = "No valid training data found for spaCy NER training"
        logger.error(msg)
        raise DataError(msg)

    ## Initialize blank model
    logger.info("Initializing blank spaCy model")
    nlp = spacy.blank("en")

    ## Create NER component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    ## Add labels to NER
    for lbl in labels:
        ner.add_label(lbl.value)

    ## Disable other pipes during training
    other_pipes = [p for p in nlp.pipe_names if p != "ner"]

    ## Train the model
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()

        for itn in range(n_iter):
            losses = {}

            ## Train in minibatches
            for batch in minibatch(training_data, size=8):
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, losses=losses)

            logger.info("spaCy training iteration %d | loss=%.4f", itn + 1, losses.get("ner", 0.0))

    ## Save trained model
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    nlp.to_disk(output_dir)
    logger.info("spaCy model saved to: %s", output_dir)

    return output_dir
