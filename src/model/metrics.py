'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Evaluation metrics for Clinical NER: precision, recall, F1 at entity level."
'''

from __future__ import annotations

## Standard library imports
from typing import Iterable

## Centralized errors and logging
from src.core.errors import DataError
from src.utils.logging_utils import get_logger

## Core domain imports
from src.core.schema import Entity, Record
from src.core.entities import EntityLabel


## Module-level logger
logger = get_logger(name="clinical_ner.metrics")


def _entity_key(ent: Entity) -> tuple:
    """
        Build a comparable key for entity-level evaluation

        Args:
            ent: Entity instance

        Returns:
            Tuple uniquely identifying an entity span and label
    """
    return (ent.start, ent.end, ent.label.value)


def compute_entity_metrics(
    gold_records: Iterable[Record],
    pred_records: Iterable[Record],
    labels: Iterable[EntityLabel] | None = None,
) -> dict[str, float]:
    """
        Compute precision, recall and F1-score at entity level

        Args:
            gold_records: Records containing gold entities
            pred_records: Records containing predicted entities
            labels: Optional subset of EntityLabel to evaluate

        Returns:
            Dictionary with precision, recall and f1 scores

        Raises:
            DataError: If records are inconsistent
    """

    ## Convert label filter to set
    label_filter = {lbl.value for lbl in labels} if labels else None

    ## Build flat lists of entity keys
    gold_entities: set[tuple] = set()
    pred_entities: set[tuple] = set()

    for rec in gold_records:
        for ent in rec.entities:
            if label_filter and ent.label.value not in label_filter:
                continue
            gold_entities.add(_entity_key(ent))

    for rec in pred_records:
        for ent in rec.entities:
            if label_filter and ent.label.value not in label_filter:
                continue
            pred_entities.add(_entity_key(ent))

    ## Validate non-empty gold set
    if not gold_entities:
        msg = "No gold entities provided for metric computation"
        logger.error(msg)
        raise DataError(msg)

    ## Compute counts
    true_positives = len(gold_entities & pred_entities)
    false_positives = len(pred_entities - gold_entities)
    false_negatives = len(gold_entities - pred_entities)

    ## Compute metrics safely
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    ## Log summary
    logger.info(
        "Entity metrics computed | TP=%d FP=%d FN=%d",
        true_positives,
        false_positives,
        false_negatives,
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
