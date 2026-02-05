'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main pipeline orchestration for Clinical NER: loading, labeling, inference and export."
'''

from __future__ import annotations

## Standard library imports
import csv
from pathlib import Path
from typing import Iterable

## Core config, schema and errors
from src.core.config import ProjectConfig
from src.core.errors import ConfigurationError, DataError
from src.core.schema import Record

## Logging
from src.utils.logging_utils import get_logger

## Generic utilities
from src.utils.utils import (
    ensure_str,
    json_dumps,
    parse_date_to_iso,
)

## NLP rules (auto-label + enrichment)
from src.nlp.rules import (
    apply_negation_rules,
    apply_temporality_rules,
    dictionary_match,
    load_dictionary_terms,
)


## Module-level logger
logger = get_logger(name="clinical_ner.pipeline")


def load_labeled_csv(csv_path: str | Path) -> list[Record]:
    """
        Load labeled records from a CSV file

        Args:
            csv_path: Path to labeled CSV file

        Returns:
            List of Record objects

        Raises:
            DataError: If file is missing or rows are invalid
    """

    ## Resolve CSV path
    path = Path(csv_path).resolve()

    if not path.exists():
        msg = f"Labeled CSV file not found: {path}"
        logger.error(msg)
        raise DataError(msg)

    records: list[Record] = []

    ## Read CSV rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            ## Build record from raw dict (handles entities JSON/list)
            record = Record.from_dict(
                {
                    "record_id": row.get("record_id"),
                    "patient_id": row.get("patient_id"),
                    "name_document": row.get("name_document"),
                    "type_document": row.get("type_document"),
                    "text": row.get("text"),
                    "date_document": row.get("date_document"),
                    "entities": row.get("entities", "[]"),
                }
            )

            records.append(record)

    return records


def load_text_documents(folder: str | Path, max_docs: int | None = None) -> list[Record]:
    """
        Load raw .txt documents and convert them to Record objects

        Args:
            folder: Folder containing text documents
            max_docs: Optional maximum number of documents

        Returns:
            List of Record objects

        Raises:
            DataError: If folder does not exist
    """

    ## Resolve input folder
    root = Path(folder).resolve()

    if not root.exists():
        msg = f"Raw text folder not found: {root}"
        logger.error(msg)
        raise DataError(msg)

    ## Collect text files
    files = sorted(p for p in root.rglob("*.txt") if p.is_file())

    if max_docs is not None:
        files = files[:max_docs]

    records: list[Record] = []

    for p in files:
        ## Read file content
        text = p.read_text(encoding="utf-8", errors="ignore")

        ## Build minimal record (schema requires strict fields)
        record = Record(
            record_id=ensure_str(p.stem),
            patient_id="unknown",
            name_document=ensure_str(p.name),
            type_document="unknown",
            text=ensure_str(text),
            date_document=None,
            entities=[],
        )

        ## Validate record (no spans yet)
        record.validate(validate_spans=False)

        records.append(record)

    return records


def _auto_label_and_enrich_records(cfg: ProjectConfig, records: list[Record]) -> list[Record]:
    """
        Auto-label and enrich records in-place using dictionaries + rules

        Args:
            cfg: Project configuration
            records: Input records (unlabeled)

        Returns:
            Enriched records

        Raises:
            ConfigurationError: If dictionary config is missing
            DataError: If labeling or enrichment fails
    """

    if cfg.dictionaries is None:
        msg = "Dictionary configuration is required for auto-labeling"
        logger.error(msg)
        raise ConfigurationError(msg)

    ## Load dictionary terms once
    dictionary = load_dictionary_terms(cfg.dictionaries.dictionaries_root)

    ## Apply dictionary matching per label
    for rec in records:
        ## Keep a running entity counter to avoid id collisions across labels
        entities = []

        for label in cfg.runtime.ner_labels:
            ## Match entities for this label
            matches = dictionary_match(
                text=rec.text,
                dictionary=dictionary,
                label=label,
                max_entities=cfg.runtime.max_entities_per_record,
            )
            entities.extend(matches)

            if len(entities) >= cfg.runtime.max_entities_per_record:
                break

        ## Assign entities and validate spans
        rec.entities = entities
        rec.validate(validate_spans=True, max_entities=cfg.runtime.max_entities_per_record)

        ## Enrich with negation and temporality
        if cfg.runtime.enable_negation:
            apply_negation_rules(record=rec, window=12)

        if cfg.runtime.enable_temporality:
            apply_temporality_rules(record=rec)

        ## Final validation after enrichment
        rec.validate(validate_spans=True, max_entities=cfg.runtime.max_entities_per_record)

    return records


def save_records_to_csv(records: Iterable[Record], output_path: str | Path) -> None:
    """
        Save records to CSV with entities serialized as JSON

        Args:
            records: Iterable of Record objects
            output_path: Output CSV path
    """

    ## Resolve output path
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    ## Define stable CSV schema
    fieldnames = [
        "record_id",
        "patient_id",
        "name_document",
        "type_document",
        "date_document",
        "text",
        "entities",
    ]

    ## Write CSV file
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            ## Use canonical schema export
            row = r.to_csv_row()

            ## Ensure date is ISO if present
            row["date_document"] = (
                parse_date_to_iso(row.get("date_document"))
                if row.get("date_document")
                else None
            )

            writer.writerow(row)


def run_pipeline(
    cfg: ProjectConfig,
    labeled_csv_path: str | Path | None = None,
    unlabeled_texts_dir: str | Path | None = None,
    output_csv_path: str | Path | None = None,
) -> Path:
    """
        Execute the Clinical NER pipeline

        Args:
            cfg: ProjectConfig instance
            labeled_csv_path: Optional labeled CSV input
            unlabeled_texts_dir: Optional raw text input directory
            output_csv_path: Optional output CSV path

        Returns:
            Path to the generated CSV file
    """

    ## Ensure project directories exist
    cfg.ensure_dirs()

    ## Detect input mode
    has_labeled = labeled_csv_path is not None
    has_unlabeled = unlabeled_texts_dir is not None

    if has_labeled and has_unlabeled:
        msg = "Provide either labeled_csv_path or unlabeled_texts_dir, not both"
        logger.error(msg)
        raise ConfigurationError(msg)

    if not has_labeled and not has_unlabeled:
        msg = "No input provided to pipeline"
        logger.error(msg)
        raise ConfigurationError(msg)

    ## Resolve output path
    if output_csv_path is None:
        output_csv_path = cfg.paths.artifacts_exports / "clinical_ner_records.csv"

    ## Labeled mode
    if has_labeled:
        if not cfg.runtime.accept_labeled_data:
            msg = "Labeled input is disabled by configuration"
            logger.error(msg)
            raise ConfigurationError(msg)

        logger.info("Running pipeline in labeled mode")
        records = load_labeled_csv(labeled_csv_path)  # type: ignore[arg-type]

        ## Optional enrichment on labeled data
        for rec in records:
            if cfg.runtime.enable_negation:
                apply_negation_rules(record=rec, window=12)
            if cfg.runtime.enable_temporality:
                apply_temporality_rules(record=rec)

            rec.validate(validate_spans=True, max_entities=cfg.runtime.max_entities_per_record)

    ## Unlabeled mode
    else:
        if not cfg.runtime.accept_unlabeled_texts:
            msg = "Unlabeled input is disabled by configuration"
            logger.error(msg)
            raise ConfigurationError(msg)

        logger.info("Running pipeline in unlabeled mode")

        records = load_text_documents(
            folder=unlabeled_texts_dir,  # type: ignore[arg-type]
            max_docs=cfg.runtime.max_docs,
        )

        ## Auto-label and enrich
        records = _auto_label_and_enrich_records(cfg=cfg, records=records)

    ## Save results
    save_records_to_csv(records, output_csv_path)

    logger.info("Pipeline completed successfully")
    return Path(output_csv_path).resolve()
