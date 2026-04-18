'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Combined unit tests for Clinical NER: schema, config, NLP rules and pipeline smoke tests."
'''

from __future__ import annotations

## Standard library imports
from pathlib import Path

## Third-party imports
import pytest

## Core imports
from src.core.data_consistency import run_data_consistency
from src.core.data_quality import run_data_quality
from src.core.config import ProjectConfig
from src.core.entities import EntityLabel, EntityProvenance, NegationStatus
from src.core.errors import ConfigurationError, DataError
from src.core.schema import Entity, Record

## NLP imports
from src.nlp.normalization import normalize_text
from src.nlp.rules import apply_negation_rules, apply_temporality_rules

## Pipeline imports
from src.pipeline import run_pipeline

def test_normalize_text_basic() -> None:
    """
        Test basic text normalization pipeline

        Ensures:
            - Accents are removed
            - Whitespace is normalized
            - Lowercasing is applied

        Returns:
            None
    """
    out = normalize_text("  Héllo   World  ", to_case="lower", remove_accents=True)
    assert out == "hello world"

def test_entity_validation_ok() -> None:
    """
        Test validation of a correct Entity object

        Ensures:
            - Valid entities pass validation without error

        Returns:
            None
    """
    ent = Entity(
        id="ent_000001",
        text="aspirin",
        start=0,
        end=7,
        label=EntityLabel.MEDICATION,
        concept_id="D001241",
        concept_name="Aspirin",
        source=EntityProvenance.MESH_AUTO,
        negation=NegationStatus.NOT_NEGATED,
        temporality="current",
        confidence=0.9,
        meta={},
    )
    ent.validate()

def test_entity_validation_invalid_id() -> None:
    """
        Test validation failure on invalid entity identifier

        Ensures:
            - Invalid entity id format raises DataError

        Returns:
            None
    """
    ent = Entity(
        id="bad_id",
        text="aspirin",
        start=0,
        end=7,
        label=EntityLabel.MEDICATION,
        meta={},
    )

    with pytest.raises(DataError):
        ent.validate()

def test_entity_validation_future_for_disease_raises() -> None:
    """
        Test business rule: DISEASE cannot have future temporality

        Ensures:
            - DISEASE with temporality future raises DataError

        Returns:
            None
    """
    ent = Entity(
        id="ent_000001",
        text="flu",
        start=0,
        end=3,
        label=EntityLabel.DISEASE,
        temporality="future",
        meta={},
    )

    with pytest.raises(DataError):
        ent.validate()

def test_record_validation_duplicate_entity_ids() -> None:
    """
        Test record validation failure on duplicate entity identifiers

        Ensures:
            - Duplicate entity ids raise DataError

        Returns:
            None
    """
    ent1 = Entity(
        id="ent_000001",
        text="aspirin",
        start=0,
        end=7,
        label=EntityLabel.MEDICATION,
        meta={},
    )
    ent2 = Entity(
        id="ent_000001",
        text="flu",
        start=13,
        end=16,
        label=EntityLabel.DISEASE,
        meta={},
    )

    rec = Record(
        record_id="rec_000001",
        patient_id="pat_000001",
        name_document="doc.txt",
        type_document="note",
        text="aspirin then flu",
        entities=[ent1, ent2],
        date_document="2025-01-01",
    )

    with pytest.raises(DataError):
        rec.validate(validate_spans=True)

def test_record_validation_span_out_of_bounds_raises() -> None:
    """
        Test record validation failure on span out of bounds

        Ensures:
            - Out of bounds spans raise DataError

        Returns:
            None
    """
    ent = Entity(
        id="ent_000001",
        text="aspirin",
        start=0,
        end=999,
        label=EntityLabel.MEDICATION,
        meta={},
    )

    rec = Record(
        record_id="rec_000001",
        patient_id="pat_000001",
        name_document="doc.txt",
        type_document="note",
        text="aspirin",
        entities=[ent],
    )

    with pytest.raises(DataError):
        rec.validate(validate_spans=True)

def test_negation_rules_basic() -> None:
    """
        Test basic negation rule application

        Ensures:
            - Negation detection assigns a value

        Returns:
            None
    """
    ent = Entity(
        id="ent_000001",
        text="asthma",
        start=14,
        end=20,
        label=EntityLabel.DISEASE,
        meta={},
    )
    rec = Record(
        record_id="rec_000001",
        patient_id="pat_000001",
        name_document="doc.txt",
        type_document="note",
        text="No history of asthma",
        entities=[ent],
    )

    apply_negation_rules(rec, window=5)

    assert rec.entities[0].negation in (
        NegationStatus.NEGATED,
        NegationStatus.NOT_NEGATED,
    )

def test_temporality_rules_basic() -> None:
    """
        Test temporality rule application

        Ensures:
            - Temporality inference does not crash
            - Assigned temporality is valid for DISEASE

        Returns:
            None
    """
    ent = Entity(
        id="ent_000001",
        text="diabetes",
        start=8,
        end=16,
        label=EntityLabel.DISEASE,
        meta={},
    )
    rec = Record(
        record_id="rec_000001",
        patient_id="pat_000001",
        name_document="doc.txt",
        type_document="note",
        text="chronic diabetes",
        entities=[ent],
    )

    apply_temporality_rules(rec)

    assert rec.entities[0].temporality in (
        None,
        "past",
        "current",
        "chronic",
    )

def test_pipeline_no_input_raises() -> None:
    """
        Test pipeline behavior with missing inputs

        Ensures:
            - Missing inputs raise ConfigurationError

        Returns:
            None
    """
    cfg = ProjectConfig.from_env(project_root=Path.cwd())

    with pytest.raises(ConfigurationError):
        run_pipeline(cfg=cfg)

def test_pipeline_unlabeled_smoke(tmp_path: Path) -> None:
    """
        Smoke test for unlabeled pipeline execution

        Ensures:
            - Pipeline runs on raw text input
            - Output CSV file is generated

        Args:
            tmp_path: Temporary pytest directory

        Returns:
            None
    """
    raw_dir = tmp_path / "raw_texts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "doc1.txt").write_text("Patient denies asthma", encoding="utf-8")

    cfg = ProjectConfig.from_env(project_root=tmp_path)

    out = run_pipeline(
        cfg=cfg,
        unlabeled_texts_dir=raw_dir,
        output_csv_path=tmp_path / "out.csv",
    )

    assert out.exists()
    assert out.suffix == ".csv"

def test_record_validation_ok() -> None:
    """
        Test validation of a correct Record object

        Ensures:
            - Valid record passes validation without error

        Returns:
            None
    """

    ent = Entity(
        id="ent_000001",
        text="aspirin",
        start=0,
        end=7,
        label=EntityLabel.MEDICATION,
        meta={},
    )

    rec = Record(
        record_id="rec_000001",
        patient_id="pat_000001",
        name_document="doc.txt",
        type_document="note",
        text="aspirin",
        entities=[ent],
        date_document="2025-01-01",
    )

    rec.validate(validate_spans=True)
    
def test_negation_rules_empty_entities() -> None:
    """
        Test negation rules on a record without entities

        Ensures:
            - Rule application does not crash on empty entity list

        Returns:
            None
    """

    rec = Record(
        record_id="rec_000001",
        patient_id="pat_000001",
        name_document="doc.txt",
        type_document="note",
        text="No relevant finding",
        entities=[],
    )

    apply_negation_rules(rec, window=5)

    assert rec.entities == []
    
def test_temporality_rules_empty_entities() -> None:
    """
        Test temporality rules on a record without entities

        Ensures:
            - Rule application does not crash on empty entity list

        Returns:
            None
    """

    rec = Record(
        record_id="rec_000001",
        patient_id="pat_000001",
        name_document="doc.txt",
        type_document="note",
        text="No relevant finding",
        entities=[],
    )

    apply_temporality_rules(rec)

    assert rec.entities == []
    
def test_pipeline_labeled_csv_smoke(tmp_path: Path) -> None:
    """
        Smoke test for labeled CSV pipeline execution

        Ensures:
            - Pipeline runs on labeled CSV input
            - Output CSV file is generated

        Args:
            tmp_path: Temporary pytest directory

        Returns:
            None
    """

    labeled_csv = tmp_path / "labeled.csv"
    labeled_csv.write_text(
        "text,label\nPatient has asthma,DISEASE\n",
        encoding="utf-8",
    )

    cfg = ProjectConfig.from_env(project_root=tmp_path)

    out = run_pipeline(
        cfg=cfg,
        labeled_csv_path=labeled_csv,
        output_csv_path=tmp_path / "out.csv",
    )

    assert out.exists()
    assert out.suffix == ".csv"
    
## ============================================================
## DATA CONSISTENCY TESTS (NER)
## ============================================================
def test_data_consistency_valid_ner() -> None:
    """
        Validate correct NER payload

        Returns:
            None
    """

    data = {
        "text": "patient asthma",
        "entities": [
            {"start": 8, "end": 14, "label": "DISEASE"}
        ],
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is True

def test_data_consistency_invalid_span() -> None:
    """
        Detect invalid entity span

        Returns:
            None
    """

    data = {
        "text": "test",
        "entities": [
            {"start": 0, "end": 999, "label": "TEST"}
        ],
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is False

def test_data_consistency_overlap() -> None:
    """
        Detect overlapping entities

        Returns:
            None
    """

    data = {
        "text": "abcdef",
        "entities": [
            {"start": 0, "end": 3, "label": "A"},
            {"start": 2, "end": 5, "label": "B"},
        ],
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is True  ## overlap = warning

def test_data_consistency_empty_text() -> None:
    """
        Detect empty text

        Returns:
            None
    """

    data = {
        "text": "",
        "entities": [
            {"start": 0, "end": 1, "label": "A"}
        ],
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is False
    
## ============================================================
## DATA QUALITY TESTS
## ============================================================
def test_data_quality_valid() -> None:
    """
        Test valid dataset for data quality

        Returns:
            None
    """

    texts = ["patient has asthma"]
    labels = [["O", "O", "B-DISEASE"]]

    result = run_data_quality(texts=texts, labels=labels)

    assert result["is_valid"] is True

def test_data_quality_empty_text() -> None:
    """
        Detect empty text anomaly

        Returns:
            None
    """

    texts = [""]
    labels = [["O"]]

    result = run_data_quality(texts=texts, labels=labels)

    assert result["is_valid"] is False

def test_data_quality_length_anomaly() -> None:
    """
        Detect abnormal text length

        Returns:
            None
    """

    texts = ["short", "this is a very very very very very long clinical sentence"]
    labels = [["O"], ["O"] * 12]

    result = run_data_quality(texts=texts, labels=labels, method="zscore")

    assert "sequence_length_anomaly" in [i["rule"] for i in result["issues"]]

def test_data_quality_invalid_bio() -> None:
    """
        Detect invalid BIO tags

        Returns:
            None
    """

    texts = ["patient asthma"]
    labels = [["X", "Y"]]

    result = run_data_quality(texts=texts, labels=labels)

    assert result["is_valid"] is False

def test_data_quality_strict_mode() -> None:
    """
        Strict mode raises error

        Returns:
            None
    """

    texts = [""]
    labels = [["O"]]

    with pytest.raises(Exception):
        run_data_quality(
            texts=texts,
            labels=labels,
            strict=True,
        )