'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Central project configuration for Clinical NER: paths, runtime flags, models and dictionaries."
'''

from __future__ import annotations

## Standard library imports
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

## Centralized errors and logging
from src.core.errors import ConfigurationError
from src.utils.logging_utils import get_logger

## Core domain imports
from src.core.entities import (
    EntityLabel,
    TEMPORALITY_LABELS_MEDICATION,
    TEMPORALITY_LABELS_PATHOLOGY,
)

## Generic utilities
from src.utils.utils import ensure_str, ensure_str_or_none


## Module-level logger
logger = get_logger(name="clinical_ner.config")


def _parse_bool(value: str | None, default: bool) -> bool:
    """
        Parse boolean from environment-like strings

        Args:
            value: Raw string value
            default: Default boolean if value is None/empty

        Returns:
            Parsed boolean
    """
    if value is None:
        return default

    raw = ensure_str(value).strip().lower()
    if raw == "":
        return default

    return raw in ("1", "true", "yes", "y", "on")


def _parse_int(value: str | None, default: int) -> int:
    """
        Parse integer from string

        Args:
            value: Raw string value
            default: Default integer if value is None/empty

        Returns:
            Parsed integer

        Raises:
            ConfigurationError: If conversion fails
    """
    if value is None:
        return default

    raw = ensure_str(value).strip()
    if raw == "":
        return default

    try:
        return int(raw)
    except Exception as exc:
        msg = f"Invalid integer value: {value}"
        logger.error(msg)
        raise ConfigurationError(msg) from exc


def _parse_float(value: str | None, default: float) -> float:
    """
        Parse float from string

        Args:
            value: Raw string value
            default: Default float if value is None/empty

        Returns:
            Parsed float

        Raises:
            ConfigurationError: If conversion fails
    """
    if value is None:
        return default

    raw = ensure_str(value).strip()
    if raw == "":
        return default

    try:
        return float(raw)
    except Exception as exc:
        msg = f"Invalid float value: {value}"
        logger.error(msg)
        raise ConfigurationError(msg) from exc


def _validate_confidence(value: float, name: str) -> None:
    """
        Validate confidence thresholds

        Args:
            value: Threshold value
            name: Threshold name

        Raises:
            ConfigurationError: If out of [0, 1]
    """
    if not 0.0 <= value <= 1.0:
        msg = f"{name} must be in [0, 1], got {value}"
        logger.error(msg)
        raise ConfigurationError(msg)


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """
        Centralized filesystem paths for the Clinical NER project

        Attributes:
            project_root: Root directory of the project
            data_raw: Folder for raw input data
            data_annotated: Folder for labeled datasets
            data_interim: Folder for intermediate artifacts
            data_processed: Folder for processed datasets
            artifacts_models: Folder for trained models
            artifacts_reports: Folder for reports and metrics
            artifacts_exports: Folder for CSV/JSON exports
            artifacts_dictionaries: Folder for dictionaries (MeSH, custom)
            logs_dir: Folder for application logs
    """

    project_root: Path
    data_raw: Path
    data_annotated: Path
    data_interim: Path
    data_processed: Path
    artifacts_models: Path
    artifacts_reports: Path
    artifacts_exports: Path
    artifacts_dictionaries: Path
    logs_dir: Path

    @staticmethod
    def from_root(project_root: str | Path) -> "ProjectPaths":
        """
            Build project paths from a root directory

            Args:
                project_root: Root directory of the project

            Returns:
                ProjectPaths instance
        """
        root = Path(project_root).resolve()

        ## Validate root existence early
        if not root.exists():
            msg = f"Project root does not exist: {root}"
            logger.error(msg)
            raise ConfigurationError(msg)

        data_dir = root / "data"
        artifacts_dir = root / "artifacts"

        return ProjectPaths(
            project_root=root,
            data_raw=data_dir / "raw",
            data_annotated=data_dir / "annotated",
            data_interim=data_dir / "interim",
            data_processed=data_dir / "processed",
            artifacts_models=artifacts_dir / "models",
            artifacts_reports=artifacts_dir / "reports",
            artifacts_exports=artifacts_dir / "exports",
            artifacts_dictionaries=artifacts_dir / "dictionaries",
            logs_dir=root / "logs",
        )

    def ensure_dirs(self) -> None:
        """
            Ensure all project directories exist
        """
        for path in (
            self.data_raw,
            self.data_annotated,
            self.data_interim,
            self.data_processed,
            self.artifacts_models,
            self.artifacts_reports,
            self.artifacts_exports,
            self.artifacts_dictionaries,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class RuntimeConfig:
    """
        Runtime configuration flags for pipeline execution

        Attributes:
            accept_labeled_data: Allow labeled CSV input
            accept_unlabeled_texts: Allow raw .txt input
            enable_negation: Enable negation detection
            enable_temporality: Enable temporality detection
            device: Execution device (cpu, cuda, auto)
            seed: Global random seed
            max_docs: Optional cap on number of processed documents
            max_entities_per_record: Safety limit per document
            dictionary_min_confidence: Threshold for dictionary matching
            model_min_confidence: Threshold for model predictions
    """

    accept_labeled_data: bool = True
    accept_unlabeled_texts: bool = True

    enable_negation: bool = True
    enable_temporality: bool = True

    device: str = "auto"
    seed: int = 42

    max_docs: int | None = None
    max_entities_per_record: int = 300

    dictionary_min_confidence: float = 0.7
    model_min_confidence: float = 0.5

    ner_labels: tuple[EntityLabel, ...] = field(
        default_factory=lambda: (
            EntityLabel.MEDICATION,
            EntityLabel.DISEASE,
            EntityLabel.ALLERGY,
            EntityLabel.PROCEDURE,
            EntityLabel.TEST,
            EntityLabel.ANATOMY,
        )
    )

    temporality_labels_medication: tuple[EntityLabel, ...] = field(
        default_factory=lambda: TEMPORALITY_LABELS_MEDICATION
    )

    temporality_labels_pathology: tuple[EntityLabel, ...] = field(
        default_factory=lambda: TEMPORALITY_LABELS_PATHOLOGY
    )

    def resolved_device(self) -> str:
        """
            Resolve the execution device

            Returns:
                Final device string: cpu or cuda
        """
        raw = ensure_str(self.device).strip().lower()

        if raw in ("cpu", "cuda"):
            return raw

        ## Auto-detect cuda if torch is available
        try:
            import torch  # type: ignore
        except Exception:
            return "cpu"

        return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(slots=True)
class ModelConfig:
    """
        Configuration for NER, negation and temporality models

        Attributes:
            hf_ner_model_name: HF model name for token classification
            spacy_model_name: spaCy model name or local path
            negation_strategy: rules or model
            temporality_strategy: rules or model
    """

    hf_ner_model_name: str = "distilbert-base-multilingual-cased"
    spacy_model_name: str | None = None

    negation_strategy: str = "rules"
    temporality_strategy: str = "rules"


@dataclass(slots=True)
class DictionaryConfig:
    """
        Configuration for dictionary-based auto-labeling

        Attributes:
            dictionaries_root: Root directory for dictionary files
            enable_fuzzy: Enable fuzzy matching
            fuzzy_max_distance: Maximum edit distance for fuzzy match
            enable_embeddings: Enable embedding-based matching
            embeddings_model_name: Sentence-transformers model name
    """

    dictionaries_root: Path

    enable_fuzzy: bool = True
    fuzzy_max_distance: int = 1

    enable_embeddings: bool = False
    embeddings_model_name: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


@dataclass(slots=True)
class ProjectConfig:
    """
        Full project configuration container
    """

    paths: ProjectPaths
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    dictionaries: DictionaryConfig | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def ensure_dirs(self) -> None:
        """
            Ensure required project directories exist
        """
        self.paths.ensure_dirs()

    @staticmethod
    def from_env(project_root: str | Path | None = None) -> "ProjectConfig":
        """
            Build ProjectConfig from environment variables

            Args:
                project_root: Optional project root override

            Returns:
                ProjectConfig instance

            Raises:
                ConfigurationError: If config is invalid
        """

        ## Resolve project root
        root = (
            project_root
            or ensure_str_or_none(os.getenv("CLINICAL_NER_ROOT"))
            or Path.cwd()
        )

        paths = ProjectPaths.from_root(root)

        ## Runtime flags from env
        accept_labeled = _parse_bool(
            ensure_str_or_none(os.getenv("CLINICAL_NER_ACCEPT_LABELED_DATA")),
            default=True,
        )
        accept_unlabeled = _parse_bool(
            ensure_str_or_none(os.getenv("CLINICAL_NER_ACCEPT_UNLABELED_TEXTS")),
            default=True,
        )

        enable_negation = _parse_bool(
            ensure_str_or_none(os.getenv("CLINICAL_NER_ENABLE_NEGATION")),
            default=True,
        )
        enable_temporality = _parse_bool(
            ensure_str_or_none(os.getenv("CLINICAL_NER_ENABLE_TEMPORALITY")),
            default=True,
        )

        ## Thresholds
        dictionary_min_conf = _parse_float(
            ensure_str_or_none(os.getenv("CLINICAL_NER_DICTIONARY_MIN_CONFIDENCE")),
            default=0.7,
        )
        model_min_conf = _parse_float(
            ensure_str_or_none(os.getenv("CLINICAL_NER_MODEL_MIN_CONFIDENCE")),
            default=0.5,
        )

        _validate_confidence(dictionary_min_conf, "CLINICAL_NER_DICTIONARY_MIN_CONFIDENCE")
        _validate_confidence(model_min_conf, "CLINICAL_NER_MODEL_MIN_CONFIDENCE")

        ## Max docs
        max_docs_raw = ensure_str_or_none(os.getenv("CLINICAL_NER_MAX_DOCS"))
        max_docs = None if max_docs_raw in (None, "") else _parse_int(max_docs_raw, default=0)
        if max_docs is not None and max_docs <= 0:
            msg = "CLINICAL_NER_MAX_DOCS must be positive when provided"
            logger.error(msg)
            raise ConfigurationError(msg)

        runtime = RuntimeConfig(
            accept_labeled_data=accept_labeled,
            accept_unlabeled_texts=accept_unlabeled,
            enable_negation=enable_negation,
            enable_temporality=enable_temporality,
            device=ensure_str_or_none(os.getenv("CLINICAL_NER_DEVICE")) or "auto",
            seed=_parse_int(ensure_str_or_none(os.getenv("CLINICAL_NER_SEED")), default=42),
            max_docs=max_docs,
            max_entities_per_record=_parse_int(
                ensure_str_or_none(os.getenv("CLINICAL_NER_MAX_ENTITIES_PER_RECORD")),
                default=300,
            ),
            dictionary_min_confidence=dictionary_min_conf,
            model_min_confidence=model_min_conf,
        )

        ## Validate runtime flags
        if not runtime.accept_labeled_data and not runtime.accept_unlabeled_texts:
            msg = "Both labeled and unlabeled inputs are disabled"
            logger.error(msg)
            raise ConfigurationError(msg)

        ## Validate device early
        _ = runtime.resolved_device()

        ## Models config from env
        models = ModelConfig(
            hf_ner_model_name=ensure_str_or_none(os.getenv("CLINICAL_NER_HF_NER_MODEL"))
            or "distilbert-base-multilingual-cased",
            spacy_model_name=ensure_str_or_none(os.getenv("CLINICAL_NER_SPACY_MODEL_NAME")),
            negation_strategy=ensure_str_or_none(os.getenv("CLINICAL_NER_NEGATION_STRATEGY"))
            or "rules",
            temporality_strategy=ensure_str_or_none(os.getenv("CLINICAL_NER_TEMPORALITY_STRATEGY"))
            or "rules",
        )

        ## Dictionaries config from env
        dictionaries = DictionaryConfig(
            dictionaries_root=paths.artifacts_dictionaries,
            enable_fuzzy=_parse_bool(
                ensure_str_or_none(os.getenv("CLINICAL_NER_DICTIONARIES_FUZZY")),
                default=True,
            ),
            fuzzy_max_distance=_parse_int(
                ensure_str_or_none(os.getenv("CLINICAL_NER_DICTIONARIES_FUZZY_MAX_DISTANCE")),
                default=1,
            ),
            enable_embeddings=_parse_bool(
                ensure_str_or_none(os.getenv("CLINICAL_NER_DICTIONARIES_USE_EMBEDDINGS")),
                default=False,
            ),
            embeddings_model_name=ensure_str_or_none(
                os.getenv("CLINICAL_NER_DICTIONARIES_EMBEDDINGS_MODEL")
            )
            or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )

        cfg = ProjectConfig(
            paths=paths,
            runtime=runtime,
            models=models,
            dictionaries=dictionaries,
        )

        ## Ensure expected directories exist
        cfg.ensure_dirs()

        return cfg
