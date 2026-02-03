'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unified configuration (env, paths, GPU, labels, defaults) for the medical document classification pipeline."
'''

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from src.core.errors import ConfigurationError

## -----------------------------
## Types
## -----------------------------
UseGpuMode = Literal["auto", "true", "false"]
EmbeddingModelName = Literal["sentence_camembert", "drbert"]

## -----------------------------
## Labels (multi-label, binary per label)
## -----------------------------
LABELS: Tuple[str, ...] = (
    "crh",
    "cro",
    "cra",
    "ordonnance_examen",
    "ordonnance_medicaments",
    "analyse_labo",
    "fiche_patient_admission",
)

LABEL_DESCRIPTIONS: Dict[str, str] = {
    "crh": "Hospital discharge summary (Compte Rendu d'Hospitalisation).",
    "cro": "Operative report (Compte Rendu Operatoire).",
    "cra": "Anesthesia report (Compte Rendu d'Anesthesie).",
    "ordonnance_examen": "Prescription for medical exams (imaging, lab, etc.).",
    "ordonnance_medicaments": "Medication prescription.",
    "analyse_labo": "Laboratory analysis results.",
    "fiche_patient_admission": "Patient admission or administrative intake form.",
}

## These hints are optional and non-deterministic (useful for EDA/debug).
LABEL_KEYWORD_HINTS: Dict[str, List[str]] = {
    "crh": ["compte rendu", "hospitalisation", "sortie", "diagnostic", "traitement"],
    "cro": ["compte rendu operatoire", "intervention", "bloc", "incision", "suture"],
    "cra": ["anesthesie", "induction", "intubation", "asa", "reveil"],
    "ordonnance_examen": ["prescription", "examen", "scanner", "irm", "radiographie"],
    "ordonnance_medicaments": ["posologie", "comprime", "mg", "prise", "renouvelable"],
    "analyse_labo": ["biochimie", "hematologie", "resultats", "normes", "valeurs"],
    "fiche_patient_admission": ["identite", "adresse", "assure", "mutuelle", "admission"],
}

## -----------------------------
## Dataclasses (config sections)
## -----------------------------
@dataclass(frozen=True)
class PathsConfig:
    """
        Filesystem paths configuration
    """
    project_root: Path
    data_dir: Path
    labeled_dir: Path
    unlabeled_dir: Path
    interim_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    indexes_dir: Path
    models_dir: Path
    reports_dir: Path
    exports_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class SegmentationConfig:
    """
        Segment (block) creation configuration
    """
    window_size_tokens: int
    window_overlap_tokens: int
    min_chars_per_segment: int


@dataclass(frozen=True)
class EmbeddingsConfig:
    """
        Embeddings configuration
    """
    model_name: EmbeddingModelName
    use_gpu: bool
    batch_size: int
    normalize: bool


@dataclass(frozen=True)
class SimilarityConfig:
    """
        Similarity search and thresholding configuration
    """
    top_k: int
    thresholds: Dict[str, float]


@dataclass(frozen=True)
class AppConfig:
    """
        Unified application configuration
    """
    paths: PathsConfig
    segmentation: SegmentationConfig
    embeddings: EmbeddingsConfig
    similarity: SimilarityConfig

## -----------------------------
## Environment helpers
## -----------------------------
def _get_env(name: str, default: str) -> str:
    """
        Read an environment variable safely
    """
    ## Avoid raising on missing variables
    return os.getenv(name, default).strip()


def _to_bool(value: str) -> bool:
    """
        Convert a string to boolean
    """
    ## Normalize input
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False

    ## Invalid boolean → config error
    raise ConfigurationError(f"Invalid boolean value: {value}")


def _resolve_project_root() -> Path:
    """
        Resolve project root path
    """
    ## Assume this file is located at: src/core/config.py
    return Path(__file__).resolve().parents[2]


def _detect_gpu_requested(mode: UseGpuMode) -> bool:
    """
        Determine if GPU usage is requested/available
    """
    ## Respect explicit override
    if mode == "true":
        return True
    if mode == "false":
        return False

    ## Auto mode: use GPU if torch reports it available
    try:
        import torch
    except Exception:
        ## Torch not installed or import error → CPU fallback
        return False

    return bool(torch.cuda.is_available())


def _default_thresholds() -> Dict[str, float]:
    """
        Provide default per-label thresholds
    """
    ## Conservative defaults; tune later
    return {label: 0.55 for label in LABELS}

## -----------------------------
## Config loader
## -----------------------------
def load_config() -> AppConfig:
    """
        Load the unified configuration from environment variables
    """

    ## --------------------------------------------------------
    ## Resolve root and base folders
    ## --------------------------------------------------------
    project_root = _resolve_project_root()

    data_dir = Path(_get_env("DATA_DIR", str(project_root / "data"))).resolve()
    artifacts_dir = Path(_get_env("ARTIFACTS_DIR", str(project_root / "artifacts"))).resolve()
    logs_dir = Path(_get_env("LOGS_DIR", str(project_root / "logs"))).resolve()

    ## --------------------------------------------------------
    ## Build nested artifact directories
    ## --------------------------------------------------------
    indexes_dir = artifacts_dir / "indexes"
    models_dir = artifacts_dir / "models"
    reports_dir = artifacts_dir / "reports"
    exports_dir = artifacts_dir / "exports"

    ## --------------------------------------------------------
    ## Build all paths
    ## --------------------------------------------------------
    paths = PathsConfig(
        project_root=project_root,
        data_dir=data_dir,
        labeled_dir=Path(_get_env("LABELED_DIR", str(data_dir / "labeled"))).resolve(),
        unlabeled_dir=Path(_get_env("UNLABELED_DIR", str(data_dir / "unlabeled"))).resolve(),
        interim_dir=Path(_get_env("INTERIM_DIR", str(data_dir / "interim"))).resolve(),
        processed_dir=Path(_get_env("PROCESSED_DIR", str(data_dir / "processed"))).resolve(),
        artifacts_dir=artifacts_dir,
        indexes_dir=indexes_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        exports_dir=exports_dir,
        logs_dir=logs_dir,
    )

    ## --------------------------------------------------------
    ## Segmentation parameters
    ## --------------------------------------------------------
    try:
        segmentation = SegmentationConfig(
            window_size_tokens=int(_get_env("WINDOW_SIZE_TOKENS", "220")),
            window_overlap_tokens=int(_get_env("WINDOW_OVERLAP_TOKENS", "60")),
            min_chars_per_segment=int(_get_env("MIN_CHARS_PER_SEGMENT", "50")),
        )
    except Exception as exc:
        raise ConfigurationError("Invalid segmentation env values") from exc

    ## --------------------------------------------------------
    ## GPU / embeddings parameters
    ## --------------------------------------------------------
    use_gpu_mode = _get_env("USE_GPU", "auto")
    if use_gpu_mode not in {"auto", "true", "false"}:
        raise ConfigurationError("USE_GPU must be auto|true|false")

    use_gpu = _detect_gpu_requested(use_gpu_mode)  # type: ignore[arg-type]

    embeddings = EmbeddingsConfig(
        model_name=_get_env("EMBEDDING_MODEL", "sentence_camembert"),  # type: ignore[arg-type]
        use_gpu=use_gpu,
        batch_size=int(_get_env("EMBEDDING_BATCH_SIZE", "32")),
        normalize=_to_bool(_get_env("EMBEDDING_NORMALIZE", "true")),
    )

    ## --------------------------------------------------------
    ## Similarity thresholds
    ## --------------------------------------------------------
    thresholds = _default_thresholds()
    for label in LABELS:
        env_key = f"THRESH_{label.upper()}"
        if env_key in os.environ:
            try:
                thresholds[label] = float(_get_env(env_key, str(thresholds[label])))
            except Exception as exc:
                raise ConfigurationError(f"{env_key} must be a float") from exc

    similarity = SimilarityConfig(
        top_k=int(_get_env("TOP_K", "5")),
        thresholds=thresholds,
    )

    return AppConfig(
        paths=paths,
        segmentation=segmentation,
        embeddings=embeddings,
        similarity=similarity,
    )

## -----------------------------
## Public singleton config
## -----------------------------
## This is the single source of truth used across the project.
CONFIG: AppConfig = load_config()
