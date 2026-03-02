'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unified configuration management: environment variables, paths resolution, runtime identifiers and MLflow defaults."
'''

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.core.errors import ConfigurationError
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger(__name__)

## ============================================================
## CONSTANTS
## ============================================================
DEFAULT_DATA_DIR = "data"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_ARTIFACTS_DIR = "artifacts"

DEFAULT_MLFLOW_DIR = "artifacts/mlflow"
DEFAULT_MLFLOW_EXPERIMENT = "lab_clustering"

DEFAULT_STRUCTURED_DIRNAME = "lab_structured_csv"
DEFAULT_DATASETS_DIRNAME = "datasets"
DEFAULT_FEATURES_DIRNAME = "features"
DEFAULT_ERROR_ANALYSIS_DIRNAME = "error_analysis"

DEFAULT_RESOURCES_DIR = "artifacts/resources"

## ============================================================
## ENV HELPERS
## ============================================================
def _get_env(name: str, default: Optional[str] = None) -> str:
    """
        Read environment variable safely

        Args:
            name: Environment variable name
            default: Optional default value

        Returns:
            Environment variable value

        Raises:
            ConfigurationError: If missing and no default provided
    """

    value = os.getenv(name)

    if value is None:
        if default is None:
            logger.error("Missing environment variable: %s", name)
            raise ConfigurationError(
                message=f"Missing environment variable: {name}",
                details={"missing": [name]},
            )
        return default

    return value.strip()

def _to_bool(value: str) -> bool:
    """
        Convert string to boolean

        Args:
            value: String representation

        Returns:
            Boolean value

        Raises:
            ConfigurationError: If invalid format
    """

    normalized = value.strip().lower()

    if normalized in {"true", "1", "yes", "y"}:
        return True

    if normalized in {"false", "0", "no", "n"}:
        return False

    raise ConfigurationError(
        message=f"Invalid boolean value: {value}",
        details={"value": value},
    )

def _resolve_project_root() -> Path:
    """
        Resolve project root directory

        Returns:
            Absolute project root path
    """

    return Path(__file__).resolve().parents[2]

def _ensure_dir(path: Path) -> Path:
    """
        Ensure a directory exists

        Args:
            path: Directory path

        Returns:
            Same path (created if missing)
    """

    path.mkdir(parents=True, exist_ok=True)
    return path

## ============================================================
## DATA CLASSES
## ============================================================
@dataclass(frozen=True)
class PathsConfig:
    """
        Filesystem paths configuration
    """

    project_root: Path

    data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path

    interim_structured_dir: Path
    interim_datasets_dir: Path

    processed_features_dir: Path
    processed_error_analysis_dir: Path

    artifacts_dir: Path
    artifacts_models_dir: Path
    artifacts_exports_dir: Path
    artifacts_resources_dir: Path
    artifacts_config_dir: Path

    logs_dir: Path

@dataclass(frozen=True)
class MlflowConfig:
    """
        MLflow configuration
    """

    tracking_uri: str
    experiment_name: str

@dataclass(frozen=True)
class RuntimeConfig:
    """
        Runtime configuration
    """

    run_id: str
    use_gpu: bool
    random_seed: int

@dataclass(frozen=True)
class AppConfig:
    """
        Unified application configuration
    """

    paths: PathsConfig
    mlflow: MlflowConfig
    runtime: RuntimeConfig

## ============================================================
## BUILD CONFIG
## ============================================================
def build_config() -> AppConfig:
    """
        Build full application configuration from environment variables

        Environment variables:
            DATA_DIR
            LOGS_DIR
            ARTIFACTS_DIR
            RESOURCES_DIR
            USE_GPU
            RANDOM_SEED
            MLFLOW_TRACKING_URI
            MLFLOW_EXPERIMENT_NAME

        Returns:
            AppConfig instance
    """

    project_root = _resolve_project_root()

    ## PATHS (ROOT)
    data_dir = project_root / _get_env("DATA_DIR", DEFAULT_DATA_DIR)
    logs_dir = project_root / _get_env("LOGS_DIR", DEFAULT_LOGS_DIR)
    artifacts_dir = project_root / _get_env("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR)

    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"

    ## PATHS (INTERIM / PROCESSED)
    interim_structured_dir = interim_dir / DEFAULT_STRUCTURED_DIRNAME
    interim_datasets_dir = interim_dir / DEFAULT_DATASETS_DIRNAME

    processed_features_dir = processed_dir / DEFAULT_FEATURES_DIRNAME
    processed_error_analysis_dir = processed_dir / DEFAULT_ERROR_ANALYSIS_DIRNAME

    ## PATHS (ARTIFACTS)
    artifacts_models_dir = artifacts_dir / "models"
    artifacts_exports_dir = artifacts_dir / "exports"
    artifacts_config_dir = artifacts_dir / "config"

    resources_dir_raw = _get_env("RESOURCES_DIR", DEFAULT_RESOURCES_DIR)
    artifacts_resources_dir = project_root / resources_dir_raw

    ## Ensure folders exist
    _ensure_dir(data_dir)
    _ensure_dir(raw_dir)
    _ensure_dir(interim_dir)
    _ensure_dir(processed_dir)

    _ensure_dir(interim_structured_dir)
    _ensure_dir(interim_datasets_dir)
    _ensure_dir(processed_features_dir)
    _ensure_dir(processed_error_analysis_dir)

    _ensure_dir(artifacts_dir)
    _ensure_dir(artifacts_models_dir)
    _ensure_dir(artifacts_exports_dir)
    _ensure_dir(artifacts_resources_dir)
    _ensure_dir(artifacts_config_dir)

    _ensure_dir(logs_dir)

    paths = PathsConfig(
        project_root=project_root,
        data_dir=data_dir,
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        interim_structured_dir=interim_structured_dir,
        interim_datasets_dir=interim_datasets_dir,
        processed_features_dir=processed_features_dir,
        processed_error_analysis_dir=processed_error_analysis_dir,
        artifacts_dir=artifacts_dir,
        artifacts_models_dir=artifacts_models_dir,
        artifacts_exports_dir=artifacts_exports_dir,
        artifacts_resources_dir=artifacts_resources_dir,
        artifacts_config_dir=artifacts_config_dir,
        logs_dir=logs_dir,
    )

    ## MLFLOW
    default_mlflow_uri = str(project_root / DEFAULT_MLFLOW_DIR)
    mlflow_tracking_uri = _get_env("MLFLOW_TRACKING_URI", default_mlflow_uri)
    mlflow_experiment_name = _get_env(
        "MLFLOW_EXPERIMENT_NAME",
        DEFAULT_MLFLOW_EXPERIMENT,
    )

    mlflow_cfg = MlflowConfig(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment_name,
    )

    ## RUNTIME
    use_gpu = _to_bool(_get_env("USE_GPU", "false"))
    random_seed = int(_get_env("RANDOM_SEED", "42"))

    runtime = RuntimeConfig(
        run_id=str(uuid.uuid4()),
        use_gpu=use_gpu,
        random_seed=random_seed,
    )

    return AppConfig(paths=paths, mlflow=mlflow_cfg, runtime=runtime)