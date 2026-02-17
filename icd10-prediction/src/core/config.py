'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unified configuration management: environment variables, paths resolution, runtime identifiers and model defaults."
'''

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.core.errors import ConfigurationError, log_and_raise_missing_env

## ============================================================
## CONSTANTS
## ============================================================
DEFAULT_DATA_DIR = "data"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_ARTIFACTS_DIR = "artifacts"

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
            log_and_raise_missing_env([name])
        return default  # type: ignore

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

    raise ConfigurationError(f"Invalid boolean value: {value}")

def _resolve_project_root() -> Path:
    """
        Resolve project root directory

        Returns:
            Absolute project root path
    """
    return Path(__file__).resolve().parents[2]

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
    artifacts_dir: Path
    logs_dir: Path

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
            USE_GPU
            RANDOM_SEED

        Returns:
            AppConfig instance
    """
    
    project_root = _resolve_project_root()

    ## -----------------------------
    ## PATHS
    ## -----------------------------
    data_dir = project_root / _get_env("DATA_DIR", DEFAULT_DATA_DIR)
    logs_dir = project_root / _get_env("LOGS_DIR", DEFAULT_LOGS_DIR)
    artifacts_dir = project_root / _get_env("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR)

    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"

    paths = PathsConfig(
        project_root=project_root,
        data_dir=data_dir,
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        artifacts_dir=artifacts_dir,
        logs_dir=logs_dir,
    )

    ## -----------------------------
    ## RUNTIME
    ## -----------------------------
    use_gpu = _to_bool(_get_env("USE_GPU", "false"))
    random_seed = int(_get_env("RANDOM_SEED", "42"))

    runtime = RuntimeConfig(
        run_id=str(uuid.uuid4()),
        use_gpu=use_gpu,
        random_seed=random_seed,
    )

    return AppConfig(paths=paths, runtime=runtime)