'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Application settings management: load .env configuration and resolve project/data paths."
'''

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field


## ============================================================
## PATH HELPERS
## ============================================================

def get_project_root() -> Path:
    """
    Resolve the project root directory.

    The project root is assumed to be the parent directory of this file's
    grandparent folder (i.e., .../mesh_semantic_expansion/).

    Returns:
        Path: Absolute path to the project root.
    """

    return Path(__file__).resolve().parents[2]


def _to_bool(value: Optional[str], default: bool = False) -> bool:
    """
    Convert an environment variable string into a boolean.

    Args:
        value (Optional[str]): Input value, typically from environment variables.
        default (bool): Fallback if value is None or empty.

    Returns:
        bool: Parsed boolean.
    """

    if value is None:
        return default

    val = value.strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False

    return default


## ============================================================
## SETTINGS MODEL
## ============================================================

class Settings(BaseModel):
    """
    Central application settings loaded from environment variables.

    Notes:
        - This project expects a `.env` file at the project root.
        - Values are read from OS environment (dotenv loading is done elsewhere).
    """

    ## Core
    environment: str = Field(default="dev", description="Execution environment: dev|prod.")
    app_version: str = Field(default="1.0.0", description="Application version.")
    log_level: str = Field(default="INFO", description="Log level: DEBUG|INFO|WARNING|ERROR.")
    api_key: Optional[str] = Field(default=None, description="Optional API key for route protection.")

    ## Data roots
    data_dir: Path = Field(default_factory=lambda: get_project_root() / "data", description="Base data directory.")
    logs_dir: Path = Field(default_factory=lambda: get_project_root() / "logs", description="Base logs directory.")

    ## Raw data
    raw_mesh_dir: Path = Field(default_factory=lambda: get_project_root() / "data" / "raw" / "mesh")
    raw_medical_docs_dir: Path = Field(default_factory=lambda: get_project_root() / "data" / "raw" / "medical_docs")

    ## Interim files
    mesh_parsed_file: Path = Field(default_factory=lambda: get_project_root() / "data" / "interim" / "mesh_parsed.jsonl")
    doc_embeddings_file: Path = Field(default_factory=lambda: get_project_root() / "data" / "interim" / "doc_embeddings.parquet")
    mesh_embeddings_file: Path = Field(default_factory=lambda: get_project_root() / "data" / "interim" / "mesh_embeddings.parquet")

    ## Processed files
    entities_detected_file: Path = Field(default_factory=lambda: get_project_root() / "data" / "processed" / "entities_detected.jsonl")
    candidates_file: Path = Field(default_factory=lambda: get_project_root() / "data" / "processed" / "candidates.jsonl")

    ## Outputs
    export_candidates_csv: Path = Field(default_factory=lambda: get_project_root() / "data" / "outputs" / "export_candidates.csv")
    export_candidates_validated_csv: Path = Field(default_factory=lambda: get_project_root() / "data" / "outputs" / "export_candidates_validated.csv")
    mesh_extended_json: Path = Field(default_factory=lambda: get_project_root() / "data" / "outputs" / "mesh_extended.json")
    report_diff_md: Path = Field(default_factory=lambda: get_project_root() / "data" / "outputs" / "report_diff.md")

    ## Feature flags (optional)
    enable_api_auth: bool = Field(default=False, description="Enable API key auth if API_KEY is defined.")
    create_dirs_on_start: bool = Field(default=True, description="Auto-create required directories on startup.")


## ============================================================
## ENV LOADING (SIMPLE)
## ============================================================

def load_dotenv_if_present(env_path: Optional[Path] = None) -> None:
    """
    Load environment variables from a .env file if present.

    This implementation is intentionally lightweight:
    - It parses KEY=VALUE lines
    - Ignores comments and blank lines
    - Does not override already existing environment variables

    Args:
        env_path (Optional[Path]): Path to the .env file. If None, defaults to <project_root>/.env
    """

    if env_path is None:
        env_path = get_project_root() / ".env"

    if not env_path.exists():
        return

    ## Read and parse .env lines
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()

        ## Skip blanks and comments
        if not stripped or stripped.startswith("#"):
            continue

        if "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        ## Do not override existing environment variables
        if key and key not in os.environ:
            os.environ[key] = value


def ensure_directories(settings: Settings) -> None:
    """
    Ensure all required directories exist.

    Args:
        settings (Settings): Loaded application settings.
    """

    dirs = [
        settings.data_dir,
        settings.logs_dir,
        settings.raw_mesh_dir,
        settings.raw_medical_docs_dir,
        settings.mesh_parsed_file.parent,
        settings.doc_embeddings_file.parent,
        settings.mesh_embeddings_file.parent,
        settings.entities_detected_file.parent,
        settings.candidates_file.parent,
        settings.export_candidates_csv.parent,
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


## ============================================================
## SETTINGS FACTORY (CACHED)
## ============================================================

@lru_cache(maxsize=1)
def get_settings(env_path: Optional[Path] = None) -> Settings:
    """
    Build and cache the application settings.

    Args:
        env_path (Optional[Path]): Optional .env path override.

    Returns:
        Settings: Loaded settings instance.
    """

    ## Load .env first (if any)
    load_dotenv_if_present(env_path)

    ## Build settings from environment variables
    settings = Settings(
        environment=os.getenv("ENVIRONMENT", "dev"),
        app_version=os.getenv("APP_VERSION", "1.0.0"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        api_key=os.getenv("API_KEY"),
        enable_api_auth=_to_bool(os.getenv("ENABLE_API_AUTH"), default=False),
        create_dirs_on_start=_to_bool(os.getenv("CREATE_DIRS_ON_START"), default=True),
    )

    ## Create directories if requested
    if settings.create_dirs_on_start:
        ensure_directories(settings)

    return settings


## ============================================================
## PUBLIC UTILS
## ============================================================

def get_data_paths(settings: Settings) -> Dict[str, Path]:
    """
    Convenience helper to return the main data paths.

    Args:
        settings (Settings): Settings instance.

    Returns:
        Dict[str, Path]: Named paths used by pipelines and services.
    """

    return {
        "project_root": get_project_root(),
        "data_dir": settings.data_dir,
        "logs_dir": settings.logs_dir,
        "raw_mesh_dir": settings.raw_mesh_dir,
        "raw_medical_docs_dir": settings.raw_medical_docs_dir,
        "mesh_parsed_file": settings.mesh_parsed_file,
        "doc_embeddings_file": settings.doc_embeddings_file,
        "mesh_embeddings_file": settings.mesh_embeddings_file,
        "entities_detected_file": settings.entities_detected_file,
        "candidates_file": settings.candidates_file,
        "export_candidates_csv": settings.export_candidates_csv,
        "export_candidates_validated_csv": settings.export_candidates_validated_csv,
        "mesh_extended_json": settings.mesh_extended_json,
        "report_diff_md": settings.report_diff_md,
    }
