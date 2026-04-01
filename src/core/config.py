'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unified configuration loader for mesh-semantic-expansion: dotenv, env parsing, paths, profiles, exports, secrets and runtime metadata."
'''

from __future__ import annotations

import json
import os
import platform
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Tuple

try:
    from src.core.errors import ConfigurationError
except Exception:
    class ConfigurationError(ValueError):
        """
            Fallback configuration error when the project error module is unavailable
        """

## ============================================================
## PLACEHOLDER TOKENS
## ============================================================
PLACEHOLDER_PREFIXES: Tuple[str, ...] = ("<YOUR_", "YOUR_", "CHANGE_ME", "REPLACE_ME", "TODO")

## ============================================================
## OS / SYSTEM CONSTANTS
## ============================================================
SYSTEM_NAME = platform.system().lower()
IS_WINDOWS = SYSTEM_NAME == "windows"
IS_LINUX = SYSTEM_NAME == "linux"
IS_MACOS = SYSTEM_NAME == "darwin"
DEFAULT_ENCODING = "utf-8"
CSV_SEPARATOR = ";"

## ============================================================
## STABLE DOMAIN CONSTANTS
## ============================================================
DEFAULT_APP_NAME = "mesh-semantic-expansion"
DEFAULT_APP_VERSION = "1.0.0"
DEFAULT_ENVIRONMENT = "dev"
DEFAULT_PROFILE = "cpu"

DEFAULT_DATA_DIR = "data"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_SECRETS_DIR = "secrets"

DEFAULT_RAW_MESH_DIR = "data/raw/mesh"
DEFAULT_RAW_MEDICAL_DOCS_DIR = "data/raw/medical_docs"
DEFAULT_INTERIM_DIR = "data/interim"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_OUTPUTS_DIR = "data/outputs"

DEFAULT_MESH_PARSED_FILE = "data/interim/mesh_parsed.jsonl"
DEFAULT_DOC_EMBEDDINGS_FILE = "data/interim/doc_embeddings.parquet"
DEFAULT_MESH_EMBEDDINGS_FILE = "data/interim/mesh_embeddings.parquet"
DEFAULT_ENTITIES_DETECTED_FILE = "data/processed/entities_detected.jsonl"
DEFAULT_CANDIDATES_FILE = "data/processed/candidates.jsonl"
DEFAULT_EXPORT_CANDIDATES_CSV = "data/outputs/export_candidates.csv"
DEFAULT_EXPORT_CANDIDATES_VALIDATED_CSV = "data/outputs/export_candidates_validated.csv"
DEFAULT_MESH_EXTENDED_JSON = "data/outputs/mesh_extended.json"
DEFAULT_REPORT_DIFF_MD = "data/outputs/report_diff.md"

SUPPORTED_INPUT_EXTENSIONS = (".txt", ".csv", ".json", ".jsonl", ".md")
SUPPORTED_EXPORT_EXTENSIONS = (".csv", ".json", ".jsonl", ".md", ".parquet")

def _read_json_secret(secret_file: Path) -> dict[str, Any]:
    """
        Read a JSON secret file safely

        Args:
            secret_file: Path to JSON file

        Returns:
            Parsed dict
    """

    if not secret_file.exists():
        return {}

    try:
        return json.loads(secret_file.read_text(encoding=DEFAULT_ENCODING))
    except Exception:
        return {}
        
## ============================================================
## CONFIG MODELS
## ============================================================
@dataclass(frozen=True)
class ExecutionMetadata:
    """
        Execution metadata

        Args:
            run_id: Unique runtime identifier
            started_at_utc: UTC timestamp when config was built
            hostname: Current host name
            platform_name: Current operating system name
            profile: Active runtime profile
            environment: Active environment
    """

    run_id: str
    started_at_utc: str
    hostname: str
    platform_name: str
    profile: str
    environment: str

@dataclass(frozen=True)
class RuntimeConfig:
    """
        Runtime configuration

        Args:
            environment: Environment name
            profile: Active runtime profile
            debug: Whether debug mode is enabled
            log_level: Logging level
            create_dirs_on_start: Whether runtime directories are created automatically
            enable_api_auth: Whether API key auth is enabled
            max_workers: Maximum worker count
            batch_size: Generic batch size
            batch_sleep_seconds: Sleep delay between batches
            request_timeout_seconds: Request timeout
            allowed_origins: Allowed HTTP origins for future API usage
    """

    environment: str
    profile: str
    debug: bool
    log_level: str
    create_dirs_on_start: bool
    enable_api_auth: bool
    max_workers: int
    batch_size: int
    batch_sleep_seconds: float
    request_timeout_seconds: int
    allowed_origins: list[str]

@dataclass(frozen=True)
class PipelineConfig:
    """
        Pipeline configuration

        Args:
            embedding_model_name: Embedding model name
            top_k_candidates: Top-k candidate matches
            similarity_threshold: Similarity threshold
            use_cache: Whether cache is enabled
            export_validated_only: Whether only validated candidates are exported
    """

    embedding_model_name: str
    top_k_candidates: int
    similarity_threshold: float
    use_cache: bool
    export_validated_only: bool

@dataclass(frozen=True)
class PathsConfig:
    """
        Filesystem paths configuration

        Args:
            project_root: Project root directory
            src_dir: Source directory
            data_dir: Data root directory
            logs_dir: Logs directory
            artifacts_dir: Artifacts root directory
            secrets_dir: Secrets directory
            raw_mesh_dir: Raw MeSH directory
            raw_medical_docs_dir: Raw medical documents directory
            interim_dir: Interim directory
            processed_dir: Processed directory
            outputs_dir: Outputs directory
            mesh_parsed_file: Parsed MeSH jsonl file
            doc_embeddings_file: Document embeddings parquet file
            mesh_embeddings_file: MeSH embeddings parquet file
            entities_detected_file: Detected entities jsonl file
            candidates_file: Candidate mappings jsonl file
            export_candidates_csv: Exported candidates CSV
            export_candidates_validated_csv: Exported validated candidates CSV
            mesh_extended_json: Extended MeSH JSON file
            report_diff_md: Markdown diff report
    """

    project_root: Path
    src_dir: Path
    data_dir: Path
    logs_dir: Path
    artifacts_dir: Path
    secrets_dir: Path
    raw_mesh_dir: Path
    raw_medical_docs_dir: Path
    interim_dir: Path
    processed_dir: Path
    outputs_dir: Path
    mesh_parsed_file: Path
    doc_embeddings_file: Path
    mesh_embeddings_file: Path
    entities_detected_file: Path
    candidates_file: Path
    export_candidates_csv: Path
    export_candidates_validated_csv: Path
    mesh_extended_json: Path
    report_diff_md: Path

@dataclass(frozen=True)
class SecretsConfig:
    """
        Secret values resolved from env or files

        Args:
            api_key: Optional application API key
            huggingface_token: Optional Hugging Face token
    """

    api_key: str
    huggingface_token: str

@dataclass(frozen=True)
class AppConfig:
    """
        Unified application configuration

        Args:
            app_name: Application name
            app_version: Application version
            execution: Execution metadata
            runtime: Runtime configuration
            pipeline: Pipeline configuration
            paths: Filesystem paths configuration
            secrets: Secret values
    """

    app_name: str
    app_version: str
    execution: ExecutionMetadata
    runtime: RuntimeConfig
    pipeline: PipelineConfig
    paths: PathsConfig
    secrets: SecretsConfig

## ============================================================
## DOTENV / ENV HELPERS
## ============================================================
def _resolve_project_root() -> Path:
    """
        Resolve the project root path

        Returns:
            Absolute project root path
    """

    ## Prefer explicit project root override when available
    project_root_raw = os.getenv("PROJECT_ROOT", "").strip()
    return Path(project_root_raw).expanduser().resolve() if project_root_raw else Path(__file__).resolve().parents[2]

def get_project_root() -> Path:
    """
        Backward-compatible project root resolver

        Returns:
            Absolute project root path
    """

    ## Keep compatibility with previous imports
    return _resolve_project_root()

def _load_dotenv_if_present(env_path: Optional[Path] = None) -> None:
    """
        Load a local .env file if available

        Args:
            env_path: Optional .env path override

        Returns:
            None
    """

    ## Use explicit env path or project-level .env
    dot_env_path = env_path or (_resolve_project_root() / ".env")
    if not dot_env_path.exists():
        return

    ## Parse simple KEY=VALUE lines without overriding process env
    for line in dot_env_path.read_text(encoding=DEFAULT_ENCODING, errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

def load_dotenv_if_present(env_path: Optional[Path] = None) -> None:
    """
        Backward-compatible dotenv loader

        Args:
            env_path: Optional .env path override

        Returns:
            None
    """

    ## Keep compatibility with previous imports
    _load_dotenv_if_present(env_path)

def _is_placeholder(value: str) -> bool:
    """
        Detect placeholder-like values

        Args:
            value: Raw environment value

        Returns:
            True if the value looks like a placeholder
    """

    ## Normalize before inspection
    normalized = value.strip().upper()
    return any(token in normalized for token in PLACEHOLDER_PREFIXES)

def _get_env(name: str, default: Optional[str] = None) -> str:
    """
        Read an environment variable safely

        Args:
            name: Environment variable name
            default: Optional default value

        Returns:
            Normalized environment value

        Raises:
            ConfigurationError: If missing and no default provided
    """

    ## Read raw value from environment
    value = os.getenv(name)
    if value is None:
        if default is None:
            raise ConfigurationError(f"Missing environment variable: {name}")
        return default
    return value.strip()

def _get_env_bool(name: str, default: bool = False) -> bool:
    """
        Parse a boolean environment variable

        Args:
            name: Environment variable name
            default: Default boolean value

        Returns:
            Parsed boolean value

        Raises:
            ConfigurationError: If invalid
    """

    ## Normalize boolean representation
    raw = _get_env(name, str(default)).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigurationError(f"Invalid boolean value for {name}: {raw}")

def _get_env_int(name: str, default: int) -> int:
    """
        Parse an integer environment variable

        Args:
            name: Environment variable name
            default: Default integer value

        Returns:
            Parsed integer value

        Raises:
            ConfigurationError: If invalid
    """

    ## Parse integer strictly
    try:
        return int(_get_env(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc

def _get_env_float(name: str, default: float) -> float:
    """
        Parse a float environment variable

        Args:
            name: Environment variable name
            default: Default float value

        Returns:
            Parsed float value

        Raises:
            ConfigurationError: If invalid
    """

    ## Parse float strictly
    try:
        return float(_get_env(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a float") from exc

def _get_env_list(name: str, default: Optional[list[str]] = None, *, separator: str = ",") -> list[str]:
    """
        Parse a list-like environment variable

        Args:
            name: Environment variable name
            default: Default fallback list
            separator: Raw value separator

        Returns:
            Parsed list of strings
    """

    ## Read raw list value
    raw = _get_env(name, "")
    if not raw:
        return list(default or [])
    return [item.strip() for item in raw.split(separator) if item.strip()]

def _expand_env_vars(value: str) -> str:
    """
        Expand shell variables and user home in a string

        Args:
            value: Raw string value

        Returns:
            Expanded string
    """

    ## Expand shell variables
    return os.path.expandvars(value)

def _resolve_path(path_value: str | Path, project_root: Path) -> Path:
    """
        Resolve a path against the project root

        Args:
            path_value: Raw path value
            project_root: Project root directory

        Returns:
            Absolute resolved path
    """

    ## Expand shell variables and user home
    path_obj = Path(_expand_env_vars(str(path_value))).expanduser()
    return path_obj.resolve() if path_obj.is_absolute() else (project_root / path_obj).resolve()

def _get_env_path(name: str, default: str, project_root: Path) -> Path:
    """
        Read and resolve a path environment variable

        Args:
            name: Environment variable name
            default: Default path value
            project_root: Project root directory

        Returns:
            Resolved path
    """

    ## Resolve env override or default path
    return _resolve_path(_get_env(name, default), project_root)

def _read_secret_value(direct_key: str, file_key: str, *, project_root: Path, default: str = "") -> str:
    """
        Read a secret from env directly or from a file path

        Args:
            direct_key: Environment variable containing the secret
            file_key: Environment variable containing the secret file path
            project_root: Project root directory
            default: Default fallback value

        Returns:
            Secret value or default
    """

    ## Prefer direct env secret value first
    direct_value = _get_env(direct_key, default)
    if direct_value and not _is_placeholder(direct_value):
        return direct_value

    ## Fallback to file-based secret
    secret_file_raw = _get_env(file_key, "")
    if not secret_file_raw:
        return default

    ## Resolve and read secret file when available
    secret_file = _resolve_path(secret_file_raw, project_root)
    if secret_file.exists() and secret_file.is_file():
        return secret_file.read_text(encoding=DEFAULT_ENCODING).strip()
    return default

## ============================================================
## PROFILE HELPERS
## ============================================================
def _get_profiled_env(name: str, default: str, profile: str) -> str:
    """
        Read an env value with optional profile override

        Args:
            name: Base environment variable name
            default: Default fallback value
            profile: Active runtime profile

        Returns:
            Resolved string value
    """

    ## Prefer profile-specific override when present
    override_key = f"{profile.upper()}_{name}"
    return _get_env(override_key, default) if os.getenv(override_key) is not None else _get_env(name, default)

def _get_profiled_env_bool(name: str, default: bool, profile: str) -> bool:
    """
        Read a boolean env value with optional profile override

        Args:
            name: Base environment variable name
            default: Default fallback value
            profile: Active runtime profile

        Returns:
            Parsed boolean value
    """

    ## Prefer profile-specific override when present
    override_key = f"{profile.upper()}_{name}"
    return _get_env_bool(override_key, default) if os.getenv(override_key) is not None else _get_env_bool(name, default)

def _get_profiled_env_int(name: str, default: int, profile: str) -> int:
    """
        Read an integer env value with optional profile override

        Args:
            name: Base environment variable name
            default: Default fallback value
            profile: Active runtime profile

        Returns:
            Parsed integer value
    """

    ## Prefer profile-specific override when present
    override_key = f"{profile.upper()}_{name}"
    return _get_env_int(override_key, default) if os.getenv(override_key) is not None else _get_env_int(name, default)

def _get_profiled_env_float(name: str, default: float, profile: str) -> float:
    """
        Read a float env value with optional profile override

        Args:
            name: Base environment variable name
            default: Default fallback value
            profile: Active runtime profile

        Returns:
            Parsed float value
    """

    ## Prefer profile-specific override when present
    override_key = f"{profile.upper()}_{name}"
    return _get_env_float(override_key, default) if os.getenv(override_key) is not None else _get_env_float(name, default)

## ============================================================
## VALIDATION / BUILD HELPERS
## ============================================================
def _validate_required_placeholders(keys: list[str]) -> None:
    """
        Validate that required env keys are not unresolved placeholders

        Args:
            keys: Environment keys to inspect

        Returns:
            None

        Raises:
            ConfigurationError: If placeholders are detected
    """

    ## Collect invalid placeholder keys
    invalid_keys = [key for key in keys if (value := _get_env(key, "")) and _is_placeholder(value)]
    if invalid_keys:
        raise ConfigurationError("Placeholder values detected for: " + ", ".join(invalid_keys))

def _validate_positive_int(value: int, field_name: str) -> None:
    """
        Validate that an integer is strictly positive

        Args:
            value: Value to validate
            field_name: Human-readable field name

        Returns:
            None
        """

    ## Reject non-positive integers
    if value <= 0:
        raise ConfigurationError(f"{field_name} must be > 0. Got: {value}")

def _validate_non_negative_float(value: float, field_name: str) -> None:
    """
        Validate that a float is non-negative

        Args:
            value: Value to validate
            field_name: Human-readable field name

        Returns:
            None
        """

    ## Reject negative floats
    if value < 0.0:
        raise ConfigurationError(f"{field_name} must be >= 0. Got: {value}")

def _validate_probability(value: float, field_name: str) -> None:
    """
        Validate that a float is inside [0, 1]

        Args:
            value: Value to validate
            field_name: Human-readable field name

        Returns:
            None
        """

    ## Reject invalid probability values
    if not 0.0 <= value <= 1.0:
        raise ConfigurationError(f"{field_name} must be in [0, 1]. Got: {value}")

def _ensure_directories_exist(paths: list[Path]) -> None:
    """
        Ensure required runtime directories exist

        Args:
            paths: Directories to create if missing

        Returns:
            None
    """

    ## Create all runtime directories
    for directory in paths:
        directory.mkdir(parents=True, exist_ok=True)

def _validate_config(config: AppConfig) -> None:
    """
        Validate the final structured configuration

        Args:
            config: Structured configuration

        Returns:
            None
        """

    ## Validate runtime numeric parameters
    _validate_positive_int(config.runtime.max_workers, "MAX_WORKERS")
    _validate_positive_int(config.runtime.batch_size, "BATCH_SIZE")
    _validate_positive_int(config.runtime.request_timeout_seconds, "REQUEST_TIMEOUT_SECONDS")
    _validate_non_negative_float(config.runtime.batch_sleep_seconds, "BATCH_SLEEP_SECONDS")

    ## Validate pipeline parameters
    _validate_positive_int(config.pipeline.top_k_candidates, "TOP_K_CANDIDATES")
    _validate_probability(config.pipeline.similarity_threshold, "SIMILARITY_THRESHOLD")

    ## Validate output suffixes
    if config.paths.export_candidates_csv.suffix.lower() != ".csv":
        raise ConfigurationError("EXPORT_CANDIDATES_CSV must point to a .csv file")
    if config.paths.export_candidates_validated_csv.suffix.lower() != ".csv":
        raise ConfigurationError("EXPORT_CANDIDATES_VALIDATED_CSV must point to a .csv file")
    if config.paths.mesh_extended_json.suffix.lower() != ".json":
        raise ConfigurationError("MESH_EXTENDED_JSON must point to a .json file")
    if config.paths.report_diff_md.suffix.lower() != ".md":
        raise ConfigurationError("REPORT_DIFF_MD must point to a .md file")

## ============================================================
## DIRECTORY HELPERS
## ============================================================
def ensure_directories(settings: AppConfig) -> None:
    """
        Ensure all required directories exist

        Args:
            settings: Loaded application settings

        Returns:
            None
    """

    ## Create the standard runtime directories
    _ensure_directories_exist([
        settings.paths.data_dir,
        settings.paths.logs_dir,
        settings.paths.raw_mesh_dir,
        settings.paths.raw_medical_docs_dir,
        settings.paths.interim_dir,
        settings.paths.processed_dir,
        settings.paths.outputs_dir,
        settings.paths.artifacts_dir,
        settings.paths.secrets_dir,
        settings.paths.mesh_parsed_file.parent,
        settings.paths.doc_embeddings_file.parent,
        settings.paths.mesh_embeddings_file.parent,
        settings.paths.entities_detected_file.parent,
        settings.paths.candidates_file.parent,
        settings.paths.export_candidates_csv.parent,
    ])

## ============================================================
## EXPORT HELPERS
## ============================================================
def config_to_dict(config: AppConfig) -> dict[str, Any]:
    """
        Convert AppConfig into a serializable dictionary

        Args:
            config: Structured configuration object

        Returns:
            Serializable dictionary
    """

    ## Convert dataclass tree into a plain dictionary
    payload = asdict(config)

    ## Normalize Path objects recursively
    def _normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: _normalize(val) for key, val in value.items()}
        if isinstance(value, list):
            return [_normalize(item) for item in value]
        return value

    return _normalize(payload)

def config_to_json(config: AppConfig) -> str:
    """
        Convert AppConfig into a JSON string

        Args:
            config: Structured configuration object

        Returns:
            JSON string
    """

    ## Serialize normalized configuration to JSON
    return json.dumps(config_to_dict(config), indent=2, ensure_ascii=False)

## ============================================================
## CONFIG FACTORY
## ============================================================
@lru_cache(maxsize=1)
def get_config(env_path: Optional[Path] = None) -> AppConfig:
    """
        Build full application configuration from environment variables

        High-level workflow:
            1) Load optional project-level .env
            2) Resolve project root and active profile
            3) Build execution, runtime, pipeline, paths and secrets sections
            4) Validate and optionally create runtime directories
            5) Cache the final AppConfig

        Args:
            env_path: Optional .env path override

        Returns:
            AppConfig instance
    """

    ## Load optional local .env file first
    _load_dotenv_if_present(env_path)

    ## Resolve project root and active runtime profile
    project_root = _resolve_project_root()
    environment = _get_env("ENVIRONMENT", DEFAULT_ENVIRONMENT).lower()
    profile = _get_env("PROFILE", DEFAULT_PROFILE).lower()

    ## Validate placeholder values where relevant
    _validate_required_placeholders(["ENVIRONMENT", "PROFILE", "API_KEY", "HUGGINGFACE_TOKEN"])

    ## Build execution metadata
    execution = ExecutionMetadata(
        run_id=_get_env("RUN_ID", str(uuid.uuid4())),
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        hostname=platform.node(),
        platform_name=SYSTEM_NAME,
        profile=profile,
        environment=environment,
    )

    ## Build runtime section
    runtime = RuntimeConfig(
        environment=environment,
        profile=profile,
        debug=_get_profiled_env_bool("DEBUG", environment == "dev", profile),
        log_level=_get_profiled_env("LOG_LEVEL", "INFO", profile),
        create_dirs_on_start=_get_profiled_env_bool("CREATE_DIRS_ON_START", True, profile),
        enable_api_auth=_get_profiled_env_bool("ENABLE_API_AUTH", False, profile),
        max_workers=_get_profiled_env_int("MAX_WORKERS", 4, profile),
        batch_size=_get_profiled_env_int("BATCH_SIZE", 32, profile),
        batch_sleep_seconds=_get_profiled_env_float("BATCH_SLEEP_SECONDS", 0.0, profile),
        request_timeout_seconds=_get_profiled_env_int("REQUEST_TIMEOUT_SECONDS", 120, profile),
        allowed_origins=_get_env_list("ALLOWED_ORIGINS", ["*"]),
    )

    ## Build pipeline section
    pipeline = PipelineConfig(
        embedding_model_name=_get_profiled_env(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            profile,
        ),
        top_k_candidates=_get_profiled_env_int("TOP_K_CANDIDATES", 10, profile),
        similarity_threshold=_get_profiled_env_float("SIMILARITY_THRESHOLD", 0.65, profile),
        use_cache=_get_profiled_env_bool("USE_CACHE", True, profile),
        export_validated_only=_get_profiled_env_bool("EXPORT_VALIDATED_ONLY", False, profile),
    )

    ## Build paths section
    paths = PathsConfig(
        project_root=project_root,
        src_dir=(project_root / "src").resolve(),
        data_dir=_get_env_path("DATA_DIR", DEFAULT_DATA_DIR, project_root),
        logs_dir=_get_env_path("LOGS_DIR", DEFAULT_LOGS_DIR, project_root),
        artifacts_dir=_get_env_path("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR, project_root),
        secrets_dir=_get_env_path("SECRETS_DIR", DEFAULT_SECRETS_DIR, project_root),
        raw_mesh_dir=_get_env_path("RAW_MESH_DIR", DEFAULT_RAW_MESH_DIR, project_root),
        raw_medical_docs_dir=_get_env_path("RAW_MEDICAL_DOCS_DIR", DEFAULT_RAW_MEDICAL_DOCS_DIR, project_root),
        interim_dir=_get_env_path("INTERIM_DIR", DEFAULT_INTERIM_DIR, project_root),
        processed_dir=_get_env_path("PROCESSED_DIR", DEFAULT_PROCESSED_DIR, project_root),
        outputs_dir=_get_env_path("OUTPUTS_DIR", DEFAULT_OUTPUTS_DIR, project_root),
        mesh_parsed_file=_get_env_path("MESH_PARSED_FILE", DEFAULT_MESH_PARSED_FILE, project_root),
        doc_embeddings_file=_get_env_path("DOC_EMBEDDINGS_FILE", DEFAULT_DOC_EMBEDDINGS_FILE, project_root),
        mesh_embeddings_file=_get_env_path("MESH_EMBEDDINGS_FILE", DEFAULT_MESH_EMBEDDINGS_FILE, project_root),
        entities_detected_file=_get_env_path("ENTITIES_DETECTED_FILE", DEFAULT_ENTITIES_DETECTED_FILE, project_root),
        candidates_file=_get_env_path("CANDIDATES_FILE", DEFAULT_CANDIDATES_FILE, project_root),
        export_candidates_csv=_get_env_path("EXPORT_CANDIDATES_CSV", DEFAULT_EXPORT_CANDIDATES_CSV, project_root),
        export_candidates_validated_csv=_get_env_path("EXPORT_CANDIDATES_VALIDATED_CSV", DEFAULT_EXPORT_CANDIDATES_VALIDATED_CSV, project_root),
        mesh_extended_json=_get_env_path("MESH_EXTENDED_JSON", DEFAULT_MESH_EXTENDED_JSON, project_root),
        report_diff_md=_get_env_path("REPORT_DIFF_MD", DEFAULT_REPORT_DIFF_MD, project_root),
    )

    ## Resolve optional JSON secrets
    secrets_path = _get_env_path("APP_SECRETS_FILE", "", project_root)

    app_json = _read_json_secret(secrets_path) if secrets_path else {}

    secrets = SecretsConfig(
        api_key=app_json.get("api_key", ""),
        huggingface_token=app_json.get("huggingface_token", ""),
    )

    ## Build final config
    config = AppConfig(
        app_name=_get_env("APP_NAME", DEFAULT_APP_NAME),
        app_version=_get_env("APP_VERSION", DEFAULT_APP_VERSION),
        execution=execution,
        runtime=runtime,
        pipeline=pipeline,
        paths=paths,
        secrets=secrets,
    )

    ## Validate final configuration
    _validate_config(config)

    ## Create directories if requested
    if config.runtime.create_dirs_on_start:
        ensure_directories(config)

    return config

def get_settings(env_path: Optional[Path] = None) -> AppConfig:
    """
        Backward-compatible settings factory

        Args:
            env_path: Optional .env path override

        Returns:
            AppConfig instance
    """

    ## Keep compatibility with previous imports
    return get_config(env_path)

def load_config(env_path: Optional[Path] = None) -> AppConfig:
    """
        Backward-compatible alias for configuration loading

        Args:
            env_path: Optional .env path override

        Returns:
            AppConfig instance
    """

    ## Keep compatibility with existing imports
    return get_config(env_path)

def build_config(env_path: Optional[Path] = None) -> AppConfig:
    """
        Backward-compatible config builder

        Args:
            env_path: Optional .env path override

        Returns:
            AppConfig instance
    """

    ## Preserve an additional public entrypoint
    return get_config(env_path)

## ============================================================
## PUBLIC UTILS
## ============================================================
def get_data_paths(settings: AppConfig) -> dict[str, Path]:
    """
        Convenience helper returning main data paths

        Args:
            settings: Settings instance

        Returns:
            Named paths used by pipelines and services
    """

    ## Expose well-known path aliases
    return {
        "project_root": settings.paths.project_root,
        "data_dir": settings.paths.data_dir,
        "logs_dir": settings.paths.logs_dir,
        "raw_mesh_dir": settings.paths.raw_mesh_dir,
        "raw_medical_docs_dir": settings.paths.raw_medical_docs_dir,
        "mesh_parsed_file": settings.paths.mesh_parsed_file,
        "doc_embeddings_file": settings.paths.doc_embeddings_file,
        "mesh_embeddings_file": settings.paths.mesh_embeddings_file,
        "entities_detected_file": settings.paths.entities_detected_file,
        "candidates_file": settings.paths.candidates_file,
        "export_candidates_csv": settings.paths.export_candidates_csv,
        "export_candidates_validated_csv": settings.paths.export_candidates_validated_csv,
        "mesh_extended_json": settings.paths.mesh_extended_json,
        "report_diff_md": settings.paths.report_diff_md,
    }

## ============================================================
## PUBLIC SINGLETONS
## ============================================================
CONFIG: AppConfig = get_config()
config = CONFIG