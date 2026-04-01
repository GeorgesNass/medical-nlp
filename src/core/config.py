'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unified configuration loader for clinical-ner: dotenv, env parsing, paths, runtime flags, models, dictionaries, profiles, secrets and execution metadata."
'''

from __future__ import annotations

import json
import os
import platform
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Tuple

from src.core.entities import EntityLabel, TEMPORALITY_LABELS_MEDICATION, TEMPORALITY_LABELS_PATHOLOGY
from src.core.errors import ConfigurationError
from src.utils.logging_utils import get_logger
from src.utils.utils import ensure_str, ensure_str_or_none

logger = get_logger(__name__)

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
DEFAULT_APP_NAME = "clinical-ner"
DEFAULT_APP_VERSION = "1.0.0"
DEFAULT_ENVIRONMENT = "dev"
DEFAULT_PROFILE = "cpu"

DEFAULT_DATA_DIRNAME = "data"
DEFAULT_ARTIFACTS_DIRNAME = "artifacts"
DEFAULT_LOGS_DIRNAME = "logs"
DEFAULT_SECRETS_DIRNAME = "secrets"

DEFAULT_RAW_DIRNAME = "raw"
DEFAULT_ANNOTATED_DIRNAME = "annotated"
DEFAULT_INTERIM_DIRNAME = "interim"
DEFAULT_PROCESSED_DIRNAME = "processed"

DEFAULT_MODELS_DIRNAME = "models"
DEFAULT_REPORTS_DIRNAME = "reports"
DEFAULT_EXPORTS_DIRNAME = "exports"
DEFAULT_DICTIONARIES_DIRNAME = "dictionaries"

SUPPORTED_NEGATION_STRATEGIES = ("rules", "model")
SUPPORTED_TEMPORALITY_STRATEGIES = ("rules", "model")
SUPPORTED_INPUT_EXTENSIONS = (".txt", ".csv", ".json", ".ann")

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
class PathsConfig:
    """
        Centralized filesystem paths for the Clinical NER project

        Args:
            project_root: Root directory of the project
            src_dir: Source directory
            data_dir: Root data directory
            data_raw_dir: Folder for raw input data
            data_annotated_dir: Folder for labeled datasets
            data_interim_dir: Folder for intermediate artifacts
            data_processed_dir: Folder for processed datasets
            artifacts_dir: Root artifact directory
            artifacts_models_dir: Folder for trained models
            artifacts_reports_dir: Folder for reports and metrics
            artifacts_exports_dir: Folder for CSV/JSON exports
            artifacts_dictionaries_dir: Folder for dictionaries
            logs_dir: Folder for application logs
            secrets_dir: Folder for file-based secrets
    """

    project_root: Path
    src_dir: Path
    data_dir: Path
    data_raw_dir: Path
    data_annotated_dir: Path
    data_interim_dir: Path
    data_processed_dir: Path
    artifacts_dir: Path
    artifacts_models_dir: Path
    artifacts_reports_dir: Path
    artifacts_exports_dir: Path
    artifacts_dictionaries_dir: Path
    logs_dir: Path
    secrets_dir: Path

@dataclass(frozen=True)
class RuntimeConfig:
    """
        Runtime configuration flags for pipeline execution

        Args:
            environment: Environment name
            profile: Active runtime profile
            debug: Whether debug mode is enabled
            log_level: Logging level
            accept_labeled_data: Allow labeled CSV input
            accept_unlabeled_texts: Allow raw .txt input
            enable_negation: Enable negation detection
            enable_temporality: Enable temporality detection
            device_mode: Requested execution device mode
            resolved_device: Final execution device
            seed: Global random seed
            max_docs: Optional cap on number of processed documents
            max_entities_per_record: Safety limit per document
            dictionary_min_confidence: Threshold for dictionary matching
            model_min_confidence: Threshold for model predictions
            batch_size: Generic batch size for processing
            max_workers: Maximum worker count
            batch_sleep_seconds: Sleep delay between batches
            request_timeout_seconds: Timeout for optional external calls
            allowed_origins: Allowed HTTP origins for future API usage
            ner_labels: NER labels supported by the application
            temporality_labels_medication: Temporality-compatible medication labels
            temporality_labels_pathology: Temporality-compatible pathology labels
    """

    environment: str
    profile: str
    debug: bool
    log_level: str
    accept_labeled_data: bool
    accept_unlabeled_texts: bool
    enable_negation: bool
    enable_temporality: bool
    device_mode: str
    resolved_device: str
    seed: int
    max_docs: int | None
    max_entities_per_record: int
    dictionary_min_confidence: float
    model_min_confidence: float
    batch_size: int
    max_workers: int
    batch_sleep_seconds: float
    request_timeout_seconds: int
    allowed_origins: list[str]
    ner_labels: tuple[EntityLabel, ...] = field(default_factory=lambda: (
        EntityLabel.MEDICATION,
        EntityLabel.DISEASE,
        EntityLabel.ALLERGY,
        EntityLabel.PROCEDURE,
        EntityLabel.TEST,
        EntityLabel.ANATOMY,
    ))
    temporality_labels_medication: tuple[EntityLabel, ...] = field(default_factory=lambda: TEMPORALITY_LABELS_MEDICATION)
    temporality_labels_pathology: tuple[EntityLabel, ...] = field(default_factory=lambda: TEMPORALITY_LABELS_PATHOLOGY)

@dataclass(frozen=True)
class ModelConfig:
    """
        Configuration for NER, negation and temporality models

        Args:
            hf_ner_model_name: HF model name for token classification
            spacy_model_name: spaCy model name or local path
            negation_strategy: rules or model
            temporality_strategy: rules or model
            max_length: Maximum token length
            learning_rate: Learning rate when fine-tuning is enabled
            num_epochs: Number of epochs when fine-tuning is enabled
    """

    hf_ner_model_name: str
    spacy_model_name: str | None
    negation_strategy: str
    temporality_strategy: str
    max_length: int
    learning_rate: float
    num_epochs: int

@dataclass(frozen=True)
class DictionaryConfig:
    """
        Configuration for dictionary-based auto-labeling

        Args:
            dictionaries_root: Root directory for dictionary files
            enable_fuzzy: Enable fuzzy matching
            fuzzy_max_distance: Maximum edit distance for fuzzy match
            enable_embeddings: Enable embedding-based matching
            embeddings_model_name: Sentence-transformers model name
            save_dictionary_matches: Whether dictionary matches are exported
    """

    dictionaries_root: Path
    enable_fuzzy: bool
    fuzzy_max_distance: int
    enable_embeddings: bool
    embeddings_model_name: str
    save_dictionary_matches: bool

@dataclass(frozen=True)
class SecretsConfig:
    """
        Secret values resolved from env or files

        Args:
            huggingface_token: Optional Hugging Face token
            api_key: Optional generic API key
    """

    huggingface_token: str
    api_key: str

@dataclass(frozen=True)
class AppConfig:
    """
        Full project configuration container

        Args:
            app_name: Application name
            app_version: Application version
            execution: Execution metadata
            paths: Centralized project paths
            runtime: Runtime flags and thresholds
            model: Model settings
            dictionaries: Dictionary settings
            secrets: Optional secrets
            extra: Additional project-specific metadata
    """

    app_name: str
    app_version: str
    execution: ExecutionMetadata
    paths: PathsConfig
    runtime: RuntimeConfig
    model: ModelConfig
    dictionaries: DictionaryConfig
    secrets: SecretsConfig
    extra: dict[str, Any] = field(default_factory=dict)

## ============================================================
## DOTENV / ENV HELPERS
## ============================================================
def _resolve_project_root() -> Path:
    """
        Resolve the project root directory

        Returns:
            Absolute project root path
    """

    ## Prefer explicit root override, then project-specific root, then cwd
    root = os.getenv("PROJECT_ROOT", "").strip() or ensure_str_or_none(os.getenv("CLINICAL_NER_ROOT")) or str(Path.cwd())
    return Path(root).expanduser().resolve()

def _load_dotenv_if_present() -> None:
    """
        Load a local .env file if available

        Returns:
            None
    """

    ## Import dotenv lazily to avoid hard dependency issues
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    ## Load only when a project-level .env file exists
    env_path = _resolve_project_root() / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)

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
        Read environment variable safely

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
            msg = f"Missing environment variable: {name}"
            logger.error(msg)
            raise ConfigurationError(msg)
        return default
    return value.strip()

def _get_env_bool(name: str, default: bool) -> bool:
    """
        Parse a boolean environment variable

        Args:
            name: Environment variable name
            default: Default boolean if value is None or empty

        Returns:
            Parsed boolean value

        Raises:
            ConfigurationError: If invalid
    """

    ## Normalize raw boolean value
    raw = ensure_str(_get_env(name, str(default))).strip().lower()
    if raw == "":
        return default
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    msg = f"Invalid boolean value for {name}: {raw}"
    logger.error(msg)
    raise ConfigurationError(msg)

def _get_env_int(name: str, default: int) -> int:
    """
        Parse integer from environment

        Args:
            name: Environment variable name
            default: Default integer if value is None or empty

        Returns:
            Parsed integer

        Raises:
            ConfigurationError: If conversion fails
    """

    ## Parse integer strictly
    raw = ensure_str(_get_env(name, str(default))).strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except Exception as exc:
        msg = f"Invalid integer value for {name}: {raw}"
        logger.error(msg)
        raise ConfigurationError(msg) from exc

def _get_env_float(name: str, default: float) -> float:
    """
        Parse float from environment

        Args:
            name: Environment variable name
            default: Default float if value is None or empty

        Returns:
            Parsed float

        Raises:
            ConfigurationError: If conversion fails
    """

    ## Parse float strictly
    raw = ensure_str(_get_env(name, str(default))).strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except Exception as exc:
        msg = f"Invalid float value for {name}: {raw}"
        logger.error(msg)
        raise ConfigurationError(msg) from exc

def _get_env_list(name: str, default: Optional[list[str]] = None, *, separator: str = ",") -> list[str]:
    """
        Parse a list-like environment variable

        Args:
            name: Environment variable name
            default: Default fallback list
            separator: Value separator

        Returns:
            Parsed list of strings
    """

    ## Read raw list value
    raw = ensure_str(_get_env(name, "")).strip()
    if raw == "":
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

    ## Expand shell variables such as %USERPROFILE% or $HOME
    return os.path.expandvars(value)

def _resolve_path(path_value: str | Path, project_root: Path) -> Path:
    """
        Resolve a path against the project root

        Args:
            path_value: Raw path value
            project_root: Project root directory

        Returns:
            Resolved absolute path
    """

    ## Expand shell variables and user home
    path_obj = Path(_expand_env_vars(str(path_value))).expanduser()
    return path_obj.resolve() if path_obj.is_absolute() else (project_root / path_obj).resolve()

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
def _build_paths(project_root: Path) -> PathsConfig:
    """
        Build centralized project paths from a root directory

        Args:
            project_root: Root directory of the project

        Returns:
            PathsConfig instance
    """

    ## Resolve and validate root existence early
    root = Path(project_root).expanduser().resolve()
    if not root.exists():
        msg = f"Project root does not exist: {root}"
        logger.error(msg)
        raise ConfigurationError(msg)

    ## Build conventional directory layout
    data_dir = root / DEFAULT_DATA_DIRNAME
    artifacts_dir = root / DEFAULT_ARTIFACTS_DIRNAME

    return PathsConfig(
        project_root=root,
        src_dir=root / "src",
        data_dir=data_dir,
        data_raw_dir=data_dir / DEFAULT_RAW_DIRNAME,
        data_annotated_dir=data_dir / DEFAULT_ANNOTATED_DIRNAME,
        data_interim_dir=data_dir / DEFAULT_INTERIM_DIRNAME,
        data_processed_dir=data_dir / DEFAULT_PROCESSED_DIRNAME,
        artifacts_dir=artifacts_dir,
        artifacts_models_dir=artifacts_dir / DEFAULT_MODELS_DIRNAME,
        artifacts_reports_dir=artifacts_dir / DEFAULT_REPORTS_DIRNAME,
        artifacts_exports_dir=artifacts_dir / DEFAULT_EXPORTS_DIRNAME,
        artifacts_dictionaries_dir=artifacts_dir / DEFAULT_DICTIONARIES_DIRNAME,
        logs_dir=root / DEFAULT_LOGS_DIRNAME,
        secrets_dir=root / DEFAULT_SECRETS_DIRNAME,
    )

def _ensure_directories_exist(paths: PathsConfig) -> None:
    """
        Ensure all standard project directories exist

        Args:
            paths: Centralized paths configuration

        Returns:
            None
    """

    ## Create runtime directories safely
    for path in (
        paths.data_dir,
        paths.data_raw_dir,
        paths.data_annotated_dir,
        paths.data_interim_dir,
        paths.data_processed_dir,
        paths.artifacts_dir,
        paths.artifacts_models_dir,
        paths.artifacts_reports_dir,
        paths.artifacts_exports_dir,
        paths.artifacts_dictionaries_dir,
        paths.logs_dir,
        paths.secrets_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

def _resolve_device(device_mode: str) -> str:
    """
        Resolve the execution device

        Args:
            device_mode: Requested device mode

        Returns:
            Final device string: cpu or cuda

        Raises:
            ConfigurationError: If device mode is invalid
    """

    ## Normalize device mode before resolution
    raw = ensure_str(device_mode).strip().lower()
    if raw in {"cpu", "cuda"}:
        return raw
    if raw != "auto":
        msg = f"Invalid device mode: {device_mode}"
        logger.error(msg)
        raise ConfigurationError(msg)

    ## Auto-detect cuda if torch is available
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def _validate_confidence(value: float, name: str) -> None:
    """
        Validate confidence thresholds

        Args:
            value: Threshold value
            name: Threshold name

        Returns:
            None

        Raises:
            ConfigurationError: If out of [0, 1]
    """

    ## Reject invalid confidence values
    if not 0.0 <= value <= 1.0:
        msg = f"{name} must be in [0, 1], got {value}"
        logger.error(msg)
        raise ConfigurationError(msg)

def _validate_positive_int(value: int, name: str) -> None:
    """
        Validate a strictly positive integer

        Args:
            value: Integer value
            name: Field name

        Returns:
            None

        Raises:
            ConfigurationError: If invalid
    """

    ## Reject non-positive values
    if value <= 0:
        msg = f"{name} must be > 0, got {value}"
        logger.error(msg)
        raise ConfigurationError(msg)

def _validate_non_negative_float(value: float, name: str) -> None:
    """
        Validate a non-negative float

        Args:
            value: Float value
            name: Field name

        Returns:
            None

        Raises:
            ConfigurationError: If invalid
    """

    ## Reject negative values
    if value < 0.0:
        msg = f"{name} must be >= 0, got {value}"
        logger.error(msg)
        raise ConfigurationError(msg)

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

    ## Collect placeholder-based invalid keys
    invalid_keys = [key for key in keys if (value := _get_env(key, "")) and _is_placeholder(value)]
    if invalid_keys:
        msg = "Placeholder values detected for: " + ", ".join(invalid_keys)
        logger.error(msg)
        raise ConfigurationError(msg)

def _validate_strategy(value: str, allowed: tuple[str, ...], field_name: str) -> str:
    """
        Validate a strategy value against allowed values

        Args:
            value: Raw strategy value
            allowed: Allowed values
            field_name: Human-readable field name

        Returns:
            Validated strategy value

        Raises:
            ConfigurationError: If unsupported
    """

    ## Normalize and validate configured strategy
    normalized = ensure_str(value).strip().lower()
    if normalized not in allowed:
        msg = f"{field_name} must be one of: {', '.join(allowed)}"
        logger.error(msg)
        raise ConfigurationError(msg)
    return normalized

def _validate_config(config: AppConfig) -> None:
    """
        Validate the final structured configuration

        Args:
            config: Structured project configuration

        Returns:
            None

        Raises:
            ConfigurationError: If config is invalid
    """

    ## Validate core runtime flags
    if not config.runtime.accept_labeled_data and not config.runtime.accept_unlabeled_texts:
        msg = "Both labeled and unlabeled inputs are disabled"
        logger.error(msg)
        raise ConfigurationError(msg)

    ## Validate numeric runtime values
    _validate_positive_int(config.runtime.seed, "CLINICAL_NER_SEED")
    _validate_positive_int(config.runtime.max_entities_per_record, "CLINICAL_NER_MAX_ENTITIES_PER_RECORD")
    _validate_positive_int(config.runtime.batch_size, "CLINICAL_NER_BATCH_SIZE")
    _validate_positive_int(config.runtime.max_workers, "CLINICAL_NER_MAX_WORKERS")
    _validate_positive_int(config.runtime.request_timeout_seconds, "CLINICAL_NER_REQUEST_TIMEOUT_SECONDS")
    _validate_non_negative_float(config.runtime.batch_sleep_seconds, "CLINICAL_NER_BATCH_SLEEP_SECONDS")

    ## Validate optional max docs
    if config.runtime.max_docs is not None and config.runtime.max_docs <= 0:
        msg = "CLINICAL_NER_MAX_DOCS must be positive when provided"
        logger.error(msg)
        raise ConfigurationError(msg)

    ## Validate confidence thresholds
    _validate_confidence(config.runtime.dictionary_min_confidence, "CLINICAL_NER_DICTIONARY_MIN_CONFIDENCE")
    _validate_confidence(config.runtime.model_min_confidence, "CLINICAL_NER_MODEL_MIN_CONFIDENCE")

    ## Validate strategies and model numeric values
    _validate_strategy(config.model.negation_strategy, SUPPORTED_NEGATION_STRATEGIES, "CLINICAL_NER_NEGATION_STRATEGY")
    _validate_strategy(config.model.temporality_strategy, SUPPORTED_TEMPORALITY_STRATEGIES, "CLINICAL_NER_TEMPORALITY_STRATEGY")
    _validate_positive_int(config.model.max_length, "CLINICAL_NER_MAX_LENGTH")
    _validate_positive_int(config.model.num_epochs, "CLINICAL_NER_NUM_EPOCHS")
    if config.model.learning_rate <= 0:
        msg = f"CLINICAL_NER_LEARNING_RATE must be > 0, got {config.model.learning_rate}"
        logger.error(msg)
        raise ConfigurationError(msg)

    ## Validate dictionary config
    if config.dictionaries.fuzzy_max_distance < 0:
        msg = "CLINICAL_NER_DICTIONARIES_FUZZY_MAX_DISTANCE must be >= 0"
        logger.error(msg)
        raise ConfigurationError(msg)

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

    ## Normalize Path and enum-like objects recursively
    def _normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [_normalize(item) for item in value]
        if isinstance(value, dict):
            return {key: _normalize(val) for key, val in value.items()}
        if isinstance(value, list):
            return [_normalize(item) for item in value]
        if hasattr(value, "value"):
            return getattr(value, "value")
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
def get_config(project_root: str | Path | None = None) -> AppConfig:
    """
        Build AppConfig from environment variables

        High-level workflow:
            1) Load optional local .env
            2) Resolve project root and runtime profile
            3) Build execution, paths, runtime, model and dictionaries
            4) Resolve secrets from env or files
            5) Validate and cache final configuration

        Args:
            project_root: Optional project root override

        Returns:
            AppConfig instance

        Raises:
            ConfigurationError: If config is invalid
    """

    ## Load optional local .env file first
    _load_dotenv_if_present()

    ## Resolve project root and active runtime profile
    root = Path(project_root).expanduser().resolve() if project_root is not None else _resolve_project_root()
    environment = _get_env("ENVIRONMENT", DEFAULT_ENVIRONMENT).lower()
    profile = _get_env("PROFILE", "gpu" if _get_env("CLINICAL_NER_DEVICE", "auto").lower() != "cpu" else DEFAULT_PROFILE).lower()

    ## Validate placeholder values where relevant
    _validate_required_placeholders(["ENVIRONMENT", "PROFILE", "HUGGINGFACE_TOKEN", "API_KEY"])

    ## Build execution metadata
    execution = ExecutionMetadata(
        run_id=_get_env("RUN_ID", str(uuid.uuid4())),
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        hostname=platform.node(),
        platform_name=SYSTEM_NAME,
        profile=profile,
        environment=environment,
    )

    ## Build and ensure centralized paths
    paths = _build_paths(root)
    _ensure_directories_exist(paths)

    ## Resolve max docs with nullable semantics
    max_docs_raw = ensure_str_or_none(os.getenv("CLINICAL_NER_MAX_DOCS"))
    max_docs = None if max_docs_raw in (None, "") else _get_env_int("CLINICAL_NER_MAX_DOCS", 0)

    ## Build runtime section
    device_mode = _get_profiled_env("CLINICAL_NER_DEVICE", "auto", profile)
    runtime = RuntimeConfig(
        environment=environment,
        profile=profile,
        debug=_get_profiled_env_bool("DEBUG", environment == "dev", profile),
        log_level=_get_profiled_env("LOG_LEVEL", "INFO", profile),
        accept_labeled_data=_get_profiled_env_bool("CLINICAL_NER_ACCEPT_LABELED_DATA", True, profile),
        accept_unlabeled_texts=_get_profiled_env_bool("CLINICAL_NER_ACCEPT_UNLABELED_TEXTS", True, profile),
        enable_negation=_get_profiled_env_bool("CLINICAL_NER_ENABLE_NEGATION", True, profile),
        enable_temporality=_get_profiled_env_bool("CLINICAL_NER_ENABLE_TEMPORALITY", True, profile),
        device_mode=device_mode,
        resolved_device=_resolve_device(device_mode),
        seed=_get_profiled_env_int("CLINICAL_NER_SEED", 42, profile),
        max_docs=max_docs,
        max_entities_per_record=_get_profiled_env_int("CLINICAL_NER_MAX_ENTITIES_PER_RECORD", 300, profile),
        dictionary_min_confidence=_get_profiled_env_float("CLINICAL_NER_DICTIONARY_MIN_CONFIDENCE", 0.7, profile),
        model_min_confidence=_get_profiled_env_float("CLINICAL_NER_MODEL_MIN_CONFIDENCE", 0.5, profile),
        batch_size=_get_profiled_env_int("CLINICAL_NER_BATCH_SIZE", 32, profile),
        max_workers=_get_profiled_env_int("CLINICAL_NER_MAX_WORKERS", 4, profile),
        batch_sleep_seconds=_get_profiled_env_float("CLINICAL_NER_BATCH_SLEEP_SECONDS", 0.0, profile),
        request_timeout_seconds=_get_profiled_env_int("CLINICAL_NER_REQUEST_TIMEOUT_SECONDS", 120, profile),
        allowed_origins=_get_env_list("ALLOWED_ORIGINS", ["*"]),
    )

    ## Build model section
    model = ModelConfig(
        hf_ner_model_name=_get_profiled_env("CLINICAL_NER_HF_NER_MODEL", "distilbert-base-multilingual-cased", profile),
        spacy_model_name=ensure_str_or_none(_get_profiled_env("CLINICAL_NER_SPACY_MODEL", "fr_core_news_md", profile)) or None,      
        negation_strategy=_validate_strategy(_get_profiled_env("CLINICAL_NER_NEGATION_STRATEGY", "rules", profile), SUPPORTED_NEGATION_STRATEGIES, "CLINICAL_NER_NEGATION_STRATEGY"),
        temporality_strategy=_validate_strategy(_get_profiled_env("CLINICAL_NER_TEMPORALITY_STRATEGY", "rules", profile), SUPPORTED_TEMPORALITY_STRATEGIES, "CLINICAL_NER_TEMPORALITY_STRATEGY"),
        max_length=_get_profiled_env_int("CLINICAL_NER_MAX_LENGTH", 512, profile),
        learning_rate=_get_profiled_env_float("CLINICAL_NER_LEARNING_RATE", 2e-5, profile),
        num_epochs=_get_profiled_env_int("CLINICAL_NER_NUM_EPOCHS", 3, profile),
    )

    ## Build dictionary section
    dictionaries = DictionaryConfig(
        dictionaries_root=paths.artifacts_dictionaries_dir,
        enable_fuzzy=_get_profiled_env_bool("CLINICAL_NER_DICTIONARIES_FUZZY", True, profile),
        fuzzy_max_distance=_get_profiled_env_int("CLINICAL_NER_DICTIONARIES_FUZZY_MAX_DISTANCE", 1, profile),
        enable_embeddings=_get_profiled_env_bool("CLINICAL_NER_DICTIONARIES_USE_EMBEDDINGS", False, profile),
        embeddings_model_name=_get_profiled_env(
            "CLINICAL_NER_DICTIONARIES_EMBEDDINGS_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            profile,
        ),
        save_dictionary_matches=_get_profiled_env_bool("CLINICAL_NER_SAVE_DICTIONARY_MATCHES", True, profile),
    )

    ## Resolve secrets from direct env or files
    secrets = SecretsConfig(
        huggingface_token=_read_secret_value("HUGGINGFACE_TOKEN", "HUGGINGFACE_TOKEN_FILE", project_root=root),
        api_key=_read_secret_value("API_KEY", "API_KEY_FILE", project_root=root),
    )

    ## Build final structured config
    config = AppConfig(
        app_name=_get_env("APP_NAME", DEFAULT_APP_NAME),
        app_version=_get_env("APP_VERSION", DEFAULT_APP_VERSION),
        execution=execution,
        paths=paths,
        runtime=runtime,
        model=model,
        dictionaries=dictionaries,
        secrets=secrets,
        extra={
            "supported_input_extensions": list(SUPPORTED_INPUT_EXTENSIONS),
            "system_name": SYSTEM_NAME,
            "is_windows": IS_WINDOWS,
            "is_linux": IS_LINUX,
            "is_macos": IS_MACOS,
        },
    )

    ## Validate final configuration
    _validate_config(config)

    ## Log concise configuration summary
    logger.info(
        "Configuration loaded | app=%s | env=%s | profile=%s | device=%s | negation=%s | temporality=%s | run_id=%s",
        config.app_name,
        config.runtime.environment,
        config.runtime.profile,
        config.runtime.resolved_device,
        config.runtime.enable_negation,
        config.runtime.enable_temporality,
        config.execution.run_id,
    )
    return config

def load_config(project_root: str | Path | None = None) -> AppConfig:
    """
        Backward-compatible alias for configuration loading

        Args:
            project_root: Optional project root override

        Returns:
            AppConfig instance
    """

    ## Keep compatibility with existing imports
    return get_config(project_root)

def build_config(project_root: str | Path | None = None) -> AppConfig:
    """
        Backward-compatible config builder

        Args:
            project_root: Optional project root override

        Returns:
            AppConfig instance
    """

    ## Preserve original public entrypoint
    return get_config(project_root)

## ============================================================
## PUBLIC SINGLETON CONFIG
## ============================================================
CONFIG: AppConfig = get_config()
config = CONFIG