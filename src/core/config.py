'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unified configuration loader for icd10-prediction: dotenv, env parsing, paths, profiles, model defaults, secrets and runtime metadata."
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
from typing import Any, Literal, Optional, Tuple

from src.core.errors import ConfigurationError, log_and_raise_missing_env
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
## TYPES
## ============================================================
UseGpuMode = Literal["auto", "true", "false"]
TaskType = Literal["multiclass", "multilabel"]

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
DEFAULT_APP_NAME = "icd10-prediction"
DEFAULT_APP_VERSION = "1.0.0"
DEFAULT_ENVIRONMENT = "dev"
DEFAULT_PROFILE = "cpu"

DEFAULT_DATA_DIR = "data"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_SECRETS_DIR = "secrets"

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_INTERIM_DIR = "data/interim"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_MODELS_DIR = "artifacts/models"
DEFAULT_REPORTS_DIR = "artifacts/reports"
DEFAULT_EXPORTS_DIR = "artifacts/exports"
DEFAULT_LABEL_MAPPING_PATH = "data/label_mapping.json"

SUPPORTED_INPUT_EXTENSIONS = (".csv", ".json", ".txt", ".xlsx", ".xls")
SUPPORTED_TASK_TYPES = ("multiclass", "multilabel")

def _read_json_secret(secret_file: Path) -> dict[str, Any]:
    """
        Read a JSON secret file safely

        Args:
            secret_file (Path): Path to the JSON file

        Returns:
            dict[str, Any]: Parsed JSON content or empty dict
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
class PathsConfig:
    """
        Filesystem paths configuration

        Args:
            project_root: Project root directory
            src_dir: Source directory
            data_dir: Main data directory
            raw_dir: Raw data directory
            interim_dir: Interim data directory
            processed_dir: Processed data directory
            artifacts_dir: Artifacts root directory
            models_dir: Trained models directory
            reports_dir: Reports directory
            exports_dir: Exports directory
            logs_dir: Logs directory
            secrets_dir: Secrets directory
            label_mapping_path: Label mapping file
    """

    project_root: Path
    src_dir: Path
    data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    models_dir: Path
    reports_dir: Path
    exports_dir: Path
    logs_dir: Path
    secrets_dir: Path
    label_mapping_path: Path

@dataclass(frozen=True)
class RuntimeConfig:
    """
        Runtime configuration

        Args:
            environment: Environment name
            profile: Active runtime profile
            debug: Whether debug mode is enabled
            log_level: Logging level
            use_gpu_mode: Raw GPU mode
            use_gpu: Final GPU decision
            random_seed: Random seed
            max_workers: Maximum worker count
            batch_size: Generic batch size
            batch_sleep_seconds: Sleep delay between batches
            request_timeout_seconds: Request timeout if external calls are used
            allowed_origins: Allowed origins for future API usage
    """

    environment: str
    profile: str
    debug: bool
    log_level: str
    use_gpu_mode: UseGpuMode
    use_gpu: bool
    random_seed: int
    max_workers: int
    batch_size: int
    batch_sleep_seconds: float
    request_timeout_seconds: int
    allowed_origins: list[str]

@dataclass(frozen=True)
class ModelConfig:
    """
        Model configuration

        Args:
            task_type: Prediction task type
            pretrained_model_name: Base model name
            max_length: Maximum token length
            learning_rate: Learning rate
            num_epochs: Number of epochs
            train_split: Train split ratio
            validation_split: Validation split ratio
            threshold: Decision threshold for multilabel setups
    """

    task_type: TaskType
    pretrained_model_name: str
    max_length: int
    learning_rate: float
    num_epochs: int
    train_split: float
    validation_split: float
    threshold: float

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
        Unified application configuration

        Args:
            app_name: Application name
            app_version: Application version
            execution: Execution metadata
            paths: Filesystem paths configuration
            runtime: Runtime configuration
            model: Model configuration
            secrets: Secret values
    """

    app_name: str
    app_version: str
    execution: ExecutionMetadata
    paths: PathsConfig
    runtime: RuntimeConfig
    model: ModelConfig
    secrets: SecretsConfig

## ============================================================
## DOTENV / ENV HELPERS
## ============================================================
def _resolve_project_root() -> Path:
    """
        Resolve project root directory

        Returns:
            Absolute project root path
    """

    ## Prefer explicit override first
    project_root_raw = os.getenv("PROJECT_ROOT", "").strip()
    return Path(project_root_raw).expanduser().resolve() if project_root_raw else Path(__file__).resolve().parents[2]

def _load_dotenv_if_present() -> None:
    """
        Load a local .env file if available

        Returns:
            None
    """

    ## Import dotenv lazily
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    ## Load project-level .env when present
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

    ## Fail when mandatory values are missing
    if value is None:
        if default is None:
            log_and_raise_missing_env([name])
        return default  # type: ignore[return-value]

    ## Normalize whitespace
    return value.strip()

def _get_env_bool(name: str, default: bool) -> bool:
    """
        Parse a boolean environment variable

        Args:
            name: Environment variable name
            default: Default fallback value

        Returns:
            Parsed boolean value

        Raises:
            ConfigurationError: If invalid
    """

    ## Parse normalized boolean values
    raw = _get_env(name, str(default)).lower()
    if raw in {"true", "1", "yes", "y", "on"}:
        return True
    if raw in {"false", "0", "no", "n", "off"}:
        return False
    raise ConfigurationError(f"Invalid boolean value for {name}: {raw}")

def _get_env_int(name: str, default: int) -> int:
    """
        Parse an integer environment variable

        Args:
            name: Environment variable name
            default: Default fallback value

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
            default: Default fallback value

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
            separator: Value separator

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

def _resolve_path(path_value: str, project_root: Path) -> Path:
    """
        Resolve a path against the project root

        Args:
            path_value: Raw path value
            project_root: Project root directory

        Returns:
            Resolved absolute path
    """

    ## Expand shell variables and user home
    path_obj = Path(_expand_env_vars(path_value)).expanduser()
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

    ## Prefer profile-specific override
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

    ## Prefer profile-specific override
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

    ## Prefer profile-specific override
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

    ## Prefer profile-specific override
    override_key = f"{profile.upper()}_{name}"
    return _get_env_float(override_key, default) if os.getenv(override_key) is not None else _get_env_float(name, default)

## ============================================================
## VALIDATION / BUILD HELPERS
## ============================================================
def _detect_gpu_requested(mode: UseGpuMode) -> bool:
    """
        Determine whether GPU usage is requested and available

        Args:
            mode: Raw GPU mode value

        Returns:
            Final GPU usage decision
    """

    ## Respect explicit override
    if mode == "true":
        return True
    if mode == "false":
        return False

    ## Auto mode falls back to torch detection
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())

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

    ## Collect invalid placeholder values
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

        Raises:
            ConfigurationError: If invalid
    """

    ## Reject non-positive integers
    if value <= 0:
        raise ConfigurationError(f"{field_name} must be > 0. Got: {value}")

def _validate_non_negative_int(value: int, field_name: str) -> None:
    """
        Validate that an integer is non-negative

        Args:
            value: Value to validate
            field_name: Human-readable field name

        Returns:
            None

        Raises:
            ConfigurationError: If invalid
    """

    ## Reject negative integers
    if value < 0:
        raise ConfigurationError(f"{field_name} must be >= 0. Got: {value}")

def _validate_probability(value: float, field_name: str) -> None:
    """
        Validate that a float is inside [0, 1]

        Args:
            value: Value to validate
            field_name: Human-readable field name

        Returns:
            None

        Raises:
            ConfigurationError: If invalid
    """

    ## Reject invalid probability values
    if not 0.0 <= value <= 1.0:
        raise ConfigurationError(f"{field_name} must be in [0, 1]. Got: {value}")

def _validate_task_type(value: str) -> TaskType:
    """
        Validate the configured task type

        Args:
            value: Raw task type

        Returns:
            Validated task type

        Raises:
            ConfigurationError: If unsupported
    """

    ## Restrict to supported task types
    if value not in SUPPORTED_TASK_TYPES:
        raise ConfigurationError("TASK_TYPE must be one of: multiclass, multilabel")
    return value  # type: ignore[return-value]

def _ensure_directories_exist(paths: list[Path]) -> None:
    """
        Ensure runtime directories exist

        Args:
            paths: Directories to create if missing

        Returns:
            None
    """

    ## Create runtime directories safely
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
    _validate_non_negative_int(config.runtime.random_seed, "RANDOM_SEED")
    _validate_positive_int(config.runtime.max_workers, "MAX_WORKERS")
    _validate_positive_int(config.runtime.batch_size, "BATCH_SIZE")
    _validate_positive_int(config.runtime.request_timeout_seconds, "REQUEST_TIMEOUT_SECONDS")
    if config.runtime.batch_sleep_seconds < 0:
        raise ConfigurationError(f"BATCH_SLEEP_SECONDS must be >= 0. Got: {config.runtime.batch_sleep_seconds}")

    ## Validate model numeric parameters
    _validate_positive_int(config.model.max_length, "MAX_LENGTH")
    _validate_positive_int(config.model.num_epochs, "NUM_EPOCHS")
    _validate_probability(config.model.train_split, "TRAIN_SPLIT")
    _validate_probability(config.model.validation_split, "VALIDATION_SPLIT")
    _validate_probability(config.model.threshold, "THRESHOLD")

    ## Validate split coherence
    if config.model.train_split + config.model.validation_split >= 1.0:
        raise ConfigurationError("TRAIN_SPLIT + VALIDATION_SPLIT must be < 1.0")

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

    ## Convert dataclass tree into a dictionary
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
def get_config() -> AppConfig:
    """
        Build full application configuration from environment variables

        High-level workflow:
            1) Load optional project-level .env
            2) Resolve project root and active profile
            3) Build execution, paths, runtime, model and secrets sections
            4) Validate and cache the final AppConfig

        Returns:
            AppConfig instance
    """

    ## Load optional local .env file first
    _load_dotenv_if_present()

    ## Resolve project root and runtime profile
    project_root = _resolve_project_root()
    environment = _get_env("ENVIRONMENT", DEFAULT_ENVIRONMENT).lower()
    profile = _get_env("PROFILE", "gpu" if _get_env("USE_GPU", "auto").lower() != "false" else DEFAULT_PROFILE).lower()

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

    ## Resolve root folders
    data_dir = _get_env_path("DATA_DIR", DEFAULT_DATA_DIR, project_root)
    logs_dir = _get_env_path("LOGS_DIR", DEFAULT_LOGS_DIR, project_root)
    artifacts_dir = _get_env_path("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR, project_root)
    secrets_dir = _get_env_path("SECRETS_DIR", DEFAULT_SECRETS_DIR, project_root)

    ## Build structured paths section
    paths = PathsConfig(
        project_root=project_root,
        src_dir=(project_root / "src").resolve(),
        data_dir=data_dir,
        raw_dir=_get_env_path("RAW_DIR", DEFAULT_RAW_DIR, project_root),
        interim_dir=_get_env_path("INTERIM_DIR", DEFAULT_INTERIM_DIR, project_root),
        processed_dir=_get_env_path("PROCESSED_DIR", DEFAULT_PROCESSED_DIR, project_root),
        artifacts_dir=artifacts_dir,
        models_dir=_get_env_path("MODELS_DIR", DEFAULT_MODELS_DIR, project_root),
        reports_dir=_get_env_path("REPORTS_DIR", DEFAULT_REPORTS_DIR, project_root),
        exports_dir=_get_env_path("EXPORTS_DIR", DEFAULT_EXPORTS_DIR, project_root),
        logs_dir=logs_dir,
        secrets_dir=secrets_dir,
        label_mapping_path=_get_env_path("LABEL_MAPPING_PATH", DEFAULT_LABEL_MAPPING_PATH, project_root),
    )

    ## Ensure runtime directories exist
    _ensure_directories_exist([
        paths.data_dir, paths.raw_dir, paths.interim_dir, paths.processed_dir,
        paths.artifacts_dir, paths.models_dir, paths.reports_dir, paths.exports_dir,
        paths.logs_dir, paths.secrets_dir,
    ])

    ## Resolve runtime section
    use_gpu_mode_raw = _get_profiled_env("USE_GPU", "auto", profile).lower()
    if use_gpu_mode_raw not in {"auto", "true", "false"}:
        raise ConfigurationError("USE_GPU must be auto|true|false")
    use_gpu_mode: UseGpuMode = use_gpu_mode_raw  # type: ignore[assignment]

    runtime = RuntimeConfig(
        environment=environment,
        profile=profile,
        debug=_get_profiled_env_bool("DEBUG", environment == "dev", profile),
        log_level=_get_profiled_env("LOG_LEVEL", "INFO", profile),
        use_gpu_mode=use_gpu_mode,
        use_gpu=_detect_gpu_requested(use_gpu_mode),
        random_seed=_get_profiled_env_int("RANDOM_SEED", 42, profile),
        max_workers=_get_profiled_env_int("MAX_WORKERS", 4, profile),
        batch_size=_get_profiled_env_int("BATCH_SIZE", 32, profile),
        batch_sleep_seconds=_get_profiled_env_float("BATCH_SLEEP_SECONDS", 0.0, profile),
        request_timeout_seconds=_get_profiled_env_int("REQUEST_TIMEOUT_SECONDS", 120, profile),
        allowed_origins=_get_env_list("ALLOWED_ORIGINS", ["*"]),
    )

    ## Resolve model section
    model = ModelConfig(
        task_type=_validate_task_type(_get_profiled_env("TASK_TYPE", "multiclass", profile)),
        pretrained_model_name=_get_profiled_env("PRETRAINED_MODEL_NAME", "camembert-base", profile),
        max_length=_get_profiled_env_int("MAX_LENGTH", 512, profile),
        learning_rate=_get_profiled_env_float("LEARNING_RATE", 2e-5, profile),
        num_epochs=_get_profiled_env_int("NUM_EPOCHS", 3, profile),
        train_split=_get_profiled_env_float("TRAIN_SPLIT", 0.8, profile),
        validation_split=_get_profiled_env_float("VALIDATION_SPLIT", 0.1, profile),
        threshold=_get_profiled_env_float("THRESHOLD", 0.5, profile),
    )

    ## Resolve secrets from direct Load JSON secrets
    secrets_path = _get_env_path("APP_SECRETS_FILE", "", project_root)

    app_json = _read_json_secret(secrets_path) if secrets_path else {}

    secrets = SecretsConfig(
        huggingface_token=app_json.get("huggingface_token", ""),
        api_key=app_json.get("api_key", ""),
    )

    ## Build final config
    config = AppConfig(
        app_name=_get_env("APP_NAME", DEFAULT_APP_NAME),
        app_version=_get_env("APP_VERSION", DEFAULT_APP_VERSION),
        execution=execution,
        paths=paths,
        runtime=runtime,
        model=model,
        secrets=secrets,
    )

    ## Validate final configuration
    _validate_config(config)

    ## Log concise configuration summary
    logger.info(
        "Configuration loaded | app=%s | env=%s | profile=%s | gpu=%s | task=%s | model=%s | run_id=%s",
        config.app_name,
        config.runtime.environment,
        config.runtime.profile,
        config.runtime.use_gpu,
        config.model.task_type,
        config.model.pretrained_model_name,
        config.execution.run_id,
    )
    return config

def load_config() -> AppConfig:
    """
        Backward-compatible alias for configuration loading

        Returns:
            AppConfig instance
    """

    ## Keep compatibility with existing imports
    return get_config()

def build_config() -> AppConfig:
    """
        Backward-compatible config builder

        Returns:
            AppConfig instance
    """

    ## Preserve the original public entrypoint
    return get_config()

## ============================================================
## PUBLIC SINGLETON CONFIG
## ============================================================
CONFIG: AppConfig = get_config()
config = CONFIG