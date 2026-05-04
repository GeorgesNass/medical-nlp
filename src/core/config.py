'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unified configuration loader for lab-clustering: dotenv, env parsing, paths, profiles, MLflow, secrets and runtime metadata."
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

from src.core.errors import ConfigurationError
from src.utils.logging_utils import get_logger

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
DEFAULT_APP_NAME = "lab-clustering"
DEFAULT_APP_VERSION = "1.0.0"
DEFAULT_ENVIRONMENT = "dev"
DEFAULT_PROFILE = "cpu"

DEFAULT_DATA_DIR = "data"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_SECRETS_DIR = "secrets"

DEFAULT_MLFLOW_DIR = "artifacts/mlflow"
DEFAULT_MLFLOW_EXPERIMENT = "lab_clustering"

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_INTERIM_DIR = "data/interim"
DEFAULT_PROCESSED_DIR = "data/processed"

DEFAULT_STRUCTURED_DIRNAME = "lab_structured_csv"
DEFAULT_DATASETS_DIRNAME = "datasets"
DEFAULT_FEATURES_DIRNAME = "features"
DEFAULT_ERROR_ANALYSIS_DIRNAME = "error_analysis"

DEFAULT_MODELS_DIR = "artifacts/models"
DEFAULT_EXPORTS_DIR = "artifacts/exports"
DEFAULT_RESOURCES_DIR = "artifacts/resources"
DEFAULT_CONFIG_DIR = "artifacts/config"

SUPPORTED_INPUT_EXTENSIONS = (".txt", ".csv", ".json", ".xlsx", ".xls")

def _read_json_secret(secret_file: Path) -> dict[str, Any]:
    """
        Read a JSON secret file safely

        Args:
            secret_file (Path): Path to the JSON secret file

        Returns:
            dict[str, Any]: Parsed JSON content or empty dict if invalid
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
            raw_dir: Raw input directory
            interim_dir: Interim data directory
            processed_dir: Processed data directory
            interim_structured_dir: Structured CSV interim directory
            interim_datasets_dir: Dataset interim directory
            processed_features_dir: Processed features directory
            processed_error_analysis_dir: Error analysis directory
            artifacts_dir: Artifacts root directory
            artifacts_models_dir: Models directory
            artifacts_exports_dir: Exports directory
            artifacts_resources_dir: Resources directory
            artifacts_config_dir: Config directory
            logs_dir: Logs directory
            secrets_dir: Secrets directory
    """

    project_root: Path
    src_dir: Path
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
    secrets_dir: Path

@dataclass(frozen=True)
class MlflowConfig:
    """
        MLflow configuration

        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Experiment name
            enabled: Whether MLflow logging is enabled
            artifact_location: Artifact location
    """

    tracking_uri: str
    experiment_name: str
    enabled: bool
    artifact_location: str

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
            max_workers: Maximum number of workers
            batch_size: Generic batch size
            batch_sleep_seconds: Sleep delay between batches
            allowed_origins: Allowed origins for future API usage
            anomaly_detection_enabled: Enable anomaly detection
            anomaly_method: Detection method (zscore or iqr)
            z_threshold: Z-score threshold
            iqr_multiplier: IQR multiplier
            anomaly_strict_mode: Raise error if anomaly detected
            drift_detection_enabled: Enable data drift detection
            drift_p_value_threshold: Statistical p-value threshold for drift detection
            drift_embedding_threshold: Threshold for embedding drift
            drift_cluster_threshold: Threshold for cluster distribution drift
            drift_distance_threshold: Threshold for distance drift
            drift_evidently_enabled: Enable Evidently report generation
            drift_strict_mode: Raise error if drift detected            
    """

    environment: str
    profile: str
    debug: bool
    log_level: str
    use_gpu_mode: str
    use_gpu: bool
    random_seed: int
    max_workers: int
    batch_size: int
    batch_sleep_seconds: float
    allowed_origins: list[str]
    anomaly_detection_enabled: bool
    anomaly_method: str
    z_threshold: float
    iqr_multiplier: float
    anomaly_strict_mode: bool
    drift_detection_enabled: bool
    drift_p_value_threshold: float
    drift_embedding_threshold: float
    drift_cluster_threshold: float
    drift_distance_threshold: float
    drift_evidently_enabled: bool
    drift_strict_mode: bool
    
@dataclass(frozen=True)
class ClusteringConfig:
    """
        Clustering configuration

        Args:
            default_algorithm: Default clustering algorithm
            default_n_clusters: Default number of clusters
            use_pca: Whether PCA is enabled
            pca_components: Number of PCA components
            scale_features: Whether feature scaling is enabled
    """

    default_algorithm: str
    default_n_clusters: int
    use_pca: bool
    pca_components: int
    scale_features: bool
    
@dataclass(frozen=True)
class DataConsistencyConfig:
    """
        Data consistency configuration

        Args:
            enabled: Enable consistency checks
            strict_mode: Raise error if inconsistency
            min_text_length: Minimum text length
            embedding_dim: Expected embedding dimension
    """

    enabled: bool
    strict_mode: bool
    min_text_length: int
    embedding_dim: int

@dataclass(frozen=True)
class SecretsConfig:
    """
        Secret values resolved from env or files

        Args:
            mlflow_tracking_username: Optional MLflow username
            mlflow_tracking_password: Optional MLflow password
    """

    mlflow_tracking_username: str
    mlflow_tracking_password: str

@dataclass(frozen=True)
class AppConfig:
    """
        Unified application configuration

        Args:
            app_name: Application name
            app_version: Application version
            execution: Execution metadata
            paths: Filesystem paths configuration
            mlflow: MLflow configuration
            runtime: Runtime configuration
            clustering: Clustering configuration
            secrets: Secret values
    """

    app_name: str
    app_version: str
    execution: ExecutionMetadata
    paths: PathsConfig
    mlflow: MlflowConfig
    runtime: RuntimeConfig
    clustering: ClusteringConfig
    secrets: SecretsConfig
    data_consistency: DataConsistencyConfig
    
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

    ## Normalize before checking placeholder tokens
    normalized = value.strip().upper()
    return any(token in normalized for token in PLACEHOLDER_PREFIXES)

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

    ## Read raw value from process environment
    value = os.getenv(name)
    if value is None:
        if default is None:
            raise ConfigurationError(message=f"Missing environment variable: {name}", details={"missing": [name]})
        return default
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

    ## Normalize boolean value
    raw = _get_env(name, str(default)).lower()
    if raw in {"true", "1", "yes", "y", "on"}:
        return True
    if raw in {"false", "0", "no", "n", "off"}:
        return False
    raise ConfigurationError(message=f"Invalid boolean value for {name}: {raw}", details={"name": name, "value": raw})

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
        raise ConfigurationError(message=f"{name} must be an integer", details={"name": name}) from exc

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
        raise ConfigurationError(message=f"{name} must be a float", details={"name": name}) from exc

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
        Expand shell variables in a string

        Args:
            value: Raw string value

        Returns:
            Expanded string
    """

    ## Expand shell variables such as %USERPROFILE% or $HOME
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

    ## Expand shell variables and user home first
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
def _detect_gpu_requested(mode: str) -> bool:
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
        Validate that required values are not unresolved placeholders

        Args:
            keys: Environment keys to inspect

        Returns:
            None

        Raises:
            ConfigurationError: If placeholders are detected
    """

    ## Collect required keys still using placeholder values
    invalid_keys = [key for key in keys if (value := _get_env(key, "")) and _is_placeholder(value)]
    if invalid_keys:
        raise ConfigurationError(message="Placeholder values detected", details={"invalid_keys": invalid_keys})

def _validate_positive_int(value: int, field_name: str) -> None:
    """
        Validate that an integer is strictly positive

        Args:
            value: Value to validate
            field_name: Human-readable field name

        Returns:
            None

        Raises:
            ConfigurationError: If the value is invalid
    """

    ## Reject non-positive values
    if value <= 0:
        raise ConfigurationError(message=f"{field_name} must be > 0. Got: {value}", details={"field": field_name, "value": value})

def _validate_non_negative_int(value: int, field_name: str) -> None:
    """
        Validate that an integer is non-negative

        Args:
            value: Value to validate
            field_name: Human-readable field name

        Returns:
            None

        Raises:
            ConfigurationError: If the value is invalid
    """

    ## Reject negative values
    if value < 0:
        raise ConfigurationError(message=f"{field_name} must be >= 0. Got: {value}", details={"field": field_name, "value": value})

def _validate_probability(value: float, field_name: str) -> None:
    """
        Validate that a float is inside [0, 1]

        Args:
            value: Value to validate
            field_name: Human-readable field name

        Returns:
            None

        Raises:
            ConfigurationError: If the value is invalid
    """

    ## Reject invalid probabilities
    if not 0.0 <= value <= 1.0:
        raise ConfigurationError(message=f"{field_name} must be in [0, 1]. Got: {value}", details={"field": field_name, "value": value})

def _validate_embedding_model(value: str) -> EmbeddingModelName:
    """
        Validate the embedding model name

        Args:
            value: Raw embedding model name

        Returns:
            Validated embedding model name

        Raises:
            ConfigurationError: If the value is unsupported
    """

    ## Restrict models to known values
    if value not in SUPPORTED_EMBEDDING_MODELS:
        raise ConfigurationError("EMBEDDING_MODEL must be one of: sentence_camembert, drbert")
    return value  # type: ignore[return-value]

def _default_thresholds(default_value: float = DEFAULT_THRESHOLD) -> Dict[str, float]:
    """
        Provide default per-label thresholds

        Args:
            default_value: Default threshold applied to each label

        Returns:
            Per-label thresholds dictionary
    """

    ## Build a homogeneous threshold map
    return {label: default_value for label in LABELS}

def _ensure_directories_exist(paths: list[Path]) -> None:
    """
        Ensure runtime directories exist

        Args:
            paths: Directories to create if missing

        Returns:
            None
    """

    ## Create each runtime directory safely
    for directory in paths:
        directory.mkdir(parents=True, exist_ok=True)

def _validate_thresholds(thresholds: Dict[str, float]) -> None:
    """
        Validate all similarity thresholds

        Args:
            thresholds: Per-label thresholds

        Returns:
            None
        """

    ## Validate each threshold individually
    for label, value in thresholds.items():
        _validate_probability(value, f"THRESH_{label.upper()}")

def _validate_config(config: AppConfig) -> None:
    """
        Validate the final structured configuration

        Args:
            config: Structured application configuration

        Returns:
            None
        """

    ## Validate segmentation parameters
    _validate_positive_int(config.segmentation.window_size_tokens, "WINDOW_SIZE_TOKENS")
    _validate_non_negative_int(config.segmentation.window_overlap_tokens, "WINDOW_OVERLAP_TOKENS")
    _validate_positive_int(config.segmentation.min_chars_per_segment, "MIN_CHARS_PER_SEGMENT")

    ## Validate embedding parameters
    _validate_positive_int(config.embeddings.batch_size, "EMBEDDING_BATCH_SIZE")

    ## Validate runtime parameters
    _validate_positive_int(config.runtime.max_workers, "MAX_WORKERS")
    if config.runtime.batch_sleep_seconds < 0:
        raise ConfigurationError(f"BATCH_SLEEP_SECONDS must be >= 0. Got: {config.runtime.batch_sleep_seconds}")

    ## Validate similarity parameters
    _validate_positive_int(config.similarity.top_k, "TOP_K")
    _validate_non_negative_int(config.similarity.min_positive_labels, "MIN_POSITIVE_LABELS")
    _validate_probability(config.similarity.default_threshold, "DEFAULT_THRESHOLD")
    _validate_thresholds(config.similarity.thresholds)

    ## Validate data consistency config
    if config.data_consistency.enabled:

        _validate_positive_int(
            config.data_consistency.min_text_length,
            "DATA_CONSISTENCY_MIN_TEXT_LENGTH",
        )

        _validate_positive_int(
            config.data_consistency.embedding_dim,
            "DATA_CONSISTENCY_EMBEDDING_DIM",
        )

        if config.data_consistency.strict_mode and not config.data_consistency.enabled:
            raise ConfigurationError(
                "DATA_CONSISTENCY_STRICT requires DATA_CONSISTENCY_ENABLED=True"
            )
            
    ## Validate cross-field consistency
    if config.segmentation.window_overlap_tokens >= config.segmentation.window_size_tokens:
        raise ConfigurationError("WINDOW_OVERLAP_TOKENS must be smaller than WINDOW_SIZE_TOKENS.")
    if set(config.similarity.thresholds.keys()) != set(LABELS):
        raise ConfigurationError("Similarity thresholds must contain exactly all known labels.")

    ## Validate anomaly detection config
    if config.runtime.anomaly_detection_enabled:

        if config.runtime.anomaly_method not in {"zscore", "iqr"}:
            raise ConfigurationError("ANOMALY_METHOD must be 'zscore' or 'iqr'")

        if config.runtime.z_threshold <= 0:
            raise ConfigurationError("Z_THRESHOLD must be > 0")

        if config.runtime.iqr_multiplier <= 0:
            raise ConfigurationError("IQR_MULTIPLIER must be > 0")
 
    ## Validate drift parameters
    if config.runtime.drift_detection_enabled:

        _validate_probability(
            config.runtime.drift_p_value_threshold,
            "DRIFT_P_VALUE_THRESHOLD",
        )

        _validate_non_negative_float(
            config.runtime.drift_embedding_threshold,
            "DRIFT_EMBEDDING_THRESHOLD",
        )

        _validate_non_negative_float(
            config.runtime.drift_cluster_threshold,
            "DRIFT_CLUSTER_THRESHOLD",
        )

        _validate_non_negative_float(
            config.runtime.drift_distance_threshold,
            "DRIFT_DISTANCE_THRESHOLD",
        )
        
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

    ## Convert dataclass tree first
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

    ## Serialize normalized config
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
            3) Build execution, paths, runtime, segmentation, embeddings and similarity
            4) Resolve optional secrets
            5) Validate and cache the final AppConfig

        Returns:
            AppConfig instance
    """

    ## Load optional local .env file first
    _load_dotenv_if_present()

    ## Resolve root and runtime profile
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

    ## Resolve main folders
    data_dir = _get_env_path("DATA_DIR", DEFAULT_DATA_DIR, project_root)
    artifacts_dir = _get_env_path("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR, project_root)
    logs_dir = _get_env_path("LOGS_DIR", DEFAULT_LOGS_DIR, project_root)
    secrets_dir = _get_env_path("SECRETS_DIR", DEFAULT_SECRETS_DIR, project_root)

    ## Build all paths
    paths = PathsConfig(
        project_root=project_root,
        src_dir=(project_root / "src").resolve(),
        data_dir=data_dir,
        labeled_dir=_get_env_path("LABELED_DIR", DEFAULT_LABELED_DIR, project_root),
        unlabeled_dir=_get_env_path("UNLABELED_DIR", DEFAULT_UNLABELED_DIR, project_root),
        interim_dir=_get_env_path("INTERIM_DIR", DEFAULT_INTERIM_DIR, project_root),
        processed_dir=_get_env_path("PROCESSED_DIR", DEFAULT_PROCESSED_DIR, project_root),
        artifacts_dir=artifacts_dir,
        indexes_dir=_get_env_path("INDEXES_DIR", DEFAULT_INDEXES_DIR, project_root),
        models_dir=_get_env_path("MODELS_DIR", DEFAULT_MODELS_DIR, project_root),
        reports_dir=_get_env_path("REPORTS_DIR", DEFAULT_REPORTS_DIR, project_root),
        exports_dir=_get_env_path("EXPORTS_DIR", DEFAULT_EXPORTS_DIR, project_root),
        logs_dir=logs_dir,
        secrets_dir=secrets_dir,
        manifest_path=_get_env_path("MANIFEST_PATH", DEFAULT_MANIFEST_PATH, project_root),
    )

    ## Ensure runtime directories exist
    _ensure_directories_exist([
        paths.data_dir, paths.labeled_dir, paths.unlabeled_dir, paths.interim_dir,
        paths.processed_dir, paths.artifacts_dir, paths.indexes_dir, paths.models_dir,
        paths.reports_dir, paths.exports_dir, paths.logs_dir, paths.secrets_dir,
    ])

    ## Resolve runtime settings
    use_gpu_mode_raw = _get_profiled_env("USE_GPU", "auto", profile).lower()
    if use_gpu_mode_raw not in {"auto", "true", "false"}:
        raise ConfigurationError("USE_GPU must be auto|true|false")
    use_gpu_mode: UseGpuMode = use_gpu_mode_raw  # type: ignore[assignment]
    use_gpu = _detect_gpu_requested(use_gpu_mode)

    runtime = RuntimeConfig(
        environment=environment,
        profile=profile,
        debug=_get_profiled_env_bool("DEBUG", environment == "dev", profile),
        log_level=_get_profiled_env("LOG_LEVEL", "INFO", profile),
        use_gpu_mode=use_gpu_mode,
        use_gpu=use_gpu,
        max_workers=_get_profiled_env_int("MAX_WORKERS", 4, profile),
        batch_sleep_seconds=_get_profiled_env_float("BATCH_SLEEP_SECONDS", 0.0, profile),
        allowed_origins=_get_env_list("ALLOWED_ORIGINS", ["*"]),
        anomaly_detection_enabled=_get_profiled_env_bool("ANOMALY_DETECTION_ENABLED", True, profile),
        anomaly_method=_get_profiled_env("ANOMALY_METHOD", "zscore", profile),
        z_threshold=_get_profiled_env_float("Z_THRESHOLD", 3.0, profile),
        iqr_multiplier=_get_profiled_env_float("IQR_MULTIPLIER", 1.5, profile),
        anomaly_strict_mode=_get_profiled_env_bool("ANOMALY_STRICT_MODE", False, profile),
        drift_detection_enabled=_get_profiled_env_bool("DRIFT_DETECTION_ENABLED", True, profile),
        drift_p_value_threshold=_get_profiled_env_float("DRIFT_P_VALUE_THRESHOLD", 0.05, profile),
        drift_embedding_threshold=_get_profiled_env_float("DRIFT_EMBEDDING_THRESHOLD", 0.2, profile),
        drift_cluster_threshold=_get_profiled_env_float("DRIFT_CLUSTER_THRESHOLD", 0.2, profile),
        drift_distance_threshold=_get_profiled_env_float("DRIFT_DISTANCE_THRESHOLD", 0.2, profile),
        drift_evidently_enabled=_get_profiled_env_bool("DRIFT_EVIDENTLY_ENABLED", True, profile),
        drift_strict_mode=_get_profiled_env_bool("DRIFT_STRICT_MODE", False, profile),        
    )

    ## Build segmentation parameters
    segmentation = SegmentationConfig(
        window_size_tokens=_get_profiled_env_int("WINDOW_SIZE_TOKENS", 220, profile),
        window_overlap_tokens=_get_profiled_env_int("WINDOW_OVERLAP_TOKENS", 60, profile),
        min_chars_per_segment=_get_profiled_env_int("MIN_CHARS_PER_SEGMENT", 50, profile),
        split_on_paragraphs=_get_profiled_env_bool("SPLIT_ON_PARAGRAPHS", True, profile),
    )

    ## Build embeddings parameters
    embeddings = EmbeddingsConfig(
        model_name=_validate_embedding_model(_get_profiled_env("EMBEDDING_MODEL", "sentence_camembert", profile)),
        use_gpu=use_gpu,
        batch_size=_get_profiled_env_int("EMBEDDING_BATCH_SIZE", 32, profile),
        normalize=_get_profiled_env_bool("EMBEDDING_NORMALIZE", True, profile),
        cache_embeddings=_get_profiled_env_bool("CACHE_EMBEDDINGS", True, profile),
    )

    ## Build similarity thresholds
    default_threshold = _get_profiled_env_float("DEFAULT_THRESHOLD", DEFAULT_THRESHOLD, profile)
    thresholds = _default_thresholds(default_threshold)
    for label in LABELS:
        env_key = f"THRESH_{label.upper()}"
        profile_key = f"{profile.upper()}_{env_key}"
        if os.getenv(profile_key) is not None:
            thresholds[label] = _get_env_float(profile_key, thresholds[label])
        elif os.getenv(env_key) is not None:
            thresholds[label] = _get_env_float(env_key, thresholds[label])

    similarity = SimilarityConfig(
        top_k=_get_profiled_env_int("TOP_K", DEFAULT_TOP_K, profile),
        thresholds=thresholds,
        default_threshold=default_threshold,
        min_positive_labels=_get_profiled_env_int("MIN_POSITIVE_LABELS", 1, profile),
    )

    ## Build data consistency config
    data_consistency = DataConsistencyConfig(
        enabled=_get_env_bool("DATA_CONSISTENCY_ENABLED", True),
        strict_mode=_get_env_bool("DATA_CONSISTENCY_STRICT", False),
        min_text_length=_get_env_int("DATA_CONSISTENCY_MIN_TEXT_LENGTH", 3),
        embedding_dim=_get_env_int("DATA_CONSISTENCY_EMBEDDING_DIM", 768),
    )
    
    ## Resolve optional secrets from direct Load JSON secrets
    secrets_path = _get_env_path("APP_SECRETS_FILE", "", project_root)

    app_json = _read_json_secret(secrets_path) if secrets_path else {}

    secrets = SecretsConfig(
        mlflow_tracking_username=_read_secret_value(
            "MLFLOW_TRACKING_USERNAME",
            "MLFLOW_TRACKING_USERNAME_FILE",
            project_root=project_root,
        ),
        mlflow_tracking_password=_read_secret_value(
            "MLFLOW_TRACKING_PASSWORD",
            "MLFLOW_TRACKING_PASSWORD_FILE",
            project_root=project_root,
        ),
    )

    ## Build final application config
    config = AppConfig(
        app_name=_get_env("APP_NAME", DEFAULT_APP_NAME),
        app_version=_get_env("APP_VERSION", DEFAULT_APP_VERSION),
        execution=execution,
        paths=paths,
        runtime=runtime,
        segmentation=segmentation,
        embeddings=embeddings,
        similarity=similarity,
        secrets=secrets,
        data_consistency=data_consistency,        
    )

    ## Validate final config
    _validate_config(config)

    ## Log concise configuration summary
    logger.info(
        "Configuration loaded | app=%s | env=%s | profile=%s | gpu=%s | model=%s | top_k=%s | run_id=%s",
        config.app_name,
        config.runtime.environment,
        config.runtime.profile,
        config.runtime.use_gpu,
        config.embeddings.model_name,
        config.similarity.top_k,
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

    ## Preserve an additional public entrypoint
    return get_config()

## ============================================================
## PUBLIC SINGLETON CONFIG
## ============================================================
CONFIG: AppConfig = get_config()
config = CONFIG