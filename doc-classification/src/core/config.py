'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unified configuration loader for doc-classification: dotenv, env parsing, paths, profiles, thresholds, secrets and runtime metadata."
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
from typing import Any, Dict, Literal, Optional, Tuple

from src.core.errors import ConfigurationError
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
## TYPES
## ============================================================
UseGpuMode = Literal["auto", "true", "false"]
EmbeddingModelName = Literal["sentence_camembert", "drbert"]

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
DEFAULT_APP_NAME = "doc-classification"
DEFAULT_APP_VERSION = "1.0.0"
DEFAULT_ENVIRONMENT = "dev"
DEFAULT_PROFILE = "cpu"

DEFAULT_DATA_DIR = "data"
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_SECRETS_DIR = "secrets"

DEFAULT_LABELED_DIR = "data/labeled"
DEFAULT_UNLABELED_DIR = "data/unlabeled"
DEFAULT_INTERIM_DIR = "data/interim"
DEFAULT_PROCESSED_DIR = "data/processed"

DEFAULT_INDEXES_DIR = "artifacts/indexes"
DEFAULT_MODELS_DIR = "artifacts/models"
DEFAULT_REPORTS_DIR = "artifacts/reports"
DEFAULT_EXPORTS_DIR = "artifacts/exports"

DEFAULT_MANIFEST_PATH = "data/manifest.json"
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.55

LABELS: Tuple[str, ...] = (
    "crh",
    "cro",
    "cra",
    "ordonnance-examen",
    "ordonnance-medicaments",
    "analyse-labo",
    "fiche-patient-admission",
)

LABEL_DESCRIPTIONS: Dict[str, str] = {
    "crh": "Hospital discharge summary.",
    "cro": "Operative report.",
    "cra": "Anesthesia report.",
    "ordonnance-examen": "Prescription for medical exams.",
    "ordonnance-medicaments": "Medication prescription.",
    "analyse-labo": "Laboratory analysis results.",
    "fiche-patient-admission": "Patient admission form.",
}

LABEL_KEYWORD_HINTS: Dict[str, list[str]] = {
    "crh": ["compte rendu", "hospitalisation", "sortie", "diagnostic", "traitement"],
    "cro": ["compte rendu operatoire", "intervention", "bloc", "incision", "suture"],
    "cra": ["anesthesie", "induction", "intubation", "asa", "reveil"],
    "ordonnance-examen": ["prescription", "examen", "scanner", "irm", "radiographie"],
    "ordonnance-medicaments": ["posologie", "comprime", "mg", "prise", "renouvelable"],
    "analyse-labo": ["biochimie", "hematologie", "resultats", "normes", "valeurs"],
    "fiche-patient-admission": ["identite", "adresse", "assure", "mutuelle", "admission"],
}

ALLOWED_INPUT_EXTENSIONS: Tuple[str, ...] = (".txt", ".csv", ".json", ".pdf", ".docx")
SUPPORTED_EMBEDDING_MODELS: Tuple[str, ...] = ("sentence_camembert", "drbert")

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
            labeled_dir: Labeled documents directory
            unlabeled_dir: Unlabeled documents directory
            interim_dir: Interim data directory
            processed_dir: Processed data directory
            artifacts_dir: Artifacts root directory
            indexes_dir: Similarity indexes directory
            models_dir: Models directory
            reports_dir: Reports directory
            exports_dir: Exports directory
            logs_dir: Logs directory
            secrets_dir: Secrets directory
            manifest_path: Label manifest path
    """

    project_root: Path
    src_dir: Path
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
    secrets_dir: Path
    manifest_path: Path

@dataclass(frozen=True)
class RuntimeConfig:
    """
        Runtime configuration

        Args:
            environment: Environment name
            profile: Runtime profile
            debug: Whether debug mode is enabled
            log_level: Logging level
            use_gpu_mode: Raw GPU mode
            use_gpu: Resolved GPU usage
            max_workers: Maximum worker count
            batch_sleep_seconds: Sleep delay between batches
            allowed_origins: Allowed HTTP origins if needed later
            anomaly_detection_enabled: Enable anomaly detection
            anomaly_method: Detection method (zscore or iqr)
            z_threshold: Z-score threshold
            iqr_multiplier: IQR multiplier
            anomaly_strict_mode: Raise error if anomalies are detected    
    """

    environment: str
    profile: str
    debug: bool
    log_level: str
    use_gpu_mode: UseGpuMode
    use_gpu: bool
    max_workers: int
    batch_sleep_seconds: float
    allowed_origins: list[str]
    anomaly_detection_enabled: bool
    anomaly_method: str
    z_threshold: float
    iqr_multiplier: float
    anomaly_strict_mode: bool
    
@dataclass(frozen=True)
class SegmentationConfig:
    """
        Segment creation configuration

        Args:
            window_size_tokens: Window size in tokens
            window_overlap_tokens: Window overlap in tokens
            min_chars_per_segment: Minimum segment size in characters
            split_on_paragraphs: Whether paragraph splitting is enabled
    """

    window_size_tokens: int
    window_overlap_tokens: int
    min_chars_per_segment: int
    split_on_paragraphs: bool

@dataclass(frozen=True)
class EmbeddingsConfig:
    """
        Embeddings configuration

        Args:
            model_name: Embedding model name
            use_gpu: Whether embeddings use GPU
            batch_size: Embedding batch size
            normalize: Whether vectors are normalized
            cache_embeddings: Whether embedding cache is enabled
    """

    model_name: EmbeddingModelName
    use_gpu: bool
    batch_size: int
    normalize: bool
    cache_embeddings: bool

@dataclass(frozen=True)
class SimilarityConfig:
    """
        Similarity search and threshold configuration

        Args:
            top_k: Top-k nearest neighbors
            thresholds: Per-label thresholds
            default_threshold: Default threshold value
            min_positive_labels: Minimum positive labels to keep
    """

    top_k: int
    thresholds: Dict[str, float]
    default_threshold: float
    min_positive_labels: int

@dataclass(frozen=True)
class DataConsistencyConfig:
    """
        Data consistency configuration

        Args:
            enabled: Enable consistency checks
            strict_mode: Raise error if inconsistency
            min_text_length: Minimum text length
            max_records: Maximum dataset size

        Returns:
            None
    """

    enabled: bool
    strict_mode: bool
    min_text_length: int
    max_records: int
    
@dataclass(frozen=True)
class SecretsConfig:
    """
        Secret values resolved from env or files

        Args:
            huggingface_token: Hugging Face token
            api_key: Generic API key
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
            segmentation: Segmentation configuration
            embeddings: Embeddings configuration
            similarity: Similarity configuration
            data_consistency: Data consistency configuration
            secrets: Secrets configuration
    """

    app_name: str
    app_version: str
    execution: ExecutionMetadata
    paths: PathsConfig
    runtime: RuntimeConfig
    segmentation: SegmentationConfig
    embeddings: EmbeddingsConfig
    similarity: SimilarityConfig
    data_consistency: DataConsistencyConfig
    secrets: SecretsConfig

## ============================================================
## DOTENV / ENV HELPERS
## ============================================================
def _resolve_project_root() -> Path:
    """
        Resolve the project root path

        High-level workflow:
            1) Prefer PROJECT_ROOT when explicitly provided
            2) Otherwise derive the root from this file location

        Returns:
            Resolved project root path
    """

    ## Prefer explicit override first
    project_root_raw = os.getenv("PROJECT_ROOT", "").strip()
    return Path(project_root_raw).expanduser().resolve() if project_root_raw else Path(__file__).resolve().parents[2]

def _load_dotenv_if_present() -> None:
    """
        Load .env if python-dotenv is available and a file exists

        Returns:
            None
    """

    ## Import dotenv lazily
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    ## Load only when .env exists
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

def _get_env(name: str, default: str = "") -> str:
    """
        Read an environment variable safely

        Args:
            name: Environment variable name
            default: Default fallback value

        Returns:
            Normalized string value
    """

    ## Avoid None propagation
    value = os.getenv(name, default)
    return (value if value is not None else default).strip()

def _get_env_bool(name: str, default: bool) -> bool:
    """
        Parse a boolean environment variable

        Args:
            name: Environment variable name
            default: Default fallback value

        Returns:
            Parsed boolean value

        Raises:
            ConfigurationError: If the value is invalid
    """

    ## Read raw value
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
            ConfigurationError: If the value is invalid
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
            ConfigurationError: If the value is invalid
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
            separator: Separator used in the raw value

        Returns:
            Parsed list of strings
    """

    ## Read raw value
    raw = _get_env(name, "")
    if not raw:
        return list(default or [])
    return [item.strip() for item in raw.split(separator) if item.strip()]

def _expand_env_vars(value: str) -> str:
    """
        Expand shell variables and user home in a path-like string

        Args:
            value: Raw path-like string

        Returns:
            Expanded string
    """

    ## Expand shell variables and user home
    return os.path.expandvars(value)

def _resolve_path(path_value: str, project_root: Path) -> Path:
    """
        Resolve a path against the project root

        Args:
            path_value: Raw path value
            project_root: Project root directory

        Returns:
            Resolved path
    """

    ## Expand shell variables and user home
    path_obj = Path(_expand_env_vars(path_value)).expanduser()
    return path_obj.resolve() if path_obj.is_absolute() else (project_root / path_obj).resolve()

def _get_env_path(name: str, default: str, project_root: Path) -> Path:
    """
        Read and resolve a path environment variable

        Args:
            name: Environment variable name
            default: Default path string
            project_root: Project root directory

        Returns:
            Resolved path
    """

    ## Resolve path from env or default
    return _resolve_path(_get_env(name, default), project_root)

def _read_secret_value(direct_key: str, file_key: str, *, project_root: Path, default: str = "") -> str:
    """
        Read a secret from env directly or from a file path

        High-level workflow:
            1) Prefer direct env value
            2) Otherwise use the env file path if provided
            3) Return default when not available

        Args:
            direct_key: Environment variable containing the secret
            file_key: Environment variable containing the secret file path
            project_root: Project root directory
            default: Default fallback value

        Returns:
            Secret value or default
    """

    ## Prefer direct value first
    direct_value = _get_env(direct_key, default)
    if direct_value and not _is_placeholder(direct_value):
        return direct_value

    ## Fallback to secret file
    secret_file_raw = _get_env(file_key, "")
    if not secret_file_raw:
        return default

    ## Resolve and read file when available
    secret_file = _resolve_path(secret_file_raw, project_root)
    return secret_file.read_text(encoding=DEFAULT_ENCODING).strip() if secret_file.exists() and secret_file.is_file() else default

## ============================================================
## PROFILE HELPERS
## ============================================================
def _get_profiled_env(name: str, default: str, profile: str) -> str:
    """
        Read an env value with optional profile override

        Args:
            name: Base environment variable name
            default: Default fallback value
            profile: Active profile name

        Returns:
            Resolved string value
    """

    ## Try profile override first
    override_key = f"{profile.upper()}_{name}"
    return _get_env(override_key, default) if os.getenv(override_key) is not None else _get_env(name, default)

def _get_profiled_env_bool(name: str, default: bool, profile: str) -> bool:
    """
        Read a boolean env value with optional profile override

        Args:
            name: Base environment variable name
            default: Default fallback value
            profile: Active profile name

        Returns:
            Parsed boolean value
    """

    ## Try profile override first
    override_key = f"{profile.upper()}_{name}"
    return _get_env_bool(override_key, default) if os.getenv(override_key) is not None else _get_env_bool(name, default)

def _get_profiled_env_int(name: str, default: int, profile: str) -> int:
    """
        Read an integer env value with optional profile override

        Args:
            name: Base environment variable name
            default: Default fallback value
            profile: Active profile name

        Returns:
            Parsed integer value
    """

    ## Try profile override first
    override_key = f"{profile.upper()}_{name}"
    return _get_env_int(override_key, default) if os.getenv(override_key) is not None else _get_env_int(name, default)

def _get_profiled_env_float(name: str, default: float, profile: str) -> float:
    """
        Read a float env value with optional profile override

        Args:
            name: Base environment variable name
            default: Default fallback value
            profile: Active profile name

        Returns:
            Parsed float value
    """

    ## Try profile override first
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

def _default_thresholds(default_value: float = DEFAULT_THRESHOLD) -> Dict[str, float]:
    """
        Provide default per-label thresholds

        Args:
            default_value: Default threshold applied to each label

        Returns:
            Per-label thresholds dictionary
    """

    ## Build homogeneous threshold map
    return {label: default_value for label in LABELS}

def _validate_required_placeholders(keys: list[str]) -> None:
    """
        Validate that required env keys do not keep placeholder values

        Args:
            keys: Environment keys to inspect

        Returns:
            None

        Raises:
            ConfigurationError: If placeholders are detected
    """

    ## Track invalid required keys
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
            ConfigurationError: If the value is invalid
    """

    ## Reject non-positive values
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
            ConfigurationError: If the value is invalid
    """

    ## Reject negative values
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
            ConfigurationError: If the value is invalid
    """

    ## Reject invalid probabilities
    if not 0.0 <= value <= 1.0:
        raise ConfigurationError(f"{field_name} must be in [0, 1]. Got: {value}")

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
            config.data_consistency.max_records,
            "DATA_CONSISTENCY_MAX_RECORDS",
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
        Load the unified configuration from environment variables

        High-level workflow:
            1) Load optional .env
            2) Resolve project root and runtime profile
            3) Resolve filesystem paths
            4) Build segmentation, embeddings and similarity sections
            5) Read secrets from env or files
            6) Validate and cache the final AppConfig

        Returns:
            Structured AppConfig
    """

    ## Load optional local .env file
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
        anomaly_detection_enabled=_get_env_bool("ANOMALY_DETECTION_ENABLED", True),
        anomaly_method=_get_env("ANOMALY_METHOD", "zscore"),
        z_threshold=_get_env_float("Z_THRESHOLD", 3.0),
        iqr_multiplier=_get_env_float("IQR_MULTIPLIER", 1.5),
        anomaly_strict_mode=_get_env_bool("ANOMALY_STRICT_MODE", False),        
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
        max_records=_get_env_int("DATA_CONSISTENCY_MAX_RECORDS", 100000),
    )
    
    ## Resolve secrets
    secrets = SecretsConfig(
        huggingface_token=_read_secret_value("HUGGINGFACE_TOKEN", "HUGGINGFACE_TOKEN_FILE", project_root=project_root),
        api_key=_read_secret_value("API_KEY", "API_KEY_FILE", project_root=project_root),
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
            Structured AppConfig
    """

    ## Keep compatibility with existing imports
    return get_config()

def build_config() -> AppConfig:
    """
        Backward-compatible config builder

        Returns:
            Structured AppConfig
    """

    ## Preserve an additional public entrypoint
    return get_config()

## ============================================================
## PUBLIC SINGLETON CONFIG
## ============================================================
CONFIG: AppConfig = get_config()
config = CONFIG