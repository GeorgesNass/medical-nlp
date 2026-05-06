'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "I/O + normalization utilities for medical documents (.txt assumed) with safe UTF-8 handling."
'''

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional
import os
from typing import Dict, Any
import redis

try:
    from feast import FeatureStore
except Exception:
    FeatureStore = None

from src.core.config import get_config
from src.utils.normalization import normalize_medical_text
from src.core.errors import DataError, PipelineError
from src.utils.logging_utils import get_logger

FEATURE_STORE_MODE = os.getenv("FEATURE_STORE_MODE", "redis")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "./feature_repo")


redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True
)

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("io_utils")

## ============================================================
## CONSTANTS
## ============================================================
## We assume documents are already converted to .txt (as requested)
SUPPORTED_EXTENSIONS = {".txt"}

## Common whitespace cleanup
_WHITESPACE_RE = re.compile(r"\s+")

## Some PDF conversions introduce weird control chars (keep conservative)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

## ============================================================
## FILE LISTING
## ============================================================
def list_supported_files(folder: str | Path, recursive: bool = False) -> List[Path]:
    """
        List supported document files in a folder.

        Notes:
            - Only .txt is supported (project assumption)
            - By default, this is non-recursive

        Args:
            folder: Input folder path
            recursive: If True, search recursively

        Returns:
            Sorted list of file paths

        Raises:
            DataError: If folder does not exist or is not a directory
    """

    folder_path = Path(folder).expanduser().resolve()

    ## Validate folder
    if not folder_path.exists() or not folder_path.is_dir():
        raise DataError(f"Invalid folder: {folder_path}")

    ## Collect files
    if recursive:
        files = [p for p in folder_path.rglob("*") if p.is_file()]
    else:
        files = [p for p in folder_path.iterdir() if p.is_file()]

    ## Filter supported extensions
    supported = [
        p for p in sorted(files)
        if p.suffix.lower().strip() in SUPPORTED_EXTENSIONS
    ]

    return supported

## ============================================================
## LOADERS (TXT ONLY)
## ============================================================
def load_text_from_path(file_path: str | Path) -> str:
    """
        Load raw text from a supported file path.

        Supported formats:
            - .txt

        Args:
            file_path: Input file path

        Returns:
            Extracted raw text

        Raises:
            DataError: If file is missing or extension not supported
            PipelineError: If reading fails
    """

    path = Path(file_path).expanduser().resolve()

    ## Defensive checks
    if not path.exists():
        raise DataError(f"File not found: {path}")

    if path.is_dir():
        raise DataError(f"Expected a file path, got a directory: {path}")

    suffix = path.suffix.lower().strip()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DataError(
            f"Unsupported file extension: {suffix} ({path.name}). "
            "This project assumes all documents are already in .txt."
        )

    ## Read text with safe fallbacks
    return _load_txt(path)

def load_and_normalize_text_from_path(file_path: str | Path) -> str:
    """
        Load and normalize text using feature engineering pipeline

        Args:
            file_path: Input file path

        Returns:
            Processed text ready for downstream pipeline
    """

    ## Load raw text first
    text = load_text_from_path(file_path)

    ## Apply feature engineering if enabled
    config = get_config()

    if config.feature_engineering.enabled:
        return normalize_medical_text(text)

    return text
    
def load_text_from_bytes(
    content: bytes,
    encoding: str = "utf-8",
) -> str:
    """
        Load raw text from in-memory bytes.

        Notes:
            - Mainly used for tests or API ingestion
            - Decoding errors are replaced

        Args:
            content: Raw bytes content
            encoding: Preferred encoding (default: utf-8)

        Returns:
            Decoded text

        Raises:
            PipelineError: If decoding fails unexpectedly
    """

    try:
        return content.decode(encoding, errors="replace")
    except Exception as exc:
        raise PipelineError(f"Failed to decode bytes as {encoding}") from exc

def _load_txt(path: Path) -> str:
    """
        Load a .txt file with safe encoding fallbacks.

        Strategy:
            1) utf-8
            2) latin-1 (legacy exports)

        Args:
            path: TXT file path

        Returns:
            Extracted raw text

        Raises:
            PipelineError: If reading fails
    """

    ## Try UTF-8 first
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ## Fallback to latin-1
    try:
        return path.read_text(encoding="latin-1", errors="replace")
    except Exception as exc:
        raise PipelineError(f"Failed to read TXT file: {path}") from exc

## ============================================================
## NORMALIZATION
## ============================================================
def normalize_document_text(text: str) -> str:
    """
        Normalize extracted document text for downstream NLP.

        What we do (conservative):
            - Replace control characters
            - Collapse repeated whitespace
            - Strip leading/trailing spaces

        What we do NOT do (by design):
            - Do not lowercase (medical acronyms, names)
            - Do not remove punctuation (useful for patterns and evidence)
            - Do not remove accents (French medical terms)

        Args:
            text: Raw extracted text

        Returns:
            Normalized text
    """

    if not text:
        return ""

    ## Feature engineering override
    config = get_config()

    if config.feature_engineering.enabled:
        return normalize_medical_text(text)
        
    ## Remove control characters sometimes introduced by converters
    cleaned = _CONTROL_CHARS_RE.sub(" ", text)

    ## Collapse whitespace while preserving line breaks semantics loosely
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)

    return cleaned.strip()

def normalize_texts(texts: Iterable[str]) -> List[str]:
    """
        Normalize a list of texts safely.

        Args:
            texts: Iterable of raw texts

        Returns:
            List of normalized texts
    """

    config = get_config()

    if config.feature_engineering.enabled:
        return [normalize_medical_text(t or "") for t in texts]

    return [normalize_document_text(t or "") for t in texts]

def safe_strip(text: Optional[str]) -> str:
    """
        Safe strip helper.

        Args:
            text: Optional string

        Returns:
            Stripped string or empty string
    """

    return (text or "").strip()

## ============================================================
## FEATURE ENGINEERING (STRUCTURED FEATURES)
## ============================================================
def build_features(row: Dict[str, Any]) -> Dict[str, Any]:
    """
        Build structured features from input row

        Design:
            - Lightweight feature engineering
            - Complements normalize_medical_text
            - Generic for all pipelines

        Args:
            row: Input dictionary

        Returns:
            Feature dictionary
    """

    features: Dict[str, Any] = {}

    for key, value in row.items():

        value_str = str(value)

        ## TEXT FEATURES
        if isinstance(value, str):
            normalized = value_str.lower()
            features[f"{key}_normalized"] = normalized
            features[f"{key}_length"] = len(normalized)

        ## NUMERIC FEATURES
        if isinstance(value, (int, float)):
            features[f"{key}_scaled"] = value

    logger.debug("Feature engineering done | keys=%s", list(features.keys()))

    return features
    
## ============================================================
## FEATURE STORE OPERATIONS
## ============================================================
def push_features(entity_id: str, features: Dict[str, Any]) -> None:
    """
        Store features into feature store

        Design:
            - Redis for local
            - Feast for production

        Args:
            entity_id: Unique identifier
            features: Feature dictionary

        Returns:
            None
    """

    if FEATURE_STORE_MODE == "redis":

        redis_client.hset(entity_id, mapping=features)
        logger.info("Features stored in Redis | entity_id=%s", entity_id)

    else:

        import pandas as pd

        if FeatureStore is None:
            raise ImportError("Feast is not installed")

        store = FeatureStore(repo_path=FEAST_REPO_PATH)

        df = pd.DataFrame([{**features, "entity_id": entity_id}])

        store.write_to_online_store(df)

        logger.info("Features stored in Feast | entity_id=%s", entity_id)

def get_features(entity_id: str) -> Dict[str, Any]:
    """
        Retrieve features from feature store

        Args:
            entity_id: Unique identifier

        Returns:
            Feature dictionary
    """

    if FEATURE_STORE_MODE == "redis":

        result = redis_client.hgetall(entity_id)
        logger.info("Features retrieved from Redis | entity_id=%s", entity_id)
        return result

    else:

        if FeatureStore is None:
            raise ImportError("Feast is not installed")

        store = FeatureStore(repo_path=FEAST_REPO_PATH)

        result = store.get_online_features(
            features=["features:*"],
            entity_rows=[{"entity_id": entity_id}]
        ).to_dict()

        logger.info("Features retrieved from Feast | entity_id=%s", entity_id)

        return result