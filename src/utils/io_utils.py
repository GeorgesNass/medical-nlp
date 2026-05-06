'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Safe IO helpers for CSV/JSONL/Parquet, with path validation and consistent encoding."
'''

from __future__ import annotations

## Standard library imports
## Third-party imports
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
import os
import redis

try:
    from feast import FeatureStore
except Exception:
    FeatureStore = None

## Internal imports
from src.core.config import get_config
from src.nlp.preprocess import normalize_medical_text
from src.utils.logging_utils import get_logger
from src.core.errors import (
    log_and_raise_missing_file,
    log_and_raise_missing_folder,
)

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
logger = get_logger("io_utils", log_file="io_utils.log")

## ============================================================
## PATH HELPERS
## ============================================================
def ensure_dir(path: str | Path) -> Path:
    """
        Ensure directory exists

        Args:
            path: Directory path

        Returns:
            Resolved directory path
    """
    
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    
    return p

def ensure_parent_dir(file_path: str | Path) -> Path:
    """
        Ensure parent directory exists for a file path

        Args:
            file_path: File path

        Returns:
            Resolved file path
    """
    
    p = Path(file_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    
    return p

def validate_file_exists(file_path: str | Path) -> Path:
    """
        Validate that a file exists

        Args:
            file_path: File path

        Returns:
            Resolved file path

        Raises:
            DataError: If file does not exist
    """
    
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        log_and_raise_missing_file(p)
    
    return p

def validate_folder_exists(folder_path: str | Path) -> Path:
    """
        Validate that a folder exists

        Args:
            folder_path: Folder path

        Returns:
            Resolved folder path

        Raises:
            DataError: If folder does not exist
    """
    
    p = Path(folder_path).expanduser().resolve()
    if not p.exists():
        log_and_raise_missing_folder(p)
    
    return p

## ============================================================
## CSV HELPERS
## ============================================================
def read_csv(
    csv_path: str | Path,
    encoding: str = "utf-8",
    sep: str = ",",
) -> pd.DataFrame:
    """
        Read a CSV file safely

        Args:
            csv_path: Path to CSV file
            encoding: File encoding
            sep: CSV separator

        Returns:
            DataFrame
    """
    
    path = validate_file_exists(csv_path)
    logger.info("Reading CSV: %s", path)
    
    return pd.read_csv(path, encoding=encoding, sep=sep)

def write_csv(
    df: pd.DataFrame,
    csv_path: str | Path,
    encoding: str = "utf-8",
    index: bool = False,
) -> Path:
    """
        Write a DataFrame to CSV safely

        Args:
            df: DataFrame to write
            csv_path: Output CSV path
            encoding: File encoding
            index: Whether to write index

        Returns:
            Resolved output path
    """
    
    path = ensure_parent_dir(csv_path)
    df.to_csv(path, index=index, encoding=encoding)
    logger.info("Wrote CSV: %s | rows=%d cols=%d", path, len(df), len(df.columns))
    
    return path

## ============================================================
## JSON HELPERS
## ============================================================
def read_json(json_path: str | Path, encoding: str = "utf-8") -> Dict[str, Any]:
    """
        Read a JSON file into dict

        Args:
            json_path: Path to JSON file
            encoding: File encoding

        Returns:
            Parsed JSON dict
    """
    
    path = validate_file_exists(json_path)
    logger.info("Reading JSON: %s", path)

    with path.open("r", encoding=encoding) as f:
        return json.load(f)

def write_json(
    payload: Dict[str, Any],
    json_path: str | Path,
    encoding: str = "utf-8",
    indent: int = 2,
) -> Path:
    """
        Write a dict to JSON file

        Args:
            payload: JSON-serializable dict
            json_path: Output JSON path
            encoding: File encoding
            indent: Indentation for readability

        Returns:
            Resolved output path
    """
    
    path = ensure_parent_dir(json_path)

    with path.open("w", encoding=encoding) as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)

    logger.info("Wrote JSON: %s", path)
    
    return path

## ============================================================
## JSONL HELPERS
## ============================================================
def read_jsonl(jsonl_path: str | Path, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    """
        Read a JSONL file into list of dicts

        Args:
            jsonl_path: Path to JSONL file
            encoding: File encoding

        Returns:
            List of JSON objects
    """
    
    path = validate_file_exists(jsonl_path)
    logger.info("Reading JSONL: %s", path)

    items: List[Dict[str, Any]] = []

    with path.open("r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    return items

def write_jsonl(
    items: Iterable[Dict[str, Any]],
    jsonl_path: str | Path,
    encoding: str = "utf-8",
) -> Path:
    """
        Write an iterable of dicts to JSONL

        Args:
            items: Iterable of JSON-serializable dicts
            jsonl_path: Output JSONL path
            encoding: File encoding

        Returns:
            Resolved output path
    """
    
    path = ensure_parent_dir(jsonl_path)

    count = 0
    with path.open("w", encoding=encoding) as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Wrote JSONL: %s | rows=%d", path, count)
    
    return path

## ============================================================
## PARQUET HELPERS
## ============================================================
def read_parquet(parquet_path: str | Path) -> pd.DataFrame:
    """
        Read a Parquet file safely

        Args:
            parquet_path: Path to parquet file

        Returns:
            DataFrame
    """
    
    path = validate_file_exists(parquet_path)
    logger.info("Reading Parquet: %s", path)
    
    return pd.read_parquet(path)

def write_parquet(
    df: pd.DataFrame,
    parquet_path: str | Path,
    index: bool = False,
) -> Path:
    """
        Write DataFrame to Parquet safely

        Args:
            df: DataFrame
            parquet_path: Output parquet path
            index: Whether to include index

        Returns:
            Resolved output path
    """
   
    path = ensure_parent_dir(parquet_path)
    df.to_parquet(path, index=index)
    logger.info("Wrote Parquet: %s | rows=%d cols=%d", path, len(df), len(df.columns))
    
    return path

## ============================================================
## TEXT HELPERS
## ============================================================
def read_text(file_path: str | Path, encoding: str = "utf-8") -> str:
    """
        Read text file safely

        Args:
            file_path: Path to file
            encoding: File encoding

        Returns:
            File content
    """
    
    path = validate_file_exists(file_path)
    logger.info("Reading text: %s", path)
    
    return path.read_text(encoding=encoding, errors="ignore")

def write_text(
    content: str,
    file_path: str | Path,
    encoding: str = "utf-8",
) -> Path:
    """
        Write text content safely

        Args:
            content: Text content
            file_path: Output file path
            encoding: File encoding

        Returns:
            Resolved output path
    """
    
    path = ensure_parent_dir(file_path)
    path.write_text(content, encoding=encoding)
    logger.info("Wrote text: %s | chars=%d", path, len(content))
    
    return path
    
## ============================================================
## FEATURE ENGINEERING IO HELPERS
## ============================================================
def load_and_normalize_text_from_path(
    file_path: str | Path,
    encoding: str = "utf-8",
) -> str:
    """
        Load and normalize text using feature engineering pipeline

        Args:
            file_path: Input file path
            encoding: File encoding

        Returns:
            Normalized text ready for pipeline
    """

    ## Load raw text
    text = read_text(file_path, encoding=encoding)

    ## Apply feature engineering if enabled
    config = get_config()

    if config.feature_engineering.enabled:
        return normalize_medical_text(text)

    return text    
    
## ============================================================
## FEATURE ENGINEERING (STORE READY)
## ============================================================
def build_features(row: Dict[str, Any]) -> Dict[str, Any]:
    """
        Build structured features from input row

        Design:
            - Lightweight feature engineering
            - Compatible with all pipelines
            - Uses normalized text when available

        Args:
            row: Input dictionary

        Returns:
            Feature dictionary
    """

    features: Dict[str, Any] = {}

    for key, value in row.items():

        ## Convert safely to string
        value_str = str(value)

        ## TEXT FEATURES
        if isinstance(value, str):

            normalized = value_str.lower()

            features[f"{key}_normalized"] = normalized
            features[f"{key}_length"] = len(normalized)

        ## NUMERIC FEATURES
        if isinstance(value, (int, float)):

            features[f"{key}_scaled"] = value

    logger.debug("Features built | keys=%s", list(features.keys()))

    return features
    
## ============================================================
## FEATURE STORE (REDIS + FEAST)
## ============================================================
def push_features(entity_id: str, features: Dict[str, Any]) -> None:
    """
        Store features in feature store

        Design:
            - Redis for local mode
            - Feast for production mode
            - Controlled via FEATURE_STORE_MODE

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

        Design:
            - Unified interface (Redis / Feast)

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