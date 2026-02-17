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
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

## Third-party imports
import pandas as pd

## Internal imports
from src.utils.logging_utils import get_logger
from src.core.errors import (
    log_and_raise_missing_file,
    log_and_raise_missing_folder,
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