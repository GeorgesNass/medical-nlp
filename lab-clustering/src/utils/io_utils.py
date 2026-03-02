'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Safe IO helpers for CSV/JSON/Parquet with consistent encoding, path checks, and structured logging."
'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.core.errors import (
    DataValidationError,
    ResourceNotFoundError,
)
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger(__name__)
PathLike = Union[str, Path]

## ============================================================
## PATH HELPERS
## ============================================================
def ensure_parent_dir(path: PathLike) -> Path:
    """
        Ensure the parent directory of a file path exists

        Args:
            path: File path

        Returns:
            Resolved Path
    """

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    
    return resolved

def ensure_dir(path: PathLike) -> Path:
    """
        Ensure a directory exists

        Args:
            path: Directory path

        Returns:
            Resolved Path
    """

    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    
    return resolved

def assert_exists(path: PathLike, kind: str = "file") -> Path:
    """
        Assert a filesystem path exists

        Args:
            path: Path to validate
            kind: "file" or "dir"

        Returns:
            Resolved Path

        Raises:
            ResourceNotFoundError: If path does not exist
    """

    resolved = Path(path).expanduser().resolve()

    if not resolved.exists():
        raise ResourceNotFoundError(
            message=f"Missing {kind}: {resolved}",
            details={"path": str(resolved), "kind": kind},
        )

    if kind == "file" and not resolved.is_file():
        raise ResourceNotFoundError(
            message=f"Expected file but found non-file: {resolved}",
            details={"path": str(resolved), "kind": kind},
        )

    if kind == "dir" and not resolved.is_dir():
        raise ResourceNotFoundError(
            message=f"Expected directory but found non-directory: {resolved}",
            details={"path": str(resolved), "kind": kind},
        )

    return resolved

## ============================================================
## JSON HELPERS
## ============================================================
def read_json(path: PathLike) -> Dict[str, Any]:
    """
        Read a JSON file safely

        Args:
            path: Path to JSON file

        Returns:
            JSON content as dict
    """

    file_path = assert_exists(path, kind="file")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as exc:
        logger.error("Failed to read JSON | path=%s | error=%s", file_path, str(exc))
        logger.debug("Traceback:", exc_info=True)
        raise

def write_json(path: PathLike, data: Dict[str, Any], indent: int = 2) -> Path:
    """
        Write a JSON file safely

        Args:
            path: Output JSON path
            data: JSON-serializable content
            indent: Indentation level

        Returns:
            Output path
    """

    file_path = ensure_parent_dir(path)

    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

        return file_path

    except Exception as exc:
        logger.error("Failed to write JSON | path=%s | error=%s", file_path, str(exc))
        logger.debug("Traceback:", exc_info=True)
        raise

## ============================================================
## CSV HELPERS
## ============================================================
def read_csv(
    path: PathLike,
    sep: str = ",",
    encoding: str = "utf-8",
    dtype: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
        Read a CSV file safely

        Args:
            path: Path to CSV
            sep: Separator
            encoding: File encoding
            dtype: Optional dtype mapping

        Returns:
            Loaded DataFrame
    """

    file_path = assert_exists(path, kind="file")

    try:
        return pd.read_csv(
            file_path,
            sep=sep,
            encoding=encoding,
            dtype=dtype,
        )

    except Exception as exc:
        logger.error("Failed to read CSV | path=%s | error=%s", file_path, str(exc))
        logger.debug("Traceback:", exc_info=True)
        raise

def write_csv(
    df: pd.DataFrame,
    path: PathLike,
    sep: str = ",",
    encoding: str = "utf-8",
    index: bool = False,
) -> Path:
    """
        Write a DataFrame to CSV safely

        Args:
            df: DataFrame to write
            path: Output CSV path
            sep: Separator
            encoding: File encoding
            index: Whether to include index

        Returns:
            Output path
    """

    file_path = ensure_parent_dir(path)

    try:
        df.to_csv(
            file_path,
            sep=sep,
            encoding=encoding,
            index=index,
        )
        return file_path

    except Exception as exc:
        logger.error("Failed to write CSV | path=%s | error=%s", file_path, str(exc))
        logger.debug("Traceback:", exc_info=True)
        raise

## ============================================================
## PARQUET HELPERS
## ============================================================
def read_parquet(path: PathLike) -> pd.DataFrame:
    """
        Read a Parquet file safely

        Args:
            path: Path to Parquet file

        Returns:
            Loaded DataFrame
    """

    file_path = assert_exists(path, kind="file")

    try:
        return pd.read_parquet(file_path)

    except Exception as exc:
        logger.error("Failed to read Parquet | path=%s | error=%s", file_path, str(exc))
        logger.debug("Traceback:", exc_info=True)
        raise

def write_parquet(df: pd.DataFrame, path: PathLike, index: bool = False) -> Path:
    """
        Write a DataFrame to Parquet safely

        Args:
            df: DataFrame to write
            path: Output Parquet path
            index: Whether to include index

        Returns:
            Output path
    """

    file_path = ensure_parent_dir(path)

    try:
        df.to_parquet(file_path, index=index)
        return file_path

    except Exception as exc:
        logger.error("Failed to write Parquet | path=%s | error=%s", file_path, str(exc))
        logger.debug("Traceback:", exc_info=True)
        raise

## ============================================================
## DATA VALIDATION HELPERS
## ============================================================
def assert_required_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    context: str,
) -> None:
    """
        Validate required columns are present in a DataFrame

        Args:
            df: DataFrame to validate
            required_cols: List of required columns
            context: Validation context (for logs/errors)

        Raises:
            DataValidationError: If missing columns
    """

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise DataValidationError(
            message=f"Missing required columns in {context}",
            details={"missing": missing, "context": context},
        )

def assert_non_empty_df(df: pd.DataFrame, context: str) -> None:
    """
        Validate DataFrame is non-empty

        Args:
            df: DataFrame to validate
            context: Validation context

        Raises:
            DataValidationError: If DataFrame is empty
    """

    if df is None or df.empty:
        raise DataValidationError(
            message=f"Empty DataFrame in {context}",
            details={"context": context},
        )