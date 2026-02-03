'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data utilities: CSV export, JSON I/O, filesystem helpers and generic collection helpers."
'''

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.core.errors import PipelineError
from src.utils.logging_utils import get_logger


## ============================================================
## LOGGER
## ============================================================
logger = get_logger("data_utils")


## ============================================================
## FILESYSTEM HELPERS
## ============================================================
def ensure_dir(path: str | Path) -> Path:
    """
        Ensure a directory exists (mkdir -p behavior).

        Args:
            path: Directory path

        Returns:
            Resolved Path object

        Raises:
            PipelineError: If directory cannot be created
    """

    try:
        p = Path(path).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception as exc:
        raise PipelineError(f"Failed to create directory: {path}") from exc


def ensure_parent_dir(path: str | Path) -> Path:
    """
        Ensure the parent directory of a file path exists.

        Args:
            path: File path

        Returns:
            Parent directory path

        Raises:
            PipelineError: If parent directory cannot be created
    """

    try:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p.parent
    except Exception as exc:
        raise PipelineError(f"Failed to create parent directory for: {path}") from exc


## ============================================================
## JSON HELPERS
## ============================================================
def read_json(path: str | Path) -> Dict[str, Any]:
    """
        Read a JSON file safely.

        Args:
            path: JSON file path

        Returns:
            Parsed JSON dictionary

        Raises:
            PipelineError: If reading or parsing fails
    """

    try:
        p = Path(path).expanduser().resolve()
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise PipelineError(f"Failed to read JSON file: {path}") from exc


def write_json(path: str | Path, data: Dict[str, Any], indent: int = 2) -> None:
    """
        Write a JSON file safely.

        Args:
            path: JSON file path
            data: Data to serialize
            indent: JSON indentation

        Raises:
            PipelineError: If writing fails
    """

    try:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except Exception as exc:
        raise PipelineError(f"Failed to write JSON file: {path}") from exc


## ============================================================
## CSV EXPORT
## ============================================================
def export_dicts_to_csv(
    rows: List[Dict[str, Any]],
    output_path: str | Path,
    delimiter: str = ";",
) -> Path:
    """
        Export a list of dictionaries to a CSV file.

        Notes:
            - Header is inferred from keys of the first row
            - Missing keys are written as empty strings

        Args:
            rows: List of row dictionaries
            output_path: Destination CSV file path
            delimiter: CSV delimiter (default ';')

        Returns:
            Path to written CSV file

        Raises:
            PipelineError: If export fails
    """

    if not rows:
        raise PipelineError("Cannot export empty rows list to CSV")

    try:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(rows[0].keys())

        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                delimiter=delimiter,
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        logger.info(f"CSV exported to {out}")
        return out

    except Exception as exc:
        raise PipelineError(f"Failed to export CSV to {output_path}") from exc


## ============================================================
## COLLECTION HELPERS
## ============================================================
def flatten(list_of_lists: Iterable[Iterable[Any]]) -> List[Any]:
    """
        Flatten a list of lists.

        Args:
            list_of_lists: Iterable of iterables

        Returns:
            Flattened list
    """

    return [item for sub in list_of_lists for item in sub]


def unique_preserve_order(items: Iterable[Any]) -> List[Any]:
    """
        Return unique items while preserving order.

        Args:
            items: Input iterable

        Returns:
            List of unique items
    """

    seen = set()
    unique_items: List[Any] = []

    for item in items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)

    return unique_items


## ============================================================
## TEXT HELPERS
## ============================================================
def safe_str(value: Optional[Any]) -> str:
    """
        Convert a value to string safely.

        Args:
            value: Any value

        Returns:
            String representation (never None)
    """

    return "" if value is None else str(value)
