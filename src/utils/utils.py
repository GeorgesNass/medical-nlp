'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Generic utility helpers shared across the doc-classification project."
'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.core.errors import PipelineError


## -----------------------------
## File system helpers
## -----------------------------
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


## -----------------------------
## JSON helpers
## -----------------------------
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


## -----------------------------
## Collection helpers
## -----------------------------
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


## -----------------------------
## Text helpers
## -----------------------------
def safe_str(value: Optional[Any]) -> str:
    """
        Convert a value to string safely.

        Args:
            value: Any value

        Returns:
            String representation (never None)
    """

    return "" if value is None else str(value)
