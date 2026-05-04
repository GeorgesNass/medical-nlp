'''
__author__ = "Olivia Tortosa"
__copyright__ = None
__credits__ = ["Olivia Tortosa", "Georges Nassopoulos"]
__version__ = "1.0.0"
__maintainer__ = "Georges Nassopoulos"
__email__ = "olivia.tortosa@gmail.com", "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Resource loader for laboratory parsing: regex patterns, keyword dictionaries, norms tables and unit conversion tables"
'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.core.config import AppConfig
from src.core.errors import (
    NormsLoadingError,
    RegexLoadingError,
    ResourceNotFoundError,
)
from src.utils.utils import normalize_clinical_text
from src.utils.io_utils import assert_exists, read_csv, read_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
## CONSTANTS
## ============================================================
DEFAULT_REGEX_FILENAMES = [
    "regex_canonique_unifie.json",
]

DEFAULT_KEYWORDS_FILENAMES = [
    "keywords_canonique_unifie.json",
]

DEFAULT_NORMS_FILENAMES = [
    "normes_canonique_unifie.csv",
]

DEFAULT_CONVERSION_FILENAMES = [
    "conversion_canonique_unifie.csv",
]

## ============================================================
## DATA CLASSES
## ============================================================
@dataclass(frozen=True)
class ParserResources:
    """
        Container for parser resources

        Args:
            regex: Regex configuration dictionary
            keywords: Keyword dictionary used by parser (category -> analyte_key -> config)
            norms_tables: Mapping category -> norms DataFrame
            conversion_table: Optional conversion DataFrame
            conversion_factors: Optional conversion factors DataFrame

        Returns:
            ParserResources instance
    """

    regex: Dict[str, Any]
    keywords: Dict[str, Dict[str, Any]]
    norms_tables: Dict[str, pd.DataFrame]
    conversion_table: Optional[pd.DataFrame]
    conversion_factors: Optional[pd.DataFrame]

## ============================================================
## INTERNAL CACHE
## ============================================================
_CACHE: Dict[str, ParserResources] = {}

## ============================================================
## PATH HELPERS
## ============================================================
def _resources_dir(config: AppConfig) -> Path:
    """
        Resolve resources directory path

        Args:
            config: AppConfig instance

        Returns:
            Path to artifacts/resources directory
    """

    return config.paths.artifacts_resources_dir

def _find_required(resources_dir: Path, filename: str) -> Path:
    """
        Resolve a required resource file path

        Args:
            resources_dir: Base resources directory
            filename: Expected filename

        Returns:
            Full path

        Raises:
            ResourceNotFoundError: If file does not exist
    """

    path = resources_dir / filename

    if not path.exists() or not path.is_file():
        raise ResourceNotFoundError(
            message="Required resource file not found",
            details={
                "resources_dir": str(resources_dir),
                "filename": filename,
                "expected_path": str(path),
            },
        )

    return path

## ============================================================
## LOADERS
## ============================================================
def load_regex_config(config: AppConfig) -> Dict[str, Any]:
    """
        Load regex configuration JSON file

        High-level workflow:
            1) Resolve resources directory
            2) Load regex_canonique_unifie.json
            3) Flatten canonical structure to {key: pattern}

        Args:
            config: AppConfig instance

        Returns:
            Regex configuration dictionary (flattened)

        Raises:
            RegexLoadingError: If loading fails
    """

    resources_dir = _resources_dir(config)

    try:
        ## Locate required file
        regex_path = _find_required(resources_dir, DEFAULT_REGEX_FILENAMES[0])

        ## Ensure file exists
        assert_exists(regex_path, kind="file")

        ## Load JSON
        payload = read_json(regex_path)

        ## Canonical format: {key: {value: "...", ...}} -> flatten to {key: value}
        if isinstance(payload, dict):
            flattened: Dict[str, Any] = {}
            for key, val in payload.items():
                if isinstance(val, dict) and "value" in val:
                    flattened[key] = val.get("value", "")
                else:
                    flattened[key] = val
            return flattened

        raise ValueError("Invalid regex payload format: expected dict")

    except Exception as exc:
        logger.error("Regex loading failed | error=%s", str(exc))
        logger.debug("Traceback:", exc_info=True)

        raise RegexLoadingError(
            message="Failed to load regex configuration",
            details={
                "resources_dir": str(resources_dir),
                "error": str(exc),
            },
        )

def load_keywords(config: AppConfig) -> Dict[str, Dict[str, Any]]:
    """
        Load keywords JSON dictionary

        High-level workflow:
            1) Resolve resources directory
            2) Load keywords_canonique_unifie.json
            3) Convert list format to {category: {analyte_key: {...}}}

        Args:
            config: AppConfig instance

        Returns:
            Keywords dictionary (category -> analyte_key -> config)

        Raises:
            RegexLoadingError: If loading fails
    """

    resources_dir = _resources_dir(config)

    try:
        ## Locate required file
        keywords_path = _find_required(resources_dir, DEFAULT_KEYWORDS_FILENAMES[0])

        ## Ensure file exists
        assert_exists(keywords_path, kind="file")

        ## Load JSON
        payload = read_json(keywords_path)

        ## Canonical file can be either:
        ##  - list[dict] (preferred canonical export)
        ##  - dict[category][analyte_key] (already mapped)
        if isinstance(payload, dict):
            ## Validate minimal structure
            mapped_dict: Dict[str, Dict[str, Any]] = {}
            for category, analytes in payload.items():
                if not isinstance(analytes, dict):
                    continue
                mapped_dict[str(category).strip()] = analytes
            return mapped_dict

        mapped: Dict[str, Dict[str, Any]] = {}

        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue

                category = str(item.get("category", "unknown")).strip()
                analyte_key = str(item.get("analyte_key", "")).strip()
                analyte_key = normalize_clinical_text(analyte_key)

                if not analyte_key:
                    continue

                mapped.setdefault(category, {})
                mapped[category][analyte_key] = {
                    "regex": str(item.get("regex", "")).strip(),
                    #"display_name": item.get("display_name", analyte_key),
                    "display_name" = normalize_clinical_text(str(item.get("display_name", analyte_key)))
                    "has_valeur": bool(item.get("has_valeur", False)),
                    "valeur": item.get("valeur", None),
                }

            return mapped

        raise ValueError("Invalid keywords payload format: expected list or dict")

    except Exception as exc:
        logger.error("Keywords loading failed | error=%s", str(exc))
        logger.debug("Traceback:", exc_info=True)

        raise RegexLoadingError(
            message="Failed to load keywords dictionary",
            details={
                "resources_dir": str(resources_dir),
                "error": str(exc),
            },
        )

def load_norms_tables(config: AppConfig) -> Dict[str, pd.DataFrame]:
    """
        Load canonical norms CSV file

        High-level workflow:
            1) Resolve resources directory
            2) Load normes_canonique_unifie.csv
            3) Return mapping {"canonique": df}

        Args:
            config: AppConfig instance

        Returns:
            Mapping "canonique" -> norms DataFrame

        Raises:
            NormsLoadingError: If loading fails
    """

    resources_dir = _resources_dir(config)

    try:
        ## Locate required file
        norms_path = _find_required(resources_dir, DEFAULT_NORMS_FILENAMES[0])

        ## Ensure file exists
        assert_exists(norms_path, kind="file")

        ## Read CSV
        df = read_csv(
            norms_path,
            sep=",",
            encoding="utf-8",
        )

        return {"canonique": df}

    except Exception as exc:
        logger.error("Norms loading failed | error=%s", str(exc))
        logger.debug("Traceback:", exc_info=True)

        raise NormsLoadingError(
            message="Failed to load norms tables",
            details={
                "resources_dir": str(resources_dir),
                "error": str(exc),
            },
        )

def load_conversion_tables(
    config: AppConfig,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
        Load canonical unit conversion table

        High-level workflow:
            1) Resolve resources directory
            2) Load conversion_canonique_unifie.csv
            3) Return (conversion_table, None)

        Args:
            config: AppConfig instance

        Returns:
            Tuple (conversion_table, None)
    """

    resources_dir = _resources_dir(config)

    conversion_table: Optional[pd.DataFrame] = None
    conversion_factors: Optional[pd.DataFrame] = None

    try:
        ## Locate required file
        conversion_path = _find_required(resources_dir, DEFAULT_CONVERSION_FILENAMES[0])

        ## Ensure file exists
        assert_exists(conversion_path, kind="file")

        conversion_table = read_csv(
            conversion_path,
            sep=",",
            encoding="utf-8",
        )

    except Exception as exc:
        logger.warning("Failed to load conversion table | error=%s", str(exc))

    return conversion_table, conversion_factors

## ============================================================
## PUBLIC API
## ============================================================
def load_parser_resources(
    config: AppConfig,
    cache_key: str = "default",
) -> ParserResources:
    """
        Load and cache all parser resources

        High-level workflow:
            1) Check in-memory cache
            2) Load regex configuration
            3) Load keywords dictionary
            4) Load norms tables
            5) Load conversion tables
            6) Cache and return resources

        Args:
            config: AppConfig instance
            cache_key: Cache key identifier

        Returns:
            ParserResources instance
    """

    ## Return cached resources if available
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    ## Load individual resource components
    regex = load_regex_config(config)
    keywords = load_keywords(config)
    norms_tables = load_norms_tables(config)
    conversion_table, conversion_factors = load_conversion_tables(config)

    resources = ParserResources(
        regex=regex,
        keywords=keywords,
        norms_tables=norms_tables,
        conversion_table=conversion_table,
        conversion_factors=conversion_factors,
    )

    ## Store in cache for reuse
    _CACHE[cache_key] = resources

    logger.info(
        "Parser resources loaded | norms_tables=%s | has_conversion_table=%s | has_conversion_factors=%s",
        len(norms_tables),
        conversion_table is not None,
        conversion_factors is not None,
    )

    return resources