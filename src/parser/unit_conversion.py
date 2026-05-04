'''
__author__ = "Olivia Tortosa"
__copyright__ = None
__credits__ = ["Olivia Tortosa", "Georges Nassopoulos"]
__version__ = "1.0.0"
__maintainer__ = "Georges Nassopoulos"
__email__ = "olivia.tortosa@gmail.com", "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unit normalization and optional conversion logic based on conversion tables"
'''

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from src.core.config import AppConfig
from src.parser.regex_store import ParserResources
from src.utils.logging_utils import get_logger
from src.utils.utils import safe_float, safe_strip

logger = get_logger(__name__)

## ============================================================
## INTERNAL HELPERS
## ============================================================
def _normalize_string_unit(unit: Optional[str]) -> Optional[str]:
    """
        Normalize unit string formatting

        High-level workflow:
            1) Handle None
            2) Strip whitespace
            3) Lowercase
            4) Remove redundant spaces

        Args:
            unit: Raw unit string

        Returns:
            Normalized unit string
    """

    ## Handle None safely
    if unit is None:
        return None

    ## Normalize casing and spacing
    normalized = safe_strip(unit).lower()
    normalized = normalized.replace(" ", "")

    return normalized

def _coerce_conversion_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
        Coerce canonical conversion table schema to a normalized internal schema

        Canonical expected columns (conversion_canonique_unifie.csv):
            - Type
            - In
            - Out
            - Factor

        Internal normalized columns:
            - analyte_type
            - unit_in
            - unit_out
            - factor

        Args:
            df: Raw conversion DataFrame

        Returns:
            Normalized conversion DataFrame
    """

    out = df.copy()

    ## Normalize column names defensively
    rename_map = {
        "Type": "analyte_type",
        "In": "unit_in",
        "Out": "unit_out",
        "Factor": "factor",
    }

    for src, dst in rename_map.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})

    ## Ensure minimal columns exist
    for col in ["analyte_type", "unit_in", "unit_out", "factor"]:
        if col not in out.columns:
            out[col] = None

    ## Normalize strings
    out["analyte_type"] = out["analyte_type"].astype(str).str.strip().str.lower()
    out["unit_in"] = out["unit_in"].astype(str).str.strip().str.lower().str.replace(" ", "", regex=False)
    out["unit_out"] = out["unit_out"].astype(str).str.strip().str.lower().str.replace(" ", "", regex=False)

    ## Factor to float
    out["factor"] = out["factor"].apply(safe_float)

    return out

def _lookup_unit_mapping(
    analyte: str,
    unit: str,
    resources: ParserResources,
) -> Tuple[Optional[str], Optional[float]]:
    """
        Lookup unit normalization + conversion factor from canonical conversion table

        High-level workflow:
            1) Validate conversion_table presence
            2) Normalize canonical schema
            3) Match analyte_type == analyte and unit_in == unit
            4) Return (unit_out, factor)

        Args:
            analyte: Analyte key (Type)
            unit: Normalized unit string
            resources: ParserResources container

        Returns:
            Tuple (unit_out, factor) or (None, None)
    """

    if resources.conversion_table is None:
        return None, None

    df: pd.DataFrame = resources.conversion_table

    ## Coerce schema for canonical file
    coerced = _coerce_conversion_schema(df)

    analyte_key = safe_strip(analyte).lower()
    unit_key = safe_strip(unit).lower().replace(" ", "")

    ## Filter by analyte + unit_in
    matches = coerced[
        (coerced["analyte_type"] == analyte_key)
        & (coerced["unit_in"] == unit_key)
    ]

    if matches.empty:
        return None, None

    row = matches.iloc[0]
    unit_out = safe_strip(str(row.get("unit_out", ""))) or None
    factor = row.get("factor", None)

    return unit_out, factor

## ============================================================
## PUBLIC API
## ============================================================
def normalize_unit(
    unit: Optional[str],
    analyte: str,
    resources: ParserResources,
    config: Optional[AppConfig] = None,
) -> Optional[str]:
    """
        Normalize and optionally convert unit based on conversion tables

        High-level workflow:
            1) Normalize raw unit string
            2) Apply feature engineering normalization (e.g., micro symbol handling)
            3) Attempt mapping using canonical conversion table (Type/In/Out/Factor)
            4) Fallback to normalized unit

        Args:
            unit: Raw unit string
            analyte: Analyte name (used for Type matching in conversion table)
            resources: ParserResources container
            config: Optional AppConfig to enable feature engineering

        Returns:
            Normalized unit string
    """

    use_fe = getattr(config, "feature_engineering", False) if config else False
    
    ## Normalize raw unit string
    normalized_unit = _normalize_string_unit(unit)

    if use_fe and normalized_unit:
        ## Extra normalization for noisy lab units
        normalized_unit = normalized_unit.replace("µ", "u")
        normalized_unit = normalized_unit.replace("μ", "u")
        
    if normalized_unit is None:
        return None

    ## Attempt mapping via canonical conversion_table
    unit_out, _factor = _lookup_unit_mapping(
        analyte=analyte,
        unit=normalized_unit,
        resources=resources,
    )

    if unit_out is not None:
        return unit_out

    ## Fallback to normalized unit
    return normalized_unit