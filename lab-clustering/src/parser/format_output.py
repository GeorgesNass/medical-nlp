'''
__author__ = "Olivia Tortosa"
__copyright__ = None
__credits__ = ["Olivia Tortosa", "Georges Nassopoulos"]
__version__ = "1.0.0"
__maintainer__ = "Georges Nassopoulos"
__email__ = "olivia.tortosa@gmail.com", "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Output schema normalization for structured laboratory extraction (stable CSV columns + metadata placeholders)."
'''

from __future__ import annotations

from typing import List

import pandas as pd

from src.core.config import AppConfig
from src.parser.check_norms import compute_status_from_norms
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
## OFFICIAL OUTPUT SCHEMA
## ============================================================
OFFICIAL_COLUMNS: List[str] = [
    "file",
    "gender",
    "sampling_time",
    "dates_dob",
    "dates_edition",
    "analysis_group",
    "analyzed_variable",
    "raw_data_entry",
    "structured_data_origin_value",
    "structured_data_origin_metric",
    "structured_data_transform_value",
    "structured_data_transform_metric",
    "norms_min",
    "norms_max",
    "norms_metric",
    "status",
    "Enfant",
    "Femme",
    "Homme",
    "Metric",
    "normalized_text",
    "token_count",
    "char_length",    
]

## ============================================================
## INTERNAL HELPERS
## ============================================================
def _ensure_column(df: pd.DataFrame, col: str, default_value) -> None:
    """
        Ensure a column exists in the DataFrame

        Args:
            df: DataFrame to mutate
            col: Column name
            default_value: Default value if missing

        Returns:
            None
    """

    if col not in df.columns:
        df[col] = default_value

def _coerce_numeric(df: pd.DataFrame, col: str) -> None:
    """
        Coerce column to numeric when possible

        Args:
            df: DataFrame to mutate
            col: Column name

        Returns:
            None
    """

    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

def _fill_status_if_missing(df: pd.DataFrame) -> None:
    """
        Fill status column if missing or partially missing

        High-level workflow:
            1) Ensure status column exists
            2) For rows where status is missing, compute it from norms_min/norms_max

        Args:
            df: DataFrame to mutate

        Returns:
            None
    """

    _ensure_column(df, "status", "unknown")

    if "structured_data_transform_value" not in df.columns:
        return

    ## Compute only for missing/blank statuses
    mask = df["status"].isna() | (df["status"].astype(str).str.strip() == "")
    if not mask.any():
        return

    ## Apply row-wise computation (safe for small/medium CSVs)
    def _compute(row) -> str:
        return compute_status_from_norms(
            value=row.get("structured_data_transform_value", None),
            norms_min=row.get("norms_min", None),
            norms_max=row.get("norms_max", None),
        )

    df.loc[mask, "status"] = df.loc[mask].apply(_compute, axis=1)

def _derive_norms_flags(df: pd.DataFrame) -> None:
    """
        Create minimal traceability flags for norms source columns

        Notes:
            - These flags are placeholders to keep compatibility with existing CSV schema
            - They can be refined later when we port the full norms logic (child/woman/man)

        Args:
            df: DataFrame to mutate

        Returns:
            None
    """

    _ensure_column(df, "Enfant", "")
    _ensure_column(df, "Femme", "")
    _ensure_column(df, "Homme", "")
    _ensure_column(df, "Metric", "")

    ## If norms are present, set a minimal indicator
    has_norms = df["norms_min"].notna() | df["norms_max"].notna()
    df.loc[has_norms, "Metric"] = df.loc[has_norms, "norms_metric"].astype(str)

def _normalize_metrics(df: pd.DataFrame) -> None:
    """
        Normalize metric columns to lower-case for consistency

        Args:
            df: DataFrame to mutate

        Returns:
            None
    """

    for col in ["structured_data_origin_metric", "structured_data_transform_metric", "norms_metric"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().replace({"nan": ""})

## ============================================================
## PUBLIC API
## ============================================================
def format_structured_output(
    df: pd.DataFrame,
    source_file: str,
    config: AppConfig,
) -> pd.DataFrame:
    """
        Format structured output DataFrame to the official CSV schema

        High-level workflow:
            1) Add mandatory metadata columns (placeholders for now)
            2) Ensure analyte columns exist
            3) Ensure norms columns exist
            4) Fill missing status values
            5) Normalize metrics fields
            6) Reorder columns to official schema

        Args:
            df: Parsed analyte-level DataFrame (may be partially structured)
            source_file: Original TXT filename
            config: AppConfig instance

        Returns:
            DataFrame aligned with OFFICIAL_COLUMNS
    """

    out = df.copy()

    use_fe = getattr(config, "feature_engineering", False)
    
    ## Mandatory metadata columns (placeholders)
    _ensure_column(out, "file", source_file)
    _ensure_column(out, "gender", "")
    _ensure_column(out, "sampling_time", "")
    _ensure_column(out, "dates_dob", "")
    _ensure_column(out, "dates_edition", "")
    _ensure_column(out, "analysis_group", "")

    ## Mandatory analyte columns
    _ensure_column(out, "analyzed_variable", "")
    _ensure_column(out, "raw_data_entry", "")
    _ensure_column(out, "structured_data_origin_value", None)
    _ensure_column(out, "structured_data_origin_metric", "")
    _ensure_column(out, "structured_data_transform_value", None)
    _ensure_column(out, "structured_data_transform_metric", "")

    ## Mandatory norms columns
    _ensure_column(out, "norms_min", None)
    _ensure_column(out, "norms_max", None)

    ## If norms_metric missing, default to transformed metric
    if "norms_metric" not in out.columns:
        out["norms_metric"] = out["structured_data_transform_metric"]

    if use_fe:
        _ensure_column(out, "normalized_text", "")
        _ensure_column(out, "token_count", None)
        _ensure_column(out, "char_length", None)
        
    ## Type normalization
    _coerce_numeric(out, "structured_data_origin_value")
    _coerce_numeric(out, "structured_data_transform_value")
    _coerce_numeric(out, "norms_min")
    _coerce_numeric(out, "norms_max")

    ## Status fill + metric normalization
    _fill_status_if_missing(out)
    _normalize_metrics(out)
    _derive_norms_flags(out)

    ## Reorder columns to official schema
    for col in OFFICIAL_COLUMNS:
        _ensure_column(out, col, "")

    out = out[OFFICIAL_COLUMNS].copy()

    ## Final safety: ensure file column is correct
    out["file"] = source_file

    return out