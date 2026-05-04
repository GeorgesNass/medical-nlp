'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Generic utility helpers to keep modules lightweight and below 400 lines."
'''

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.core.errors import DataValidationError

## ============================================================
## STRING UTILITIES
## ============================================================
def normalize_text(text: Optional[str]) -> str:
    """
        Normalize text for consistent downstream processing

        High-level workflow:
            1) Handle None input
            2) Normalize unicode
            3) Remove accents
            4) Lowercase
            5) Collapse whitespace

        Args:
            text: Input string that may be None

        Returns:
            Normalized string
    """

    if text is None:
        return ""

    ## Normalize unicode and remove accents
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))

    ## Normalize casing and whitespace
    normalized = normalized.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized

## ============================================================
## FEATURE ENGINEERING - TEXT NORMALIZATION
## ============================================================
def normalize_clinical_text(text: str) -> str:
    """
        Normalize clinical / lab text

        High-level workflow:
            1) Lowercase
            2) Remove extra spaces
            3) Remove special characters (keep units/numbers)

        Args:
            text: Input string

        Returns:
            Normalized text
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\-/%]", " ", text)

    return text.strip()
    
def safe_strip(value: Any) -> str:
    """
        Convert a value to string safely and strip surrounding whitespace

        High-level workflow:
            1) Handle None explicitly
            2) Convert to string
            3) Strip spaces

        Args:
            value: Any input value

        Returns:
            Cleaned string value
    """

    if value is None:
        return ""

    return str(value).strip()

## ============================================================
## NUMERIC UTILITIES
## ============================================================
def safe_float(value: Any) -> Optional[float]:
    """
        Convert a value to float safely

        High-level workflow:
            1) Handle None explicitly
            2) Normalize decimal separator
            3) Attempt float conversion
            4) Return None on failure

        Args:
            value: Any input value

        Returns:
            Float value if conversion succeeds otherwise None
    """

    if value is None:
        return None

    try:
        normalized = str(value).replace(",", ".")
        return float(normalized)
    except (ValueError, TypeError):
        return None

def safe_int(value: Any) -> Optional[int]:
    """
        Convert a value to int safely

        High-level workflow:
            1) Handle None explicitly
            2) Attempt int conversion
            3) Return None on failure

        Args:
            value: Any input value

        Returns:
            Integer value if conversion succeeds otherwise None
    """

    if value is None:
        return None

    try:
        return int(value)
    except (ValueError, TypeError):
        return None

## ============================================================
## DATAFRAME UTILITIES
## ============================================================
def enforce_column_order(
    df: pd.DataFrame,
    ordered_columns: List[str],
) -> pd.DataFrame:
    """
        Reorder a DataFrame columns while keeping any extra columns

        High-level workflow:
            1) Keep requested columns that exist
            2) Append remaining columns at the end
            3) Return reordered DataFrame

        Args:
            df: Input DataFrame
            ordered_columns: Desired column order

        Returns:
            Reordered DataFrame
    """

    ## Keep only existing columns
    existing = [c for c in ordered_columns if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    
    return df[existing + remaining]

def fillna_with_empty_string(df: pd.DataFrame) -> pd.DataFrame:
    """
        Replace missing values with empty string

        High-level workflow:
            1) Replace NaN values with empty string
            2) Return cleaned DataFrame

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with NaN replaced by empty string
    """

    return df.fillna("")

def validate_unique_column(
    df: pd.DataFrame,
    column: str,
    context: str,
) -> None:
    """
        Validate that a column has unique values

        High-level workflow:
            1) Check column exists
            2) Detect duplicates
            3) Raise DataValidationError on duplicates

        Args:
            df: DataFrame to validate
            column: Column name to enforce uniqueness
            context: Human-readable validation context

        Returns:
            None

        Raises:
            DataValidationError: If column missing or duplicates found
    """

    ## Validate column existence
    if column not in df.columns:
        raise DataValidationError(
            message=f"Column '{column}' not found in {context}",
            details={"column": column, "context": context},
        )

    ## Compute duplicate count
    duplicates = df[column].duplicated().sum()

    ## Raise error if duplicates found
    if duplicates > 0:
        raise DataValidationError(
            message=f"Duplicate values found in column '{column}'",
            details={
                "column": column,
                "duplicates": int(duplicates),
                "context": context,
            },
        )

## ============================================================
## STATISTICAL UTILITIES
## ============================================================
def compute_basic_statistics(
    df: pd.DataFrame,
    numeric_columns: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    """
        Compute basic summary statistics for numeric columns

        High-level workflow:
            1) Iterate over numeric columns
            2) Convert to numeric safely
            3) Compute mean, std, min, max, median
            4) Skip empty columns

        Args:
            df: Input DataFrame
            numeric_columns: Iterable of numeric column names

        Returns:
            Dictionary mapping column -> statistics dictionary
    """

    stats: Dict[str, Dict[str, float]] = {}

    ## Iterate over numeric columns
    for col in numeric_columns:

        ## Skip missing columns
        if col not in df.columns:
            continue

        ## Convert to numeric safely
        series = pd.to_numeric(df[col], errors="coerce").dropna()

        ## Skip empty series
        if series.empty:
            continue

        ## Compute statistics
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
        }

    return stats

def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """
        Detect outliers using the IQR rule

        High-level workflow:
            1) Convert series to numeric safely
            2) Compute Q1, Q3 and IQR
            3) Compute lower and upper bounds
            4) Return boolean mask of outliers

        Args:
            series: Input pandas Series

        Returns:
            Boolean Series mask indicating outliers
    """

    ## Convert safely to numeric
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()

    ## Return empty mask if no data
    if numeric_series.empty:
        return pd.Series(dtype=bool)

    ## Compute quartiles
    q1 = np.percentile(numeric_series, 25)
    q3 = np.percentile(numeric_series, 75)

    ## Compute IQR
    iqr = q3 - q1

    ## Define bounds
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    ## Return boolean mask
    return (numeric_series < lower) | (numeric_series > upper)