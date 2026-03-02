'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Exploratory Data Analysis utilities for structured laboratory data and clustering outputs."
'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from src.core.config import AppConfig
from src.core.errors import DataValidationError, FeatureEngineeringError
from src.utils.io_utils import (
    assert_exists,
    ensure_dir,
    read_csv,
    read_parquet,
    write_csv,
    write_json,
)
from src.utils.logging_utils import get_logger
from src.utils.utils import compute_basic_statistics

logger = get_logger(__name__)
PathLike = Union[str, Path]

## ============================================================
## CONSTANTS
## ============================================================
STRUCTURED_NUMERIC_COLUMNS = [
    "structured_data_origin_value",
    "structured_data_transform_value",
    "norms_min",
    "norms_max",
]

## ============================================================
## DATA CLASSES
## ============================================================
@dataclass(frozen=True)
class EdaResult:
    """
        EDA result container

        Args:
            summary_path: JSON summary path
            status_counts_path: CSV status counts path
            numeric_stats_path: JSON numeric stats path
            message: Human-readable status message

        Returns:
            EdaResult instance
    """

    summary_path: str
    status_counts_path: str
    numeric_stats_path: str
    message: str

## ============================================================
## CORE HELPERS
## ============================================================
def _coerce_numeric_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """
        Coerce selected columns to numeric safely

        High-level workflow:
            1) Iterate over requested columns
            2) If present convert with pandas to_numeric
            3) Keep NaN when conversion fails

        Args:
            df: Input DataFrame
            columns: Columns to coerce

        Returns:
            DataFrame with coerced numeric columns
    """

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _compute_status_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
        Compute status distribution table

        High-level workflow:
            1) Count status values
            2) Sort descending
            3) Reset index into a 2-column DataFrame

        Args:
            df: Structured lab DataFrame

        Returns:
            DataFrame with columns: status, count
    """

    if "status" not in df.columns:
        raise DataValidationError(
            message="Missing required column for EDA",
            details={"missing": ["status"], "context": "core.eda"},
        )

    counts = (
        df["status"]
        .fillna("")
        .astype(str)
        .replace({"nan": ""})
        .value_counts(dropna=False)
        .reset_index()
    )
    
    counts.columns = ["status", "count"]
    
    return counts

def _compute_missingness(df: pd.DataFrame) -> Dict[str, Any]:
    """
        Compute missingness summary

        High-level workflow:
            1) Compute missing counts per column
            2) Compute missing ratios per column
            3) Return dict for JSON export

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with missingness stats
    """

    missing_counts = df.isna().sum().to_dict()
    missing_ratios = (df.isna().mean()).to_dict()

    return {
        "missing_counts": {k: int(v) for k, v in missing_counts.items()},
        "missing_ratios": {k: float(v) for k, v in missing_ratios.items()},
    }

def _dataset_shape(df: pd.DataFrame) -> Dict[str, int]:
    """
        Compute dataset shape

        Args:
            df: Input DataFrame

        Returns:
            Dict with n_rows and n_cols
    """

    return {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])}

## ============================================================
## PUBLIC API
## ============================================================
def run_structured_eda(
    structured_paths: List[PathLike],
    config: AppConfig,
    output_subdir: str = "eda",
) -> EdaResult:
    """
        Run EDA on structured laboratory CSV files

        High-level workflow:
            1) Load and concatenate structured CSV files
            2) Coerce numeric columns when present
            3) Compute basic summaries (shape, status counts, missingness)
            4) Compute basic numeric statistics
            5) Export JSON and CSV artifacts to artifacts/exports/eda

        Args:
            structured_paths: List of paths to structured CSV files
            config: AppConfig instance
            output_subdir: Export subfolder name under artifacts/exports

        Returns:
            EdaResult with artifact paths
    """

    if not structured_paths:
        raise FeatureEngineeringError(
            message="No structured files provided for EDA",
            details={"context": "run_structured_eda"},
        )

    ## Load CSV files and concatenate
    frames: List[pd.DataFrame] = []
    for p in structured_paths:
        path = assert_exists(p, kind="file")
        frames.append(read_csv(path))

    df = pd.concat(frames, axis=0, ignore_index=True)

    ## Coerce numeric columns
    df = _coerce_numeric_columns(df, STRUCTURED_NUMERIC_COLUMNS)

    ## Build export directory
    export_dir = ensure_dir(config.paths.artifacts_exports_dir / output_subdir)

    ## Compute status counts
    status_counts = _compute_status_counts(df)

    ## Compute numeric stats (only for columns that exist)
    numeric_cols_present = [c for c in STRUCTURED_NUMERIC_COLUMNS if c in df.columns]
    numeric_stats = compute_basic_statistics(df, numeric_cols_present)

    ## Summary payload
    summary: Dict[str, Any] = {
        "shape": _dataset_shape(df),
        "n_files": int(len(structured_paths)),
        "columns": list(df.columns),
        "missingness": _compute_missingness(df),
        "status_unique": int(df["status"].nunique()) if "status" in df.columns else 0,
    }

    ## Export artifacts
    summary_path = write_json(export_dir / "structured_eda_summary.json", summary)
    status_counts_path = write_csv(status_counts, export_dir / "structured_status_counts.csv")
    numeric_stats_path = write_json(export_dir / "structured_numeric_stats.json", numeric_stats)

    logger.info(
        "EDA exported | summary=%s | status_counts=%s | numeric_stats=%s",
        summary_path,
        status_counts_path,
        numeric_stats_path,
    )

    return EdaResult(
        summary_path=str(summary_path),
        status_counts_path=str(status_counts_path),
        numeric_stats_path=str(numeric_stats_path),
        message="EDA completed successfully",
    )

def run_dataset_eda(
    dataset_path: PathLike,
    config: AppConfig,
    output_subdir: str = "eda",
) -> EdaResult:
    """
        Run EDA on a clustering dataset (parquet or csv)

        High-level workflow:
            1) Load dataset from parquet or csv
            2) Compute basic summaries (shape, missingness)
            3) Compute basic numeric statistics for all numeric columns
            4) Export JSON and CSV artifacts to artifacts/exports/eda

        Args:
            dataset_path: Path to clustering dataset (csv or parquet)
            config: AppConfig instance
            output_subdir: Export subfolder name under artifacts/exports

        Returns:
            EdaResult with artifact paths
    """

    path = assert_exists(dataset_path, kind="file")
    suffix = path.suffix.lower()

    ## Load dataset
    if suffix == ".parquet":
        df = read_parquet(path)
    elif suffix == ".csv":
        df = read_csv(path)
    else:
        raise DataValidationError(
            message="Unsupported dataset format for EDA",
            details={"path": str(path), "suffix": suffix},
        )

    ## Identify numeric columns
    numeric_cols = list(df.select_dtypes(include=["number"]).columns)

    ## Compute basic stats
    numeric_stats = compute_basic_statistics(df, numeric_cols)

    ## Build export directory
    export_dir = ensure_dir(config.paths.artifacts_exports_dir / output_subdir)

    ## Summary payload
    summary: Dict[str, Any] = {
        "shape": _dataset_shape(df),
        "columns": list(df.columns),
        "numeric_columns": numeric_cols,
        "missingness": _compute_missingness(df),
    }

    ## Optional: cluster distribution if present
    if "cluster" in df.columns:
        try:
            cluster_counts = (
                df["cluster"]
                .fillna("")
                .astype(str)
                .replace({"nan": ""})
                .value_counts(dropna=False)
                .reset_index()
            )
            cluster_counts.columns = ["cluster", "count"]
            write_csv(cluster_counts, export_dir / "dataset_cluster_sizes.csv")
            summary["has_cluster_column"] = True
        except Exception:
            summary["has_cluster_column"] = True
    else:
        summary["has_cluster_column"] = False

    ## Export artifacts
    summary_path = write_json(export_dir / "dataset_eda_summary.json", summary)
    status_counts_path = write_csv(
        pd.DataFrame([], columns=["status", "count"]),
        export_dir / "dataset_status_counts.csv",
    )
    numeric_stats_path = write_json(export_dir / "dataset_numeric_stats.json", numeric_stats)

    logger.info(
        "Dataset EDA exported | summary=%s | numeric_stats=%s",
        summary_path,
        numeric_stats_path,
    )

    return EdaResult(
        summary_path=str(summary_path),
        status_counts_path=str(status_counts_path),
        numeric_stats_path=str(numeric_stats_path),
        message="Dataset EDA completed successfully",
    )