'''
__author__ = "Olivia Tortosa"
__copyright__ = None
__credits__ = ["Olivia Tortosa", "Georges Nassopoulos"]
__version__ = "1.0.0"
__maintainer__ = "Georges Nassopoulos"
__email__ = "olivia.tortosa@gmail.com", "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Dataset builder for clustering: construct long or wide dataset from structured laboratory CSV files"
'''

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.core.config import AppConfig
from src.core.errors import DatasetBuildError
from src.utils.io_utils import assert_exists, read_csv
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
## INTERNAL HELPERS
## ============================================================
def _load_structured_csv_files(
    config: AppConfig,
    structured_csv_files: Optional[List[str]],
) -> pd.DataFrame:
    """
        Load structured CSV files and concatenate them

        High-level workflow:
            1) Determine file list (all or subset)
            2) Validate existence
            3) Read each CSV
            4) Concatenate into single DataFrame

        Args:
            config: AppConfig instance
            structured_csv_files: Optional subset of filenames

        Returns:
            Concatenated DataFrame
    """

    base_dir = config.paths.interim_structured_dir

    ## Determine file list
    if structured_csv_files:
        file_paths = [base_dir / f for f in structured_csv_files]
    else:
        file_paths = list(base_dir.glob("*.csv"))

    if not file_paths:
        raise DatasetBuildError(
            message="No structured CSV files found",
            details={"directory": str(base_dir)},
        )

    ## Load and concatenate
    frames: List[pd.DataFrame] = []

    for path in file_paths:
        assert_exists(path, kind="file")
        frames.append(read_csv(path))

    df = pd.concat(frames, axis=0, ignore_index=True)

    logger.info(
        "Structured CSV files loaded | files=%s | rows=%s",
        len(file_paths),
        df.shape[0],
    )

    return df

def _build_long_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
        Build long-format dataset

        High-level workflow:
            1) Keep analyte-level rows
            2) Ensure numeric casting of value columns
            3) Return cleaned long dataset

        Args:
            df: Structured DataFrame

        Returns:
            Long-format DataFrame
    """

    ## Select relevant columns
    columns = [
        "file",
        "analyzed_variable",
        "structured_data_transform_value",
        "structured_data_transform_metric",
        "status",
    ]

    existing_columns = [c for c in columns if c in df.columns]
    long_df = df[existing_columns].copy()

    ## Cast numeric value safely
    if "structured_data_transform_value" in long_df.columns:
        long_df["structured_data_transform_value"] = pd.to_numeric(
            long_df["structured_data_transform_value"],
            errors="coerce",
        )

    return long_df

def _build_wide_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
        Build wide-format dataset (pivoted by analyte)

        High-level workflow:
            1) Cast numeric value
            2) Pivot table: index=file, columns=analyte
            3) Aggregate by mean if duplicates
            4) Reset index

        Args:
            df: Structured DataFrame

        Returns:
            Wide-format DataFrame
    """

    ## Ensure numeric casting
    df = df.copy()

    if "structured_data_transform_value" in df.columns:
        df["structured_data_transform_value"] = pd.to_numeric(
            df["structured_data_transform_value"],
            errors="coerce",
        )

    ## Pivot
    wide_df = df.pivot_table(
        index="file",
        columns="analyzed_variable",
        values="structured_data_transform_value",
        aggfunc="mean",
    )

    ## Reset index
    wide_df = wide_df.reset_index()

    logger.info(
        "Wide dataset built | rows=%s | cols=%s",
        wide_df.shape[0],
        wide_df.shape[1],
    )

    return wide_df

## ============================================================
## PUBLIC API
## ============================================================
def build_dataset(
    structured_csv_files: Optional[List[str]],
    dataset_format: str,
    config: AppConfig,
) -> pd.DataFrame:
    """
        Build clustering dataset from structured CSV files

        High-level workflow:
            1) Load structured CSV files
            2) Select dataset format (long or wide)
            3) Return prepared DataFrame

        Args:
            structured_csv_files: Optional subset of structured CSV filenames
            dataset_format: "long" or "wide"
            config: AppConfig instance

        Returns:
            Prepared DataFrame for clustering

        Raises:
            DatasetBuildError: If format invalid
    """

    ## Load structured data
    df = _load_structured_csv_files(config, structured_csv_files)

    ## Build requested format
    if dataset_format == "long":
        return _build_long_dataset(df)

    if dataset_format == "wide":
        return _build_wide_dataset(df)

    raise DatasetBuildError(
        message="Invalid dataset format",
        details={"dataset_format": dataset_format},
    )