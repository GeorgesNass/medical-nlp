'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Exploratory Data Analysis utilities for ICD10 datasets: label distribution, text statistics and basic diagnostics."
'''

from __future__ import annotations

## ============================================================
## IMPORTS
## ============================================================

## Standard library imports
from pathlib import Path
from typing import Dict

## Third-party imports
import pandas as pd
import matplotlib.pyplot as plt

## Internal imports
from src.utils.logging_utils import get_logger
from src.utils.io_utils import ensure_parent_dir

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("eda", log_file="eda.log") ## All EDA-related steps (plots, statistics, exports) are traced here

## ============================================================
## LABEL DISTRIBUTION ANALYSIS
## ============================================================
def compute_label_distribution(
    clinical_csv_dir: str | Path,
) -> pd.DataFrame:
    """
        Compute primary diagnosis frequency distribution

        Logic:
            - Iterate over each admission CSV file
            - Extract primary_diagnosis_code (first row per file)
            - Count frequency of each ICD10 code

        Args:
            clinical_csv_dir: Folder with per-admission CSV files

        Returns:
            DataFrame with columns:
                - primary_diagnosis_code
                - count
    """

    logger.info("Computing label distribution")

    ## Collect all CSV files (1 per admission_id)
    csv_dir = Path(clinical_csv_dir)
    csv_files = sorted(csv_dir.glob("*.csv"))

    labels = []

    for file_path in csv_files:
        df = pd.read_csv(file_path)

        if "primary_diagnosis_code" in df.columns:

            ## By design, we take the first diagnosis code per admission
            ## (one admission = one target label)
            labels.append(df["primary_diagnosis_code"].iloc[0])

    ## Convert list to frequency table
    distribution = (
        pd.Series(labels)
        .value_counts()
        .reset_index()
        .rename(columns={"index": "primary_diagnosis_code", 0: "count"})
    )

    logger.info("Label distribution computed | unique_labels=%d", len(distribution))

    return distribution

def plot_label_distribution(
    distribution_df: pd.DataFrame,
    output_path: str | Path,
    top_k: int = 20,
) -> Path:
    """
        Plot top-k most frequent ICD10 codes

        Args:
            distribution_df: Label distribution DataFrame
            output_path: Path to save plot image
            top_k: Number of top labels to display

        Returns:
            Saved plot path
    """

    logger.info("Plotting label distribution | top_k=%d", top_k)

    df_top = distribution_df.head(top_k)
    
    plt.figure(figsize=(12, 6))
    plt.bar(df_top["primary_diagnosis_code"], df_top["count"])

    ## Improve readability for ICD codes
    plt.xticks(rotation=90)
    plt.title("Top ICD10 Primary Diagnosis Codes")

    ## Adjust layout to prevent label cutoff
    plt.tight_layout()

    path = ensure_parent_dir(output_path)

    plt.savefig(path)
    plt.close()

    logger.info("EDA plot saved to %s", path)

    return path

## ============================================================
## TEXT STATISTICS
## ============================================================
def compute_text_length_stats(
    clinical_csv_dir: str | Path,
) -> Dict[str, float]:
    """
        Compute basic text length statistics

        Steps:
            - Concatenate text_content per admission
            - Measure character length
            - Compute mean, median, max

        Args:
            clinical_csv_dir: Folder with per-admission CSV files

        Returns:
            Dictionary with:
                - mean_length
                - median_length
                - max_length
    """

    logger.info("Computing text length statistics")

    ## Resolve directory path
    csv_dir = Path(clinical_csv_dir)

    ## Collect all admission CSV files
    csv_files = sorted(csv_dir.glob("*.csv"))

    lengths = []

    ## Iterate over each admission
    for file_path in csv_files:
        df = pd.read_csv(file_path)

        if "text_content" in df.columns:

            ## Concatenate all document texts for this admission
            full_text = " ".join(df["text_content"].astype(str).tolist())

            lengths.append(len(full_text))

    ## If no valid text found, return safe defaults
    if not lengths:
        logger.warning("No text content found for EDA statistics")
        return {
            "mean_length": 0.0,
            "median_length": 0.0,
            "max_length": 0.0,
        }

    ## Convert to pandas Series for aggregation
    series = pd.Series(lengths)

    stats = {
        "mean_length": float(series.mean()),
        "median_length": float(series.median()),
        "max_length": float(series.max()),
    }

    logger.info(
        "Text statistics | mean=%.2f | median=%.2f | max=%.2f",
        stats["mean_length"],
        stats["median_length"],
        stats["max_length"],
    )

    return stats