'''
__author__ = "Olivia Tortosa"
__copyright__ = None
__credits__ = ["Olivia Tortosa", "Georges Nassopoulos"]
__version__ = "1.0.0"
__maintainer__ = "Georges Nassopoulos"
__email__ = "olivia.tortosa@gmail.com", "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Export clustering artifacts: cluster assignments, cluster profiles and summary CSV outputs"
'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.clustering.evaluate import evaluate_clustering
from src.core.config import AppConfig
from src.core.errors import ModelPersistenceError
from src.utils.io_utils import ensure_dir, write_csv
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
## INTERNAL HELPERS
## ============================================================
def _export_cluster_assignments(
    dataset: pd.DataFrame,
    labels,
    export_dir: Path,
) -> Path:
    """
        Export cluster assignment CSV

        High-level workflow:
            1) Attach labels to original dataset
            2) Write CSV to disk
            3) Return file path

        Args:
            dataset: Original dataset before preprocessing
            labels: Cluster labels
            export_dir: Target directory

        Returns:
            Path to exported CSV
    """

    ## Attach cluster labels
    df_export = dataset.copy()
    df_export["cluster"] = labels

    ## Write CSV
    output_path = export_dir / "cluster_assignments.csv"
    write_csv(df_export, output_path)

    return output_path

def _export_cluster_profiles(
    processed_df: pd.DataFrame,
    labels,
    export_dir: Path,
) -> Path:
    """
        Export cluster profile statistics

        High-level workflow:
            1) Compute evaluation artifacts
            2) Save cluster profiles
            3) Save cluster sizes
            4) Return profile path

        Args:
            processed_df: Numeric dataset used for clustering
            labels: Cluster labels
            export_dir: Target directory

        Returns:
            Path to cluster_profiles.csv
    """

    ## Evaluate clustering
    evaluation = evaluate_clustering(processed_df, labels)

    cluster_profiles = evaluation["cluster_profiles"]
    cluster_sizes = evaluation["cluster_sizes"]

    ## Export profiles
    profiles_path = export_dir / "cluster_profiles.csv"
    sizes_path = export_dir / "cluster_sizes.csv"

    write_csv(cluster_profiles, profiles_path)
    write_csv(cluster_sizes, sizes_path)

    return profiles_path

## ============================================================
## PUBLIC API
## ============================================================
def export_clustering_artifacts(
    clustering_result: Dict[str, Any],
    dataset: pd.DataFrame,
    config: AppConfig,
    overwrite: bool,
) -> Dict[str, str]:
    """
        Export clustering artifacts to artifacts/exports/clustering

        High-level workflow:
            1) Resolve export directory
            2) Ensure directory exists
            3) Export assignments
            4) Export cluster profiles and sizes
            5) Return exported file paths

        Args:
            clustering_result: Output dictionary from run_clustering_algorithm
            dataset: Original dataset before preprocessing
            config: AppConfig instance
            overwrite: Overwrite existing files flag

        Returns:
            Dictionary of exported artifact paths
    """

    try:
        ## Resolve export directory
        export_dir = config.paths.artifacts_exports_clustering_dir
        ensure_dir(export_dir)

        labels = clustering_result["labels"]
        processed_df = clustering_result.get("processed_df")

        ## Export assignments
        assignments_path = _export_cluster_assignments(
            dataset=dataset,
            labels=labels,
            export_dir=export_dir,
        )

        ## Export cluster profiles
        profiles_path = _export_cluster_profiles(
            processed_df=processed_df if processed_df is not None else dataset,
            labels=labels,
            export_dir=export_dir,
        )

        logger.info(
            "Clustering artifacts exported | directory=%s",
            export_dir,
        )

        return {
            "cluster_assignments": str(assignments_path),
            "cluster_profiles": str(profiles_path),
        }

    except Exception as exc:
        logger.error("Export failed | error=%s", str(exc))
        logger.debug("Traceback:", exc_info=True)

        raise ModelPersistenceError(
            message="Failed to export clustering artifacts",
            details={"error": str(exc)},
        )