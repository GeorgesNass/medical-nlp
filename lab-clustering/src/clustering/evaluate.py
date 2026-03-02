'''
__author__ = "Olivia Tortosa"
__copyright__ = None
__credits__ = ["Olivia Tortosa", "Georges Nassopoulos"]
__version__ = "1.0.0"
__maintainer__ = "Georges Nassopoulos"
__email__ = "olivia.tortosa@gmail.com", "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Post-clustering evaluation utilities: cluster size analysis and cluster profiling statistics"
'''

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from src.core.errors import ClusteringError
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
## CLUSTER SIZE ANALYSIS
## ============================================================
def compute_cluster_sizes(labels: np.ndarray) -> pd.DataFrame:
    """
        Compute cluster size distribution

        High-level workflow:
            1) Count occurrences per cluster label
            2) Sort by cluster id
            3) Return DataFrame

        Args:
            labels: Cluster label array

        Returns:
            DataFrame with columns:
                - cluster
                - size
    """

    ## Count cluster labels
    unique, counts = np.unique(labels, return_counts=True)

    cluster_sizes = pd.DataFrame(
        {
            "cluster": unique,
            "size": counts,
        }
    )

    cluster_sizes = cluster_sizes.sort_values("cluster").reset_index(drop=True)

    return cluster_sizes

## ============================================================
## CLUSTER PROFILING
## ============================================================
def compute_cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
        Compute mean feature profile per cluster

        High-level workflow:
            1) Attach labels to DataFrame
            2) Group by cluster
            3) Compute mean per numeric feature
            4) Return profiling DataFrame

        Args:
            df: Numeric feature DataFrame used for clustering
            labels: Cluster label array

        Returns:
            DataFrame with mean feature values per cluster
    """

    if df.shape[0] != len(labels):
        raise ClusteringError(
            message="Mismatch between dataset rows and labels",
            details={
                "n_rows": df.shape[0],
                "n_labels": len(labels),
            },
        )

    ## Attach cluster labels
    df_with_labels = df.copy()
    df_with_labels["cluster"] = labels

    ## Compute mean profile per cluster
    profiles = (
        df_with_labels
        .groupby("cluster")
        .mean(numeric_only=True)
        .reset_index()
    )

    logger.info(
        "Cluster profiles computed | clusters=%s | features=%s",
        profiles.shape[0],
        profiles.shape[1] - 1,
    )

    return profiles

## ============================================================
## FULL EVALUATION WRAPPER
## ============================================================
def evaluate_clustering(
    df: pd.DataFrame,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """
        Perform full clustering evaluation

        High-level workflow:
            1) Compute cluster sizes
            2) Compute cluster profiles
            3) Return evaluation dictionary

        Args:
            df: Numeric feature DataFrame
            labels: Cluster label array

        Returns:
            Dictionary containing:
                - cluster_sizes (DataFrame)
                - cluster_profiles (DataFrame)
    """

    cluster_sizes = compute_cluster_sizes(labels)
    cluster_profiles = compute_cluster_profiles(df, labels)

    return {
        "cluster_sizes": cluster_sizes,
        "cluster_profiles": cluster_profiles,
    }