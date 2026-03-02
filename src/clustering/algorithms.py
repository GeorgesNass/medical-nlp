'''
__author__ = "Olivia Tortosa"
__copyright__ = None
__credits__ = ["Olivia Tortosa", "Georges Nassopoulos"]
__version__ = "1.0.0"
__maintainer__ = "Georges Nassopoulos"
__email__ = "olivia.tortosa@gmail.com", "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unsupervised clustering algorithms wrapper: KMeans, Agglomerative, DBSCAN, Birch"
'''

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from src.core.errors import ClusteringError
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
## INTERNAL HELPERS
## ============================================================
def _compute_unsupervised_metrics(
    X: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
        Compute clustering quality metrics

        High-level workflow:
            1) Validate number of clusters
            2) Compute silhouette score if applicable
            3) Return metrics dictionary

        Args:
            X: Feature matrix
            labels: Cluster labels

        Returns:
            Metrics dictionary
    """

    unique_labels = np.unique(labels)

    metrics: Dict[str, float] = {
        "n_clusters": float(len(unique_labels)),
    }

    ## Silhouette requires at least 2 clusters and no single cluster
    if len(unique_labels) > 1 and -1 not in unique_labels:
        try:
            metrics["silhouette_score"] = float(
                silhouette_score(X, labels)
            )
        except Exception:
            metrics["silhouette_score"] = float("nan")
    else:
        metrics["silhouette_score"] = float("nan")

    return metrics

def _build_model(
    algorithm: str,
    params: Dict[str, Any],
):
    """
        Instantiate clustering model from algorithm name

        Args:
            algorithm: Algorithm identifier
            params: Parameter dictionary

        Returns:
            Instantiated clustering model
    """

    if algorithm == "kmeans":
        return KMeans(
            n_clusters=int(params.get("n_clusters", 3)),
            random_state=42,
        )

    if algorithm == "agglomerative":
        return AgglomerativeClustering(
            n_clusters=int(params.get("n_clusters", 3)),
            linkage=params.get("linkage", "ward"),
        )

    if algorithm == "dbscan":
        return DBSCAN(
            eps=float(params.get("eps", 0.5)),
            min_samples=int(params.get("min_samples", 5)),
        )

    if algorithm == "birch":
        return Birch(
            n_clusters=int(params.get("n_clusters", 3)),
        )

    raise ClusteringError(
        message="Unsupported clustering algorithm",
        details={"algorithm": algorithm},
    )

## ============================================================
## PUBLIC API
## ============================================================
def run_clustering_algorithm(
    df: pd.DataFrame,
    clustering_params: Any,
) -> Dict[str, Any]:
    """
        Run selected clustering algorithm on dataset

        High-level workflow:
            1) Instantiate model
            2) Fit model on numeric matrix
            3) Extract labels
            4) Compute metrics
            5) Return result dictionary

        Args:
            df: Preprocessed numeric DataFrame
            clustering_params: Pydantic clustering parameter object

        Returns:
            Dictionary containing:
                - model
                - labels
                - metrics
                - n_clusters
    """

    try:
        ## Extract algorithm and parameters
        algorithm = clustering_params.algorithm
        params = clustering_params.model_dump()

        ## Instantiate model
        model = _build_model(algorithm, params)

        ## Fit model
        X = df.values
        labels = model.fit_predict(X)

        ## Compute metrics
        metrics = _compute_unsupervised_metrics(X, labels)

        result = {
            "model": model,
            "labels": labels,
            "metrics": metrics,
            "n_clusters": int(len(np.unique(labels))),
        }

        logger.info(
            "Clustering completed | algorithm=%s | n_clusters=%s",
            algorithm,
            result["n_clusters"],
        )

        return result

    except Exception as exc:
        logger.error("Clustering failed | error=%s", str(exc))
        logger.debug("Traceback:", exc_info=True)

        raise ClusteringError(
            message="Clustering execution failed",
            details={"error": str(exc)},
        )