'''
__author__ = "Olivia Tortosa"
__copyright__ = None
__credits__ = ["Olivia Tortosa", "Georges Nassopoulos"]
__version__ = "1.0.0"
__maintainer__ = "Georges Nassopoulos"
__email__ = "olivia.tortosa@gmail.com", "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Preprocessing utilities for clustering datasets: numeric selection, imputation, scaling and optional dimensionality reduction"
'''

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.core.errors import FeatureEngineeringError
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
## INTERNAL HELPERS
## ============================================================
def _select_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Select numeric feature columns for clustering

        High-level workflow:
            1) Drop non-numeric columns
            2) Keep only float/int columns

        Args:
            df: Input DataFrame

        Returns:
            Numeric-only DataFrame
    """

    numeric_df = df.select_dtypes(include=["number"]).copy()

    if numeric_df.empty:
        raise FeatureEngineeringError(
            message="No numeric features found for clustering",
            details={"columns": list(df.columns)},
        )

    return numeric_df

## ============================================================
## PUBLIC API
## ============================================================
def preprocess_dataset(
    df: pd.DataFrame,
    preprocess_params: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
        Preprocess dataset for clustering

        High-level workflow:
            1) Select numeric feature columns
            2) Impute missing values
            3) Scale features
            4) Optional PCA reduction
            5) Return processed DataFrame and metadata

        Args:
            df: Input dataset DataFrame
            preprocess_params: Preprocessing parameters dict

        Returns:
            Tuple:
                - Processed numeric DataFrame
                - Metadata dictionary
    """

    ## Select numeric columns
    numeric_df = _select_numeric_features(df)

    ## Impute missing values
    impute_strategy = preprocess_params.get("impute_strategy", "median")

    imputer = SimpleImputer(strategy=impute_strategy)
    imputed_array = imputer.fit_transform(numeric_df)

    ## Scale features
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(imputed_array)

    processed_df = pd.DataFrame(
        scaled_array,
        columns=numeric_df.columns,
    )

    ## Optional PCA reduction
    apply_pca = bool(preprocess_params.get("apply_pca", False))
    n_components = int(preprocess_params.get("pca_n_components", 2))

    metadata: Dict[str, Any] = {
        "impute_strategy": impute_strategy,
        "scaled": True,
        "apply_pca": apply_pca,
        "pca_n_components": n_components if apply_pca else None,
    }

    if apply_pca:
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(processed_df.values)

        processed_df = pd.DataFrame(
            reduced,
            columns=[f"pc_{i+1}" for i in range(reduced.shape[1])],
        )

        metadata["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()

    return processed_df, metadata