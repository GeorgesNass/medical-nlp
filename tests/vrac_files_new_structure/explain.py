'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Model explanation utilities: feature importance extraction for linear and tree-based models."
'''

from __future__ import annotations

## Standard library
from typing import Any, List, Tuple

## Third-party
import numpy as np
import pandas as pd

## Internal
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("explain", log_file="explain.log")

## ============================================================
## LINEAR MODEL EXPLANATION
## ============================================================
def get_top_features_linear(
    model: Any,
    feature_names: List[str],
    class_index: int,
    top_k: int = 20,
) -> List[Tuple[str, float]]:
    """
        Extract top positive features for a given class
        (e.g., Logistic Regression)

        Args:
            model: Trained linear model with coef_ attribute
            feature_names: Feature names aligned with vectorizer
            class_index: Target class index
            top_k: Number of features to return

        Returns:
            List of (feature, weight)
    """

    if not hasattr(model, "coef_"):
        raise ValueError("Model does not expose coef_ attribute")

    logger.info("Extracting top linear features | class_index=%d", class_index)

    coef = model.coef_[class_index]

    ## Get indices sorted by descending weight
    top_indices = np.argsort(coef)[::-1][:top_k]

    return [(feature_names[i], float(coef[i])) for i in top_indices]

def get_bottom_features_linear(
    model: Any,
    feature_names: List[str],
    class_index: int,
    top_k: int = 20,
) -> List[Tuple[str, float]]:
    """
        Extract most negative features for a given class

        Args:
            model: Trained linear model
            feature_names: Feature names
            class_index: Target class index
            top_k: Number of features

        Returns:
            List of (feature, weight)
    """

    if not hasattr(model, "coef_"):
        raise ValueError("Model does not expose coef_ attribute")

    coef = model.coef_[class_index]

    bottom_indices = np.argsort(coef)[:top_k]

    return [(feature_names[i], float(coef[i])) for i in bottom_indices]

## ============================================================
## TREE-BASED MODEL EXPLANATION
## ============================================================
def get_feature_importance_tree(
    model: Any,
    feature_names: List[str],
    top_k: int = 20,
) -> List[Tuple[str, float]]:
    """
        Extract global feature importance from tree-based model

        Args:
            model: Trained tree-based model with feature_importances_
            feature_names: Feature names
            top_k: Number of features

        Returns:
            List of (feature, importance)
    """

    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not expose feature_importances_")

    logger.info("Extracting tree-based feature importances")

    importances = model.feature_importances_

    indices = np.argsort(importances)[::-1][:top_k]

    return [(feature_names[i], float(importances[i])) for i in indices]

## ============================================================
## EXPORT UTILITY
## ============================================================
def feature_importance_dataframe(
    feature_importances: List[Tuple[str, float]],
) -> pd.DataFrame:
    """
        Convert feature importance list to DataFrame

        Args:
            feature_importances: List of (feature, score)

        Returns:
            DataFrame with columns [feature, score]
    """

    df = pd.DataFrame(feature_importances, columns=["feature", "score"])
    
    return df.sort_values(by="score", ascending=False).reset_index(drop=True)