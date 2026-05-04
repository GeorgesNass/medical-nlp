'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Statistical utilities for clustering data quality: mean/std, IQR, extremes and winsorization."
'''

from __future__ import annotations

from typing import Tuple, Dict, Any

import numpy as np

from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("stats_utils")

def compute_mean_std(arr: np.ndarray) -> Tuple[float, float]:
    """
        Compute mean and standard deviation

        Args:
            arr: Input numeric array

        Returns:
            Tuple (mean, std)
    """

    ## compute mean
    mean = float(np.mean(arr))

    ## compute std
    std = float(np.std(arr))

    return mean, std

def compute_iqr_bounds(arr: np.ndarray, multiplier: float = 1.5) -> Tuple[float, float]:
    """
        Compute IQR bounds

        Args:
            arr: Input numeric array
            multiplier: IQR multiplier

        Returns:
            Lower and upper bounds
    """

    ## compute quartiles
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))

    ## compute IQR
    iqr = q3 - q1

    ## compute bounds
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    return lower, upper

def detect_extremes(arr: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
    """
        Detect extreme values

        Args:
            arr: Input numeric array
            top_k: Number of extreme values

        Returns:
            Dict with smallest and largest values
    """

    ## sort values
    sorted_vals = np.sort(arr)

    ## extract extremes
    smallest = sorted_vals[:top_k].tolist()
    largest = sorted_vals[-top_k:].tolist()

    return {
        "smallest": smallest,
        "largest": largest,
    }

def winsorize(
    arr: np.ndarray,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> np.ndarray:
    """
        Apply winsorization to limit extreme values

        Args:
            arr: Input numeric array
            lower_q: Lower quantile
            upper_q: Upper quantile

        Returns:
            Clipped array
    """

    ## compute bounds
    lower = float(np.quantile(arr, lower_q))
    upper = float(np.quantile(arr, upper_q))

    ## clip values
    return np.clip(arr, lower, upper)
    
## ============================================================
## FEATURE ENGINEERING - TEXT STATS
## ============================================================
def compute_text_stats(text: str) -> Dict[str, Any]:
    """
        Compute text-level statistics for feature engineering

        High-level workflow:
            1) Validate input text
            2) Compute character length
            3) Compute token count
            4) Compute average token length

        Args:
            text: Input string

        Returns:
            Dictionary with text statistics
    """

    if not isinstance(text, str):
        return {
            "char_length": 0,
            "token_count": 0,
            "avg_token_length": 0.0,
        }

    tokens = text.split()

    char_length = len(text)
    token_count = len(tokens)

    avg_token_length = (
        sum(len(t) for t in tokens) / token_count
        if token_count > 0 else 0.0
    )

    return {
        "char_length": char_length,
        "token_count": token_count,
        "avg_token_length": avg_token_length,
    }