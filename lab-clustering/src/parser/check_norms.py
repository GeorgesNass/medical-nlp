'''
__author__ = "Olivia Tortosa"
__copyright__ = None
__credits__ = ["Olivia Tortosa", "Georges Nassopoulos"]
__version__ = "1.0.0"
__maintainer__ = "Georges Nassopoulos"
__email__ = "olivia.tortosa@gmail.com", "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Reference interval evaluation logic: compute low/normal/high status from numeric value and norms"
'''

from __future__ import annotations

from typing import Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
## INTERNAL HELPERS
## ============================================================
def _safe_float(value: Optional[float]) -> Optional[float]:
    """
        Safely coerce value to float

        High-level workflow:
            1) Return None if value is None
            2) Attempt float conversion
            3) Return None if conversion fails

        Args:
            value: Any numeric-like value

        Returns:
            Float or None
    """

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None

## ============================================================
## PUBLIC API
## ============================================================
def compute_status_from_norms(
    value: Optional[float],
    norms_min: Optional[float],
    norms_max: Optional[float],
) -> str:
    """
        Compute laboratory status from reference interval

        High-level workflow:
            1) Coerce inputs to float safely
            2) Handle missing value
            3) Handle missing norms
            4) Compare against min and max
            5) Return categorical status

        Args:
            value: Patient numeric value
            norms_min: Lower reference bound
            norms_max: Upper reference bound

        Returns:
            Status string among:
                - "low"
                - "normal"
                - "high"
                - "unknown"
    """

    ## Coerce values safely
    value_f = _safe_float(value)
    min_f = _safe_float(norms_min)
    max_f = _safe_float(norms_max)

    ## Missing patient value
    if value_f is None:
        return "unknown"

    ## Missing reference interval
    if min_f is None and max_f is None:
        return "unknown"

    ## Compare against lower bound
    if min_f is not None and value_f < min_f:
        return "low"

    ## Compare against upper bound
    if max_f is not None and value_f > max_f:
        return "high"

    ## Value within interval
    return "normal"