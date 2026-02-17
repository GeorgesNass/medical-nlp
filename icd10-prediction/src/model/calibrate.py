'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Probability calibration utilities for ICD10 models using Platt scaling or isotonic regression."
'''

from __future__ import annotations

## Standard library
from typing import Any, Literal

## Third-party
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

## Internal
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("calibrate", log_file="calibrate.log")

## ============================================================
## TYPES
## ============================================================
CalibrationMethod = Literal["sigmoid", "isotonic"]

## ============================================================
## CALIBRATION
## ============================================================
def calibrate_model(
    base_model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: CalibrationMethod = "sigmoid",
    cv: int = 3,
) -> CalibratedClassifierCV:
    """
        Calibrate a trained model using validation data

        Supported methods:
            - sigmoid  -> Platt scaling
            - isotonic -> Isotonic regression

        Args:
            base_model: Trained base classifier
            X_val: Validation feature matrix
            y_val: Validation labels
            method: Calibration method
            cv: Number of folds for internal calibration

        Returns:
            CalibratedClassifierCV instance
    """

    logger.info(
        "Calibrating model | method=%s | samples=%d",
        method,
        X_val.shape[0],
    )

    calibrated = CalibratedClassifierCV(
        base_estimator=base_model,
        method=method,
        cv=cv,
    )

    calibrated.fit(X_val, y_val)

    logger.info("Calibration completed")

    return calibrated

def get_calibrated_probabilities(
    calibrated_model: CalibratedClassifierCV,
    X: np.ndarray,
) -> np.ndarray:
    """
        Extract calibrated probabilities

        Args:
            calibrated_model: Calibrated classifier
            X: Feature matrix

        Returns:
            Probability matrix
    """

    logger.info("Generating calibrated probabilities")

    return calibrated_model.predict_proba(X)