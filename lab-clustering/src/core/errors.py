'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized custom exceptions and structured error handling for lab_clustering API."
'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Type

from fastapi import Request
from fastapi.responses import JSONResponse

from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger(__name__)

## ============================================================
## BASE EXCEPTION
## ============================================================
class LabClusteringError(Exception):
    """
        Base exception for lab_clustering application

        Args:
            message: Human-readable error message
            details: Optional additional contextual information
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)

## ============================================================
## PARSER ERRORS
## ============================================================
class ParsingError(LabClusteringError):
    """Raised when TXT parsing fails"""

class RegexLoadingError(LabClusteringError):
    """Raised when regex configuration cannot be loaded"""

class NormsLoadingError(LabClusteringError):
    """Raised when norms CSV files cannot be loaded"""

class UnitConversionError(LabClusteringError):
    """Raised when unit conversion fails"""

class ValueInterpretationError(LabClusteringError):
    """Raised when structured value extraction fails"""

## ============================================================
## DATASET ERRORS
## ============================================================
class DatasetBuildError(LabClusteringError):
    """Raised when dataset construction fails"""

class FeatureEngineeringError(LabClusteringError):
    """Raised when feature engineering step fails"""

class DataValidationError(LabClusteringError):
    """Raised when structured dataset validation fails"""

## ============================================================
## CLUSTERING ERRORS
## ============================================================
class ClusteringError(LabClusteringError):
    """Raised when clustering pipeline fails"""

class ModelPersistenceError(LabClusteringError):
    """Raised when model saving/loading fails"""

class MlflowTrackingError(LabClusteringError):
    """Raised when MLflow tracking fails"""

## ============================================================
## CONFIGURATION ERRORS
## ============================================================
class ConfigurationError(LabClusteringError):
    """Raised when configuration or environment setup fails"""

class ResourceNotFoundError(LabClusteringError):
    """Raised when required resource file is missing"""

## ============================================================
## EXCEPTION HANDLERS
## ============================================================
async def lab_clustering_exception_handler(
    request: Request,
    exc: LabClusteringError,
) -> JSONResponse:
    """
        Handle domain-specific LabClusteringError exceptions

        High-level workflow:
            1) Log structured error
            2) Return standardized JSON response

        Args:
            request: FastAPI request object
            exc: Raised LabClusteringError

        Returns:
            JSONResponse with structured error payload
    """

    ## Structured error log
    logger.error(
        "Application error | path=%s | type=%s | message=%s | details=%s",
        request.url.path,
        exc.__class__.__name__,
        exc.message,
        exc.details,
    )

    return JSONResponse(
        status_code=400,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details,
        },
    )

async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
        Handle unexpected exceptions

        High-level workflow:
            1) Log error with traceback
            2) Return generic 500 response

        Args:
            request: FastAPI request object
            exc: Unexpected exception

        Returns:
            JSONResponse with generic error payload
    """

    ## Log error in a structured way
    logger.error(
        "Unhandled exception | path=%s | error=%s",
        request.url.path,
        str(exc),
    )

    ## Full traceback only in DEBUG file logs
    logger.debug("Traceback:", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
        },
    )

## ============================================================
## ERROR HELPERS (LOG + RAISE)
## ============================================================
def log_and_raise(
    exc_type: Type[LabClusteringError],
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
        Log and raise a typed LabClusteringError

        Args:
            exc_type: Exception class to raise
            message: Human-readable message
            details: Optional details payload

        Raises:
            LabClusteringError: Always raised
    """

    ## Normalize details payload
    payload = details or {}

    ## Log structured error before raising
    logger.error(
        "Raising error | type=%s | message=%s | details=%s",
        exc_type.__name__,
        message,
        payload,
    )

    ## Raise typed domain exception
    raise exc_type(
        message=message,
        details=payload,
    )

def log_and_raise_missing_file(
    path: str | Path,
    reason: str,
) -> None:
    """
        Raise ResourceNotFoundError for a missing file

        Args:
            path: Missing path
            reason: Explanation / remediation hint

        Raises:
            ResourceNotFoundError: Always raised
    """

    ## Normalize path object
    p = Path(path)

    ## Build structured error payload
    payload = {
        "path": str(p),
        "reason": reason,
    }

    ## Delegate to generic log_and_raise helper
    log_and_raise(
        exc_type=ResourceNotFoundError,
        message="Required file not found",
        details=payload,
    )

def log_and_raise_data_error(
    reason: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
        Raise DataValidationError with structured details

        Args:
            reason: Human-readable explanation
            details: Optional additional context

        Raises:
            DataValidationError: Always raised
    """

    ## Initialize base payload
    payload: Dict[str, Any] = {
        "reason": reason,
    }

    ## Merge additional details if provided
    if details:
        payload.update(details)

    ## Raise structured data validation error
    log_and_raise(
        exc_type=DataValidationError,
        message="Data validation error",
        details=payload,
    )

def wrap_exception_as(
    exc: Exception,
    exc_type: Type[LabClusteringError],
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> LabClusteringError:
    """
        Wrap any exception into a domain-specific error

        Args:
            exc: Original exception
            exc_type: Target LabClusteringError type
            message: Domain message
            details: Additional context

        Returns:
            Instantiated LabClusteringError
    """

    ## Initialize payload from optional details
    payload = details.copy() if details else {}

    ## Attach original exception message
    payload["original_error"] = str(exc)

    ## Return wrapped domain exception (not raised here)
    return exc_type(
        message=message,
        details=payload,
    )