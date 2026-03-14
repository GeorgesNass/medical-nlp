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
## ERROR CODES
## ============================================================
ERROR_CODE_PARSING = "parsing_error"
ERROR_CODE_REGEX_LOADING = "regex_loading_error"
ERROR_CODE_NORMS_LOADING = "norms_loading_error"
ERROR_CODE_UNIT_CONVERSION = "unit_conversion_error"
ERROR_CODE_VALUE_INTERPRETATION = "value_interpretation_error"
ERROR_CODE_DATASET_BUILD = "dataset_build_error"
ERROR_CODE_FEATURE_ENGINEERING = "feature_engineering_error"
ERROR_CODE_DATA_VALIDATION = "data_validation_error"
ERROR_CODE_CLUSTERING = "clustering_error"
ERROR_CODE_MODEL_PERSISTENCE = "model_persistence_error"
ERROR_CODE_MLFLOW_TRACKING = "mlflow_tracking_error"
ERROR_CODE_CONFIGURATION = "configuration_error"
ERROR_CODE_RESOURCE_NOT_FOUND = "resource_not_found"
ERROR_CODE_INTERNAL = "internal_error"

## ============================================================
## BASE EXCEPTION
## ============================================================
class LabClusteringError(Exception):
    """
        Base exception for lab_clustering application

        High-level workflow:
            1) Normalize domain-specific failures
            2) Preserve structured context for debugging
            3) Support standardized API responses

        Args:
            message: Human-readable error message
            details: Optional additional contextual information
            error_code: Normalized application error code
            cause: Original exception if available
            is_retryable: Whether retry may succeed
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: str = ERROR_CODE_INTERNAL,
        cause: Optional[Exception] = None,
        is_retryable: bool = False,
    ) -> None:
        ## Store normalized error metadata
        self.message = message
        self.details = details or {}
        self.error_code = error_code
        self.cause = cause
        self.is_retryable = is_retryable

        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """
            Convert the exception into a structured dictionary

            Returns:
                A normalized error payload
        """

        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "cause_type": self.cause.__class__.__name__
            if self.cause
            else None,
            "is_retryable": self.is_retryable,
        }

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


class UnknownLabClusteringError(LabClusteringError):
    """Raised when an unexpected exception must be normalized"""

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

    ## Log the structured application error
    logger.error(
        "Application error | path=%s | type=%s | code=%s | message=%s | "
        "details=%s",
        request.url.path,
        exc.__class__.__name__,
        exc.error_code,
        exc.message,
        exc.details,
    )

    ## Return standardized API response
    return JSONResponse(
        status_code=400,
        content=exc.to_dict(),
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

    ## Log the unexpected exception in a structured way
    logger.error(
        "Unhandled exception | path=%s | error=%s",
        request.url.path,
        str(exc),
    )

    ## Emit full traceback only in debug logs
    logger.debug("Traceback:", exc_info=True)

    ## Return generic internal error response
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "error_code": ERROR_CODE_INTERNAL,
        },
    )

## ============================================================
## GENERIC HELPERS
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

    ## Resolve error code from exception type
    error_code = get_error_code_for_exception(exc_type)

    ## Log structured error before raising
    logger.error(
        "Raising error | type=%s | code=%s | message=%s | details=%s",
        exc_type.__name__,
        error_code,
        message,
        payload,
    )

    ## Raise typed domain exception
    raise exc_type(
        message=message,
        details=payload,
        error_code=error_code,
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

    ## Attach original exception metadata
    payload["original_error"] = str(exc)
    payload["original_error_type"] = exc.__class__.__name__

    ## Return wrapped domain exception without raising it
    return exc_type(
        message=message,
        details=payload,
        error_code=get_error_code_for_exception(exc_type),
        cause=exc,
    )

def log_unhandled_exception(
    exc: Exception,
    context: Optional[Dict[str, Any]] = None,
) -> UnknownLabClusteringError:
    """
        Normalize an unexpected exception into a domain-specific error

        Args:
            exc: Original unexpected exception
            context: Optional execution context

        Returns:
            Instantiated UnknownLabClusteringError
    """

    ## Initialize payload from optional context
    payload = context.copy() if context else {}

    ## Attach original exception metadata
    payload["original_error"] = str(exc)
    payload["original_error_type"] = exc.__class__.__name__

    ## Log structured unexpected error
    logger.error(
        "Unhandled domain exception | type=%s | details=%s",
        exc.__class__.__name__,
        payload,
    )
    logger.debug("Unhandled traceback", exc_info=True)

    ## Return normalized unknown domain exception
    return UnknownLabClusteringError(
        message="An unexpected lab-clustering error occurred",
        details=payload,
        error_code=ERROR_CODE_INTERNAL,
        cause=exc,
        is_retryable=False,
    )

## ============================================================
## SPECIALIZED HELPERS
## ============================================================
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

## ============================================================
## ERROR CODE MAPPING
## ============================================================
def get_error_code_for_exception(
    exc_type: Type[LabClusteringError],
) -> str:
    """
        Map a domain exception type to a normalized error code

        Args:
            exc_type: Domain exception class

        Returns:
            Normalized application error code
    """

    ## Build static exception-to-code mapping
    mapping: Dict[Type[LabClusteringError], str] = {
        ParsingError: ERROR_CODE_PARSING,
        RegexLoadingError: ERROR_CODE_REGEX_LOADING,
        NormsLoadingError: ERROR_CODE_NORMS_LOADING,
        UnitConversionError: ERROR_CODE_UNIT_CONVERSION,
        ValueInterpretationError: ERROR_CODE_VALUE_INTERPRETATION,
        DatasetBuildError: ERROR_CODE_DATASET_BUILD,
        FeatureEngineeringError: ERROR_CODE_FEATURE_ENGINEERING,
        DataValidationError: ERROR_CODE_DATA_VALIDATION,
        ClusteringError: ERROR_CODE_CLUSTERING,
        ModelPersistenceError: ERROR_CODE_MODEL_PERSISTENCE,
        MlflowTrackingError: ERROR_CODE_MLFLOW_TRACKING,
        ConfigurationError: ERROR_CODE_CONFIGURATION,
        ResourceNotFoundError: ERROR_CODE_RESOURCE_NOT_FOUND,
        UnknownLabClusteringError: ERROR_CODE_INTERNAL,
    }

    ## Return mapped code or fallback internal code
    return mapping.get(exc_type, ERROR_CODE_INTERNAL)