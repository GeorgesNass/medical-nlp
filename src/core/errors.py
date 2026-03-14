'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized custom exceptions and structured helpers for the clinical NER pipeline."
'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("errors")

## ============================================================
## ERROR CODES
## ============================================================
ERROR_CODE_CONFIGURATION = "configuration_error"
ERROR_CODE_VALIDATION = "validation_error"
ERROR_CODE_MODEL = "model_error"
ERROR_CODE_INFERENCE = "inference_error"
ERROR_CODE_DATA = "data_error"
ERROR_CODE_RESOURCE_NOT_FOUND = "resource_not_found"
ERROR_CODE_EXTERNAL_SERVICE = "external_service_error"
ERROR_CODE_PIPELINE = "pipeline_error"
ERROR_CODE_INTERNAL = "internal_error"

## ============================================================
## BASE EXCEPTION
## ============================================================
class ClinicalNERError(RuntimeError):
    """
        Base exception for the clinical NER pipeline

        High-level workflow:
            1) Normalize all project-specific failures
            2) Preserve structured context for debugging
            3) Allow wrapping of lower-level library exceptions

        Args:
            message: Human-readable error message
            error_code: Normalized application error code
            details: Optional structured context payload
            cause: Original exception if available
            is_retryable: Whether retry may succeed
    """

    def __init__(
        self,
        message: str,
        error_code: str = ERROR_CODE_INTERNAL,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        is_retryable: bool = False,
    ) -> None:
        ## Store normalized error metadata
        self.message = message
        self.error_code = error_code
        self.details = details or {}
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
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "cause_type": self.cause.__class__.__name__
            if self.cause
            else None,
            "is_retryable": self.is_retryable,
        }

## ============================================================
## CUSTOM EXCEPTIONS
## ============================================================
class ConfigurationError(ClinicalNERError):
    """
        Raised when application configuration is invalid
    """

class ModelLoadError(ClinicalNERError):
    """
        Raised when the NER model cannot be loaded
    """

class InferenceError(ClinicalNERError):
    """
        Raised when model inference fails
    """

class DataError(ClinicalNERError):
    """
        Raised when input data is missing or invalid
    """

class ValidationError(ClinicalNERError):
    """
        Raised when validation checks fail
    """

class ResourceNotFoundError(ClinicalNERError):
    """
        Raised when a required file or artifact is missing
    """

class ExternalServiceError(ClinicalNERError):
    """
        Raised when an external provider fails
    """

class PipelineError(ClinicalNERError):
    """
        Raised when pipeline orchestration fails
    """

class UnknownClinicalNERError(ClinicalNERError):
    """
        Raised when an unexpected exception must be normalized
    """

## ============================================================
## GENERIC HELPERS
## ============================================================
def raise_project_error(
    exc_type: Type[ClinicalNERError],
    message: str,
    *,
    error_code: str,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
    is_retryable: bool = False,
) -> None:
    """
        Log and raise a structured project exception

        High-level workflow:
            1) Build a normalized payload
            2) Attach original cause metadata
            3) Log the failure
            4) Raise the normalized project exception

        Args:
            exc_type: Exception class to raise
            message: Human-readable error message
            error_code: Normalized application error code
            details: Optional contextual details
            cause: Original exception if available
            is_retryable: Whether retry may succeed

        Raises:
            ClinicalNERError: Always
    """

    ## Build a normalized payload
    payload = details.copy() if details else {}

    ## Attach original cause metadata when available
    if cause is not None:
        payload["cause_message"] = str(cause)
        payload["cause_type"] = cause.__class__.__name__

    ## Emit structured error log
    logger.error(
        "Clinical NER error | type=%s | code=%s | message=%s | "
        "retryable=%s | details=%s",
        exc_type.__name__,
        error_code,
        message,
        is_retryable,
        payload,
    )

    ## Raise the normalized project exception
    raise exc_type(
        message=message,
        error_code=error_code,
        details=payload,
        cause=cause,
        is_retryable=is_retryable,
    )

def wrap_exception(
    exc: Exception,
    *,
    exc_type: Type[ClinicalNERError],
    message: str,
    error_code: str,
    details: Optional[Dict[str, Any]] = None,
    is_retryable: bool = False,
) -> ClinicalNERError:
    """
        Wrap a raw exception into a structured project exception

        Args:
            exc: Original exception
            exc_type: Target structured exception type
            message: Human-readable error message
            error_code: Normalized application error code
            details: Optional contextual details
            is_retryable: Whether retry may succeed

        Returns:
            A structured project exception instance
    """

    ## Merge existing context with original exception
    payload = details.copy() if details else {}
    payload["cause_message"] = str(exc)
    payload["cause_type"] = exc.__class__.__name__

    return exc_type(
        message=message,
        error_code=error_code,
        details=payload,
        cause=exc,
        is_retryable=is_retryable,
    )

def log_unhandled_exception(
    exc: Exception,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> UnknownClinicalNERError:
    """
        Normalize an unexpected exception into a project-specific error

        Args:
            exc: Original unexpected exception
            context: Optional execution context

        Returns:
            A normalized unknown project exception
    """

    ## Build safe payload from optional context
    payload = context.copy() if context else {}

    ## Attach original cause metadata
    payload["cause_message"] = str(exc)
    payload["cause_type"] = exc.__class__.__name__

    ## Log the unexpected failure
    logger.error(
        "Unhandled clinical-ner exception | type=%s | details=%s",
        exc.__class__.__name__,
        payload,
    )
    logger.debug("Unhandled traceback", exc_info=True)

    ## Return a normalized unknown error
    return UnknownClinicalNERError(
        message="An unexpected clinical-ner error occurred",
        error_code=ERROR_CODE_INTERNAL,
        details=payload,
        cause=exc,
        is_retryable=False,
    )

## ============================================================
## SPECIALIZED HELPERS
## ============================================================
def log_and_raise_missing_env(vars_missing: List[str]) -> None:
    """
        Log and raise a configuration error for missing environment variables

        Args:
            vars_missing: List of missing environment variable names

        Raises:
            ConfigurationError: Always
    """

    ## Keep original message style
    message = "Missing environment variables: " + ", ".join(vars_missing)

    logger.error(message)

    raise ConfigurationError(
        message=message,
        error_code=ERROR_CODE_CONFIGURATION,
        details={"missing_variables": vars_missing},
        is_retryable=False,
    )

def log_and_raise_missing_model(model_path: Path) -> None:
    """
        Log and raise a model loading error

        Args:
            model_path: Expected path to the NER model

        Raises:
            ModelLoadError: Always
    """

    ## Build explicit message
    message = f"NER model not found at path: {model_path}"

    logger.error(message)

    raise ModelLoadError(
        message=message,
        error_code=ERROR_CODE_MODEL,
        details={"model_path": str(model_path)},
        is_retryable=False,
    )

def log_and_raise_validation_error(
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
        Log and raise a validation error

        Args:
            message: Human-readable validation error message
            details: Optional validation context
    """

    raise_project_error(
        exc_type=ValidationError,
        message=message,
        error_code=ERROR_CODE_VALIDATION,
        details=details,
        is_retryable=False,
    )

def log_and_raise_inference_error(
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise an inference error

        Args:
            message: Human-readable inference error message
            details: Optional inference context
            cause: Original exception if available
    """

    raise_project_error(
        exc_type=InferenceError,
        message=message,
        error_code=ERROR_CODE_INFERENCE,
        details=details,
        cause=cause,
        is_retryable=True,
    )

def log_and_raise_data_error(
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
        Log and raise a data error

        Args:
            message: Human-readable data error message
            details: Optional data context
    """

    raise_project_error(
        exc_type=DataError,
        message=message,
        error_code=ERROR_CODE_DATA,
        details=details,
        is_retryable=False,
    )

def log_and_raise_pipeline_error(
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise a pipeline error

        Args:
            message: Human-readable pipeline error message
            details: Optional pipeline context
            cause: Original exception if available
    """

    raise_project_error(
        exc_type=PipelineError,
        message=message,
        error_code=ERROR_CODE_PIPELINE,
        details=details,
        cause=cause,
        is_retryable=False,
    )