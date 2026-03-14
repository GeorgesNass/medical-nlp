'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized custom exceptions and structured helpers for the ICD10 prediction pipeline."
'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("errors", log_file="errors.log")

## ============================================================
## ERROR CODES
## ============================================================
ERROR_CODE_CONFIGURATION = "configuration_error"
ERROR_CODE_VALIDATION = "validation_error"
ERROR_CODE_DATA = "data_error"
ERROR_CODE_PARSING = "parsing_error"
ERROR_CODE_PIPELINE = "pipeline_error"
ERROR_CODE_MODEL = "model_error"
ERROR_CODE_RESOURCE_NOT_FOUND = "resource_not_found"
ERROR_CODE_EXTERNAL_SERVICE = "external_service_error"
ERROR_CODE_INTERNAL = "internal_error"

## ============================================================
## BASE EXCEPTION
## ============================================================
class ICD10PredictionError(RuntimeError):
    """
        Base exception for the ICD10 prediction pipeline

        High-level workflow:
            1) Normalize project-specific failures
            2) Preserve structured context for debugging
            3) Support wrapping of lower-level exceptions

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
class ConfigurationError(RuntimeError):
    """
        Raised when application configuration is invalid
    """
    pass

class DataError(RuntimeError):
    """
        Raised when required data files or folders are missing or invalid
    """
    pass

class ParsingError(RuntimeError):
    """
        Raised when parsing of RSS or clinical records fails
    """
    pass

class PipelineError(RuntimeError):
    """
        Raised when a pipeline step fails unexpectedly
    """
    pass

class ModelError(RuntimeError):
    """
        Raised when model training, loading, or inference fails
    """
    pass

class ValidationError(ICD10PredictionError):
    """
        Raised when an input payload or parameter is invalid
    """

class ResourceNotFoundError(ICD10PredictionError):
    """
        Raised when a required file, folder or artifact is missing
    """

class ExternalServiceError(ICD10PredictionError):
    """
        Raised when an external provider or remote service fails
    """

class UnknownICD10PredictionError(ICD10PredictionError):
    """
        Raised when an unexpected exception must be normalized
    """

## ============================================================
## GENERIC HELPERS
## ============================================================
def raise_project_error(
    exc_type: Type[ICD10PredictionError],
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
            2) Attach original cause metadata when available
            3) Log the failure in a consistent format
            4) Raise the normalized project exception

        Args:
            exc_type: Exception class to raise
            message: Human-readable error message
            error_code: Normalized application error code
            details: Optional contextual details
            cause: Original exception if available
            is_retryable: Whether retry may succeed

        Raises:
            ICD10PredictionError: Always
    """

    ## Build a normalized payload
    payload = details.copy() if details else {}

    ## Attach original cause metadata when available
    if cause is not None:
        payload["cause_message"] = str(cause)
        payload["cause_type"] = cause.__class__.__name__

    ## Emit a structured error log
    logger.error(
        "ICD10 prediction error | type=%s | code=%s | message=%s | "
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
    exc_type: Type[ICD10PredictionError],
    message: str,
    error_code: str,
    details: Optional[Dict[str, Any]] = None,
    is_retryable: bool = False,
) -> ICD10PredictionError:
    """
        Wrap a raw exception into a structured project exception

        High-level workflow:
            1) Preserve the original exception
            2) Merge it into the structured payload
            3) Return a normalized project error instance

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

    ## Start from existing details when provided
    payload = details.copy() if details else {}

    ## Attach original cause metadata
    payload["cause_message"] = str(exc)
    payload["cause_type"] = exc.__class__.__name__

    ## Return a normalized wrapped exception
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
) -> UnknownICD10PredictionError:
    """
        Normalize an unexpected exception into a project-specific error

        High-level workflow:
            1) Build a safe execution context
            2) Preserve original exception metadata
            3) Log the unexpected failure
            4) Return a normalized unknown error

        Args:
            exc: Original unexpected exception
            context: Optional execution context

        Returns:
            A normalized unknown project exception
    """

    ## Build a safe payload from optional context
    payload = context.copy() if context else {}

    ## Attach original cause metadata
    payload["cause_message"] = str(exc)
    payload["cause_type"] = exc.__class__.__name__

    ## Log the unexpected failure
    logger.error(
        "Unhandled icd10-prediction exception | type=%s | details=%s",
        exc.__class__.__name__,
        payload,
    )
    logger.debug("Unhandled traceback", exc_info=True)

    ## Return a normalized unknown project error
    return UnknownICD10PredictionError(
        message="An unexpected icd10-prediction error occurred",
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
            vars_missing: List of missing env variable names

        Raises:
            ConfigurationError: Always raised after logging
    """

    ## Keep the original explicit message style
    message = (
        "Missing environment variables (placeholders detected): "
        + ", ".join(vars_missing)
    )
    logger.error(message)

    raise ConfigurationError(message)

def log_and_raise_missing_folder(
    folder: Path,
    reason: Optional[str] = None,
) -> None:
    """
        Log and raise a data error when a required folder is missing

        Args:
            folder: Expected folder path
            reason: Optional human-readable explanation

        Raises:
            DataError: Always raised after logging
    """

    ## Build the original explicit message
    message = f"Required folder not found: {folder}"
    if reason:
        message = f"{message} | {reason}"

    logger.error(message)

    raise DataError(message)

def log_and_raise_missing_file(
    file_path: Path,
    reason: Optional[str] = None,
) -> None:
    """
        Log and raise a data error when a required file is missing

        Args:
            file_path: Expected file path
            reason: Optional human-readable explanation

        Raises:
            DataError: Always raised after logging
    """

    ## Build the original explicit message
    message = f"Required file not found: {file_path}"
    if reason:
        message = f"{message} | {reason}"

    logger.error(message)

    raise DataError(message)

def log_and_raise_parsing_error(source: Path, reason: str) -> None:
    """
        Log and raise a parsing error

        Args:
            source: Source file that failed to parse
            reason: Human-readable failure reason

        Raises:
            ParsingError: Always raised after logging
    """

    ## Build the parsing failure message
    message = f"Parsing failed: {source} | {reason}"
    logger.error(message)

    raise ParsingError(message)

def log_and_raise_pipeline_step(step_name: str, reason: str) -> None:
    """
        Log and raise a pipeline error for a failing step

        Args:
            step_name: Name of the pipeline step
            reason: Human-readable failure reason

        Raises:
            PipelineError: Always raised after logging
    """

    ## Build the pipeline failure message
    message = f"Pipeline step failed [{step_name}]: {reason}"
    logger.error(message)

    raise PipelineError(message)

def log_and_raise_data_error(reason: str) -> None:
    """
        Log and raise a generic data error

        Args:
            reason: Human-readable failure reason

        Raises:
            DataError: Always raised after logging
    """

    ## Build the generic data error message
    message = f"Data error: {reason}"
    logger.error(message)

    raise DataError(message)

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

        Raises:
            ValidationError: Always
    """

    ## Raise a structured validation error
    raise_project_error(
        exc_type=ValidationError,
        message=message,
        error_code=ERROR_CODE_VALIDATION,
        details=details,
        is_retryable=False,
    )

def log_and_raise_model_error(
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise a structured model error

        Args:
            message: Human-readable model error message
            details: Optional model context
            cause: Original exception if available

        Raises:
            ModelError: Always
    """

    ## Keep compatibility with the original ModelError class
    payload = details.copy() if details else {}
    if cause is not None:
        payload["cause_message"] = str(cause)
        payload["cause_type"] = cause.__class__.__name__

    logger.error(
        "ICD10 model error | message=%s | details=%s",
        message,
        payload,
    )

    raise ModelError(message)

def log_and_raise_external_service_error(
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise an external service error

        Args:
            message: Human-readable external service error message
            details: Optional service context
            cause: Original exception if available

        Raises:
            ExternalServiceError: Always
    """

    ## Raise a structured external service error
    raise_project_error(
        exc_type=ExternalServiceError,
        message=message,
        error_code=ERROR_CODE_EXTERNAL_SERVICE,
        details=details,
        cause=cause,
        is_retryable=True,
    )

def log_and_raise_missing_path(
    path: str | Path,
    *,
    resource_name: str = "Required resource",
) -> None:
    """
        Log and raise a missing resource error

        Args:
            path: Missing filesystem path
            resource_name: Human-readable resource label

        Raises:
            ResourceNotFoundError: Always
    """

    ## Normalize the path for logs and payloads
    normalized_path = str(Path(path))

    ## Raise a structured missing resource error
    raise_project_error(
        exc_type=ResourceNotFoundError,
        message=f"{resource_name} not found",
        error_code=ERROR_CODE_RESOURCE_NOT_FOUND,
        details={"path": normalized_path},
        is_retryable=False,
    )