'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized custom exceptions and structured helpers for the document classification pipeline."
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
ERROR_CODE_DATA = "data_error"
ERROR_CODE_PIPELINE = "pipeline_error"
ERROR_CODE_LABELING = "labeling_error"
ERROR_CODE_RESOURCE_NOT_FOUND = "resource_not_found"
ERROR_CODE_MODEL = "model_error"
ERROR_CODE_INFERENCE = "inference_error"
ERROR_CODE_EXTERNAL_SERVICE = "external_service_error"
ERROR_CODE_INTERNAL = "internal_error"

## ============================================================
## BASE EXCEPTION
## ============================================================
class DocClassificationError(RuntimeError):
    """
        Base exception for the document classification pipeline

        High-level workflow:
            1) Normalize project-specific failures
            2) Preserve structured context for debugging
            3) Support clean wrapping of lower-level exceptions

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
class ConfigurationError(DocClassificationError):
    """
        Raised when application configuration is invalid
    """

class DataError(DocClassificationError):
    """
        Raised when dataset or documents are missing or invalid
    """

class PipelineError(DocClassificationError):
    """
        Raised when a pipeline step fails unexpectedly
    """

class LabelingError(DocClassificationError):
    """
        Raised when document labeling fails
    """

class ValidationError(DocClassificationError):
    """
        Raised when input validation fails
    """

class ResourceNotFoundError(DocClassificationError):
    """
        Raised when a required file, folder or artifact is missing
    """

class ModelError(DocClassificationError):
    """
        Raised when a model or tokenizer operation fails
    """

class InferenceError(DocClassificationError):
    """
        Raised when model inference fails
    """
    
class ExternalServiceError(DocClassificationError):
    """
        Raised when an external provider or service fails
    """

class UnknownDocClassificationError(DocClassificationError):
    """
        Raised when an unexpected exception must be normalized
    """

## ============================================================
## GENERIC HELPERS
## ============================================================
def raise_project_error(
    exc_type: Type[DocClassificationError],
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
            DocClassificationError: Always
    """

    ## Build a normalized payload
    payload = details.copy() if details else {}

    ## Attach original cause metadata when available
    if cause is not None:
        payload["cause_message"] = str(cause)
        payload["cause_type"] = cause.__class__.__name__

    ## Emit a structured error log
    logger.error(
        "Doc classification error | type=%s | code=%s | message=%s | "
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
    exc_type: Type[DocClassificationError],
    message: str,
    error_code: str,
    details: Optional[Dict[str, Any]] = None,
    is_retryable: bool = False,
) -> DocClassificationError:
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
) -> UnknownDocClassificationError:
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
        "Unhandled doc-classification exception | type=%s | details=%s",
        exc.__class__.__name__,
        payload,
    )
    logger.debug("Unhandled traceback", exc_info=True)

    ## Return a normalized unknown project error
    return UnknownDocClassificationError(
        message="An unexpected doc-classification error occurred",
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
        Log and raise a configuration error for missing environment
        variables

        Args:
            vars_missing: List of missing env variable names

        Raises:
            ConfigurationError: Always
    """

    ## Keep the original explicit message style
    message = (
        "Missing environment variables (placeholders detected): "
        + ", ".join(vars_missing)
    )

    ## Emit a direct configuration log
    logger.error(message)

    ## Raise the configuration error
    raise ConfigurationError(
        message=message,
        error_code=ERROR_CODE_CONFIGURATION,
        details={"missing_variables": vars_missing},
        is_retryable=False,
    )

def log_and_raise_missing_data_folder(data_dir: Path) -> None:
    """
        Log and raise a data error when a required data folder is missing

        Args:
            data_dir: Expected data directory path

        Raises:
            DataError: Always
    """

    ## Keep the original explicit message style
    message = (
        f"Required data directory not found: {data_dir}. "
        "Fix: create the directory or update the configuration (.env)."
    )

    ## Emit a direct data log
    logger.error(message)

    ## Raise the data error
    raise DataError(
        message=message,
        error_code=ERROR_CODE_DATA,
        details={"data_dir": str(data_dir)},
        is_retryable=False,
    )

def log_and_raise_no_documents_found(folder: Path) -> None:
    """
        Log and raise a data error when no documents are found in a folder

        Args:
            folder: Folder expected to contain documents

        Raises:
            DataError: Always
    """

    ## Keep the original explicit message style
    message = (
        f"No supported document found in: {folder}. "
        "Expected at least one file with extension: .txt, .pdf, or .docx."
    )

    ## Emit a direct data log
    logger.error(message)

    ## Raise the data error
    raise DataError(
        message=message,
        error_code=ERROR_CODE_DATA,
        details={"folder": str(folder)},
        is_retryable=False,
    )

def log_and_raise_pipeline_step(step_name: str, reason: str) -> None:
    """
        Log and raise a pipeline error for a failing step

        Args:
            step_name: Name of the pipeline step
            reason: Human-readable failure reason

        Raises:
            PipelineError: Always
    """

    ## Build the pipeline failure message
    message = f"Pipeline step failed [{step_name}]: {reason}"

    ## Emit a direct pipeline log
    logger.error(message)

    ## Raise the pipeline error
    raise PipelineError(
        message=message,
        error_code=ERROR_CODE_PIPELINE,
        details={"step_name": step_name, "reason": reason},
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

def log_and_raise_labeling_error(
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise a labeling error

        Args:
            message: Human-readable labeling error message
            details: Optional labeling context
            cause: Original exception if available

        Raises:
            LabelingError: Always
    """

    ## Raise a structured labeling error
    raise_project_error(
        exc_type=LabelingError,
        message=message,
        error_code=ERROR_CODE_LABELING,
        details=details,
        cause=cause,
        is_retryable=False,
    )