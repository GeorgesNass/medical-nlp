'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized custom exceptions and structured helpers for the mesh semantic expansion pipeline."
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
ERROR_CODE_RESOURCE_NOT_FOUND = "resource_not_found"
ERROR_CODE_ONTOLOGY_LOADING = "ontology_loading_error"
ERROR_CODE_MESH_LOADING = "mesh_loading_error"
ERROR_CODE_EXPANSION = "semantic_expansion_error"
ERROR_CODE_EMBEDDING = "embedding_error"
ERROR_CODE_SIMILARITY = "similarity_error"
ERROR_CODE_RETRIEVAL = "retrieval_error"
ERROR_CODE_RANKING = "ranking_error"
ERROR_CODE_EXTERNAL_SERVICE = "external_service_error"
ERROR_CODE_PIPELINE = "pipeline_error"
ERROR_CODE_INTERNAL = "internal_error"

## ============================================================
## BASE EXCEPTION
## ============================================================
class MeshSemanticExpansionError(RuntimeError):
    """
        Base exception for the mesh semantic expansion pipeline

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
class ConfigurationError(MeshSemanticExpansionError):
    """
        Raised when application configuration is invalid
    """

class ValidationError(MeshSemanticExpansionError):
    """
        Raised when request payload or parameters are invalid
    """

class DataError(MeshSemanticExpansionError):
    """
        Raised when input data or output writing fails
    """

class ResourceNotFoundError(MeshSemanticExpansionError):
    """
        Raised when a required file, folder or artifact is missing
    """

class OntologyLoadingError(MeshSemanticExpansionError):
    """
        Raised when ontology or terminology resources cannot be loaded
    """

class MeshLoadingError(MeshSemanticExpansionError):
    """
        Raised when MeSH resources cannot be loaded correctly
    """

class SemanticExpansionError(MeshSemanticExpansionError):
    """
        Raised when semantic expansion logic fails
    """

class EmbeddingError(MeshSemanticExpansionError):
    """
        Raised when embedding generation fails
    """

class SimilarityError(MeshSemanticExpansionError):
    """
        Raised when similarity computation fails
    """

class RetrievalError(MeshSemanticExpansionError):
    """
        Raised when candidate retrieval fails
    """

class RankingError(MeshSemanticExpansionError):
    """
        Raised when ranking or reranking fails
    """

class ExternalServiceError(MeshSemanticExpansionError):
    """
        Raised when a remote provider or external service fails
    """

class PipelineError(MeshSemanticExpansionError):
    """
        Raised when pipeline orchestration fails
    """

class UnknownMeshSemanticExpansionError(MeshSemanticExpansionError):
    """
        Raised when an unexpected exception must be normalized
    """

## ============================================================
## GENERIC HELPERS
## ============================================================
def raise_project_error(
    exc_type: Type[MeshSemanticExpansionError],
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
            MeshSemanticExpansionError: Always
    """

    ## Build a normalized payload
    payload = details.copy() if details else {}

    ## Attach original cause metadata when available
    if cause is not None:
        payload["cause_message"] = str(cause)
        payload["cause_type"] = cause.__class__.__name__

    ## Emit a structured error log
    logger.error(
        "Mesh semantic expansion error | type=%s | code=%s | message=%s | "
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
    exc_type: Type[MeshSemanticExpansionError],
    message: str,
    error_code: str,
    details: Optional[Dict[str, Any]] = None,
    is_retryable: bool = False,
) -> MeshSemanticExpansionError:
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
) -> UnknownMeshSemanticExpansionError:
    """
        Normalize an unexpected exception into a project-specific error

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
        "Unhandled mesh-semantic-expansion exception | type=%s | details=%s",
        exc.__class__.__name__,
        payload,
    )
    logger.debug("Unhandled traceback", exc_info=True)

    ## Return a normalized unknown project error
    return UnknownMeshSemanticExpansionError(
        message="An unexpected mesh-semantic-expansion error occurred",
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
    """

    ## Build the explicit configuration error message
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
    """

    ## Normalize path for payload stability
    normalized_path = str(Path(path))

    ## Raise structured missing resource error
    raise_project_error(
        exc_type=ResourceNotFoundError,
        message=f"{resource_name} not found",
        error_code=ERROR_CODE_RESOURCE_NOT_FOUND,
        details={"path": normalized_path},
        is_retryable=False,
    )

def log_and_raise_ontology_loading_error(
    source_name: str,
    reason: str,
    *,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise an ontology loading error

        Args:
            source_name: Ontology or terminology source name
            reason: Human-readable failure reason
            cause: Original exception if available
    """

    ## Build ontology loading failure message
    message = f"Ontology loading failed for source: {source_name}"

    ## Raise structured ontology loading error
    raise_project_error(
        exc_type=OntologyLoadingError,
        message=message,
        error_code=ERROR_CODE_ONTOLOGY_LOADING,
        details={"source_name": source_name, "reason": reason},
        cause=cause,
        is_retryable=False,
    )

def log_and_raise_mesh_loading_error(
    resource_name: str,
    reason: str,
    *,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise a MeSH loading error

        Args:
            resource_name: MeSH resource identifier
            reason: Human-readable failure reason
            cause: Original exception if available
    """

    ## Build MeSH loading failure message
    message = f"MeSH resource loading failed: {resource_name}"

    ## Raise structured MeSH loading error
    raise_project_error(
        exc_type=MeshLoadingError,
        message=message,
        error_code=ERROR_CODE_MESH_LOADING,
        details={"resource_name": resource_name, "reason": reason},
        cause=cause,
        is_retryable=False,
    )

def log_and_raise_semantic_expansion_error(
    query: str,
    reason: str,
    *,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise a semantic expansion error

        Args:
            query: Query or concept being expanded
            reason: Human-readable failure reason
            cause: Original exception if available
    """

    ## Build semantic expansion failure message
    message = "Semantic expansion failed"

    ## Raise structured semantic expansion error
    raise_project_error(
        exc_type=SemanticExpansionError,
        message=message,
        error_code=ERROR_CODE_EXPANSION,
        details={"query": query, "reason": reason},
        cause=cause,
        is_retryable=False,
    )

def log_and_raise_embedding_error(
    target: str,
    reason: str,
    *,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise an embedding error

        Args:
            target: Target item for embedding generation
            reason: Human-readable failure reason
            cause: Original exception if available
    """

    ## Build embedding failure message
    message = f"Embedding generation failed for target: {target}"

    ## Raise structured embedding error
    raise_project_error(
        exc_type=EmbeddingError,
        message=message,
        error_code=ERROR_CODE_EMBEDDING,
        details={"target": target, "reason": reason},
        cause=cause,
        is_retryable=True,
    )

def log_and_raise_similarity_error(
    left_item: str,
    right_item: str,
    reason: str,
    *,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise a similarity error

        Args:
            left_item: Left comparison item
            right_item: Right comparison item
            reason: Human-readable failure reason
            cause: Original exception if available
    """

    ## Build similarity failure message
    message = "Similarity computation failed"

    ## Raise structured similarity error
    raise_project_error(
        exc_type=SimilarityError,
        message=message,
        error_code=ERROR_CODE_SIMILARITY,
        details={
            "left_item": left_item,
            "right_item": right_item,
            "reason": reason,
        },
        cause=cause,
        is_retryable=False,
    )

def log_and_raise_retrieval_error(
    query: str,
    reason: str,
    *,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise a retrieval error

        Args:
            query: Query used for candidate retrieval
            reason: Human-readable failure reason
            cause: Original exception if available
    """

    ## Build retrieval failure message
    message = "Candidate retrieval failed"

    ## Raise structured retrieval error
    raise_project_error(
        exc_type=RetrievalError,
        message=message,
        error_code=ERROR_CODE_RETRIEVAL,
        details={"query": query, "reason": reason},
        cause=cause,
        is_retryable=True,
    )

def log_and_raise_ranking_error(
    query: str,
    reason: str,
    *,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise a ranking error

        Args:
            query: Query used for ranking
            reason: Human-readable failure reason
            cause: Original exception if available
    """

    ## Build ranking failure message
    message = "Candidate ranking failed"

    ## Raise structured ranking error
    raise_project_error(
        exc_type=RankingError,
        message=message,
        error_code=ERROR_CODE_RANKING,
        details={"query": query, "reason": reason},
        cause=cause,
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

    ## Raise structured validation error
    raise_project_error(
        exc_type=ValidationError,
        message=message,
        error_code=ERROR_CODE_VALIDATION,
        details=details,
        is_retryable=False,
    )

def log_and_raise_external_service_error(
    service_name: str,
    reason: str,
    *,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise an external service error

        Args:
            service_name: External service identifier
            reason: Human-readable failure reason
            cause: Original exception if available
    """

    ## Build external service failure message
    message = f"External semantic service failed: {service_name}"

    ## Raise structured external service error
    raise_project_error(
        exc_type=ExternalServiceError,
        message=message,
        error_code=ERROR_CODE_EXTERNAL_SERVICE,
        details={"service_name": service_name, "reason": reason},
        cause=cause,
        is_retryable=True,
    )

def log_and_raise_pipeline_error(
    step_name: str,
    reason: str,
    *,
    cause: Optional[Exception] = None,
) -> None:
    """
        Log and raise a pipeline error

        Args:
            step_name: Pipeline step name
            reason: Human-readable failure reason
            cause: Original exception if available
    """

    ## Build pipeline failure message
    message = f"Pipeline step failed [{step_name}]: {reason}"

    ## Raise structured pipeline error
    raise_project_error(
        exc_type=PipelineError,
        message=message,
        error_code=ERROR_CODE_PIPELINE,
        details={"step_name": step_name, "reason": reason},
        cause=cause,
        is_retryable=False,
    )