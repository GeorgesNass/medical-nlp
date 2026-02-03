'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized custom exceptions and helpers for clean, user-friendly pipeline errors."
'''

from __future__ import annotations

from pathlib import Path
from typing import List

from src.utils.logging_utils import get_logger


## ============================================================
## LOGGER
## ============================================================
logger = get_logger("errors")


## ============================================================
## CUSTOM EXCEPTIONS
## ============================================================
class ConfigurationError(RuntimeError):
    """
        Raised when application configuration is invalid.
    """
    pass


class DataError(RuntimeError):
    """
        Raised when dataset/files required for the pipeline are missing or invalid.
    """
    pass


class PipelineError(RuntimeError):
    """
        Raised when a pipeline step fails unexpectedly.
    """
    pass


class LabelingError(RuntimeError):
    """
        Raised when document labeling fails.
    """
    pass


## ============================================================
## HELPERS
## ============================================================
def log_and_raise_missing_env(vars_missing: List[str]) -> None:
    """
        Log and raise a configuration error for missing environment variables.

        Args:
            vars_missing: List of missing env variable names.

        Raises:
            ConfigurationError: Always raised after logging.
    """

    message = (
        "Missing environment variables (placeholders detected): "
        + ", ".join(vars_missing)
    )

    logger.error(message)
    raise ConfigurationError(message)


def log_and_raise_missing_data_folder(data_dir: Path) -> None:
    """
        Log and raise a data error when a required data folder is missing.

        Args:
            data_dir: Expected data directory path.

        Raises:
            DataError: Always raised after logging.
    """

    message = (
        f"Required data directory not found: {data_dir}. "
        "Fix: create the directory or update the configuration (.env)."
    )

    logger.error(message)
    raise DataError(message)


def log_and_raise_no_documents_found(folder: Path) -> None:
    """
        Log and raise a data error when no documents are found in a folder.

        Args:
            folder: Folder expected to contain documents.

        Raises:
            DataError: Always raised after logging.
    """

    message = (
        f"No supported document found in: {folder}. "
        "Expected at least one file with extension: .txt, .pdf, or .docx."
    )

    logger.error(message)
    raise DataError(message)


def log_and_raise_pipeline_step(step_name: str, reason: str) -> None:
    """
        Log and raise a pipeline error for a failing step.

        Args:
            step_name: Name of the pipeline step.
            reason: Human-readable failure reason.

        Raises:
            PipelineError: Always raised after logging.
    """

    message = f"Pipeline step failed [{step_name}]: {reason}"
    logger.error(message)
    raise PipelineError(message)
