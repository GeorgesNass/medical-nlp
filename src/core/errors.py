'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized custom exceptions and helpers for clean, user-friendly Clinical NER errors."
'''

from __future__ import annotations

## ---------------------------------------------------------------------------
## Standard library imports
## ---------------------------------------------------------------------------

from pathlib import Path
from typing import Iterable

## ---------------------------------------------------------------------------
## Project imports
## ---------------------------------------------------------------------------

from src.utils.logging_utils import get_logger


## ---------------------------------------------------------------------------
## Logger
## ---------------------------------------------------------------------------

logger = get_logger(name="clinical_ner.errors")


## ---------------------------------------------------------------------------
## Base exception
## ---------------------------------------------------------------------------

class ClinicalNERError(RuntimeError):
    """
        Base exception for all Clinical NER errors

        Notes:
            - Always logged before being raised
            - Designed to be caught at pipeline / CLI level
    """


## ---------------------------------------------------------------------------
## Specialized exceptions
## ---------------------------------------------------------------------------

class ConfigurationError(ClinicalNERError):
    """
        Raised when application configuration is invalid or incomplete
    """


class DataError(ClinicalNERError):
    """
        Raised when required data is missing, malformed, or inconsistent
    """


class PipelineError(ClinicalNERError):
    """
        Raised when a pipeline execution step fails
    """


## ---------------------------------------------------------------------------
## Helpers
## ---------------------------------------------------------------------------

def log_and_raise_missing_env(vars_missing: Iterable[str]) -> None:
    """
        Log and raise a configuration error for missing environment variables

        Args:
            vars_missing: Iterable of missing environment variable names

        Raises:
            ConfigurationError
    """
    missing = sorted(set(vars_missing))
    message = (
        "Missing required environment variables: "
        + ", ".join(missing)
        + ". Fix: define them in your .env file."
    )

    logger.error(message)
    raise ConfigurationError(message)


def log_and_raise_missing_raw_data(raw_dir: Path) -> None:
    """
        Log and raise a data error when no raw dataset file is found

        Args:
            raw_dir: Directory where raw dataset files are expected

        Raises:
            DataError
    """
    message = (
        f"No raw data file found in: {raw_dir}. "
        "Expected at least one file with extension: .csv, .jsonl, or .json. "
        "Fix: put your dataset into data/raw/ "
        "or configure the RAW_DATA_DIR variable."
    )

    logger.error(message)
    raise DataError(message)


def log_and_raise_pipeline_error(step: str, reason: str) -> None:
    """
        Log and raise a pipeline execution error

        Args:
            step: Pipeline step name
            reason: Human-readable failure reason

        Raises:
            PipelineError
    """
    message = f"Pipeline step '{step}' failed: {reason}"
    logger.error(message)
    raise PipelineError(message)
