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
from typing import List, Optional

from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("errors", log_file="errors.log")

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

## ============================================================
## HELPERS
## ============================================================
def log_and_raise_missing_env(vars_missing: List[str]) -> None:
    """
        Log and raise a configuration error for missing environment variables

        Args:
            vars_missing: List of missing env variable names

        Raises:
            ConfigurationError: Always raised after logging
    """

    message = (
        "Missing environment variables (placeholders detected): "
        + ", ".join(vars_missing)
    )
    logger.error(message)

    raise ConfigurationError(message)

def log_and_raise_missing_folder(folder: Path, reason: Optional[str] = None) -> None:
    """
        Log and raise a data error when a required folder is missing

        Args:
            folder: Expected folder path
            reason: Optional human-readable explanation

        Raises:
            DataError: Always raised after logging
    """

    message = f"Required folder not found: {folder}"
    if reason:
        message = f"{message} | {reason}"

    logger.error(message)

    raise DataError(message)

def log_and_raise_missing_file(file_path: Path, reason: Optional[str] = None) -> None:
    """
        Log and raise a data error when a required file is missing

        Args:
            file_path: Expected file path
            reason: Optional human-readable explanation

        Raises:
            DataError: Always raised after logging
    """

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

    message = f"Data error: {reason}"
    logger.error(message)
    
    raise DataError(message)
