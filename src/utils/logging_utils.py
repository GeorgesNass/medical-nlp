'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Central logging utilities: consistent formatting, console/file handlers, and get_logger helper."
'''

from __future__ import annotations

## Standard library imports
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


## Default logging settings
_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _normalize_log_level(level: Optional[str]) -> int:
    """
        Normalize a string log level to a logging numeric level

        Args:
            level: Log level string (e.g., INFO, DEBUG)

        Returns:
            Logging numeric level
    """
    ## Fallback to default level
    raw = (level or _DEFAULT_LOG_LEVEL).strip().upper()
    return getattr(logging, raw, logging.INFO)


def _ensure_dir(path: Path) -> None:
    """
        Ensure a directory exists

        Args:
            path: Directory path
    """
    ## Create folder if missing
    path.mkdir(parents=True, exist_ok=True)


def _build_log_file_path(
    logs_dir: Path,
    filename: str = "clinical_ner.log",
) -> Path:
    """
        Build an absolute log file path

        Args:
            logs_dir: Base logs directory
            filename: Log filename

        Returns:
            Full log file path
    """
    ## Ensure directory exists before building path
    _ensure_dir(logs_dir)
    return logs_dir / filename


def _create_console_handler(level: int) -> logging.Handler:
    """
        Create a stdout console logging handler

        Args:
            level: Logging numeric level

        Returns:
            Configured logging handler
    """
    ## Stream logs to stdout for Docker compatibility
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT, datefmt=_DEFAULT_DATE_FORMAT))
    return handler


def _create_file_handler(
    log_file_path: Path,
    level: int,
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> logging.Handler:
    """
        Create a rotating file logging handler

        Args:
            log_file_path: Target log file
            level: Logging numeric level
            max_bytes: Max file size before rotation
            backup_count: Number of rotated backups

        Returns:
            Configured logging handler
    """
    ## Rotate logs to avoid huge files
    handler = RotatingFileHandler(
        filename=str(log_file_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT, datefmt=_DEFAULT_DATE_FORMAT))
    return handler


def get_logger(
    name: str,
    logs_dir: Optional[Path] = None,
    level: Optional[str] = None,
    log_filename: str = "clinical_ner.log",
) -> logging.Logger:
    """
        Get a configured logger with consistent handlers and formatting

        Args:
            name: Logger name (usually __name__ or module id)
            logs_dir: Directory for log files (if None, console-only)
            level: Log level string (if None, uses default or env)
            log_filename: File name used when logs_dir is provided

        Returns:
            Configured logger
    """
    ## Reuse the same logger instance by name
    logger = logging.getLogger(name)

    ## Prevent duplicate handlers when called multiple times
    if getattr(logger, "_configured", False):
        return logger

    ## Resolve effective level from env if provided
    env_level = os.getenv("CLINICAL_NER_LOG_LEVEL")
    effective_level = level or env_level or _DEFAULT_LOG_LEVEL
    numeric_level = _normalize_log_level(effective_level)

    ## Configure base logger
    logger.setLevel(numeric_level)
    logger.propagate = False

    ## Always attach console handler
    logger.addHandler(_create_console_handler(numeric_level))

    ## Optionally attach a file handler
    if logs_dir is not None:
        log_file_path = _build_log_file_path(logs_dir=logs_dir, filename=log_filename)
        logger.addHandler(_create_file_handler(log_file_path=log_file_path, level=numeric_level))

    ## Mark as configured to avoid handler duplication
    setattr(logger, "_configured", True)

    return logger
