'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized logging utilities (console + optional file), designed to avoid circular imports and duplicate handlers."
'''

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


## ============================================================
## INTERNAL HELPERS
## ============================================================
def _get_env(name: str, default: str) -> str:
    """
        Read an environment variable safely

        Args:
            name: Environment variable name
            default: Default value if missing

        Returns:
            Environment value as stripped string
    """
    return os.getenv(name, default).strip()


def _resolve_log_level() -> int:
    """
        Resolve log level from environment variables

        Supported:
            LOG_LEVEL: DEBUG | INFO | WARNING | ERROR | CRITICAL

        Returns:
            Logging level integer
    """
    level_name = _get_env("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def _resolve_logs_dir() -> Path:
    """
        Resolve logs directory from environment variables

        Supported:
            LOGS_DIR: folder path (default: logs)

        Returns:
            Absolute path to logs directory
    """
    return Path(_get_env("LOGS_DIR", "logs")).expanduser().resolve()


def _build_formatter() -> logging.Formatter:
    """
        Build default log formatter

        Returns:
            Logging formatter
    """
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _has_handler(logger: logging.Logger, handler_type: type) -> bool:
    """
        Check if logger already has a handler of the given type

        Args:
            logger: Target logger
            handler_type: Handler class type

        Returns:
            True if found, else False
    """
    return any(isinstance(h, handler_type) for h in logger.handlers)


## ============================================================
## PUBLIC API
## ============================================================
def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
        Create or return a configured logger

        Design:
            - No project imports to avoid circular dependencies
            - Prevents duplicate handlers when called multiple times
            - Console handler always enabled
            - File handler enabled only if log_file is provided

        Args:
            name: Logger name
            log_file: Optional log filename (stored in LOGS_DIR)

        Returns:
            Configured logger instance
    """
    logger = logging.getLogger(name)

    ## Do not propagate to root to avoid duplicated logs in some environments
    logger.propagate = False
    logger.setLevel(_resolve_log_level())

    formatter = _build_formatter()
    level = _resolve_log_level()

    ## Console handler
    if not _has_handler(logger, logging.StreamHandler):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    ## Optional file handler
    if log_file and not _has_handler(logger, logging.FileHandler):
        logs_dir = _resolve_logs_dir()
        logs_dir.mkdir(parents=True, exist_ok=True)

        file_path = logs_dir / log_file
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
