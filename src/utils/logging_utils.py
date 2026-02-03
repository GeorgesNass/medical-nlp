'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized logging utilities (console + optional file), without circular imports."
'''

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


## -----------------------------
## Internal helpers
## -----------------------------
def _get_log_level() -> int:
    """
        Resolve log level from environment variables.

        Returns:
            Logging level integer.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    return getattr(logging, level_name, logging.INFO)


def _get_logs_dir() -> Path:
    """
        Resolve logs directory from environment variables.

        Returns:
            Absolute path to logs directory.
    """
    return Path(os.getenv("LOGS_DIR", "logs")).expanduser().resolve()


def _build_formatter() -> logging.Formatter:
    """
        Build a default log formatter.

        Returns:
            logging.Formatter instance.
    """
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


## -----------------------------
## Public API
## -----------------------------
def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
        Create or return a configured logger.

        Notes:
            - Avoids circular imports by relying only on environment variables.
            - Prevents duplicate handlers when called multiple times.

        Args:
            name: Logger name.
            log_file: Optional log filename (stored in LOGS_DIR).

        Returns:
            Configured logger instance.
    """
    logger = logging.getLogger(name)

    ## Avoid duplicate handlers
    if logger.handlers:
        return logger

    level = _get_log_level()
    logger.setLevel(level)
    logger.propagate = False

    formatter = _build_formatter()

    ## Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    ## Optional file handler
    if log_file:
        logs_dir = _get_logs_dir()
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logs_dir / log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
