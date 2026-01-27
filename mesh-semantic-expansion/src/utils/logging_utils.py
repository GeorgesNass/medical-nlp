'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized logging utilities with consistent formatting and file/console handlers."
'''

import logging
import sys
from pathlib import Path
from typing import Optional

from src.core.config import get_settings


## ============================================================
## LOGGER FACTORY
## ============================================================

def get_logger(
    name: str,
    level: Optional[int] = None,
) -> logging.Logger:
    """
    Create or retrieve a configured logger.

    Design choices:
        - One logger per module name
        - Console + file handlers
        - No duplicate handlers
        - UTF-8 safe

    Args:
        name (str): Logger name (usually __name__ or module label).
        level (Optional[int]): Optional logging level override.

    Returns:
        logging.Logger: Configured logger instance.
    """

    settings = get_settings()
    log_level = level if level is not None else settings.log_level

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    ## Avoid duplicate handlers (important for FastAPI reloads / tests)
    if logger.handlers:
        return logger

    ## --------------------------------------------------------
    ## FORMATTER
    ## --------------------------------------------------------

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ## --------------------------------------------------------
    ## CONSOLE HANDLER
    ## --------------------------------------------------------

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    ## --------------------------------------------------------
    ## FILE HANDLER (OPTIONAL)
    ## --------------------------------------------------------

    if settings.logs_dir:
        logs_dir = Path(settings.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)

        file_path = logs_dir / f"{name.replace('.', '_')}.log"
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
