'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Text normalization utilities for Clinical NER: unicode cleanup, spacing, casing, and accent handling."
'''

from __future__ import annotations

## Standard library imports
import re
import unicodedata
from typing import Iterable

## Centralized errors and logging
from src.core.errors import DataError
from src.utils.logging_utils import get_logger

## Generic utilities
from src.utils.utils import ensure_str


## Module-level logger
logger = get_logger(name="clinical_ner.normalization")


## Regex patterns for normalization
_MULTI_SPACE_RE = re.compile(r"\s+")
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")


def normalize_unicode(text: str) -> str:
    """
        Normalize unicode representation and remove zero-width characters

        Args:
            text: Raw input text

        Returns:
            Normalized text
    """
    ## Ensure input is string
    raw = ensure_str(text)

    ## Apply NFC normalization for stable unicode
    normalized = unicodedata.normalize("NFC", raw)

    ## Remove zero-width characters
    normalized = _ZERO_WIDTH_RE.sub("", normalized)

    return normalized


def strip_accents(text: str) -> str:
    """
        Remove accents from a text string

        Args:
            text: Input text

        Returns:
            Text without accents
    """
    ## Normalize to NFD then drop combining marks
    normalized = unicodedata.normalize("NFD", ensure_str(text))
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_whitespace(text: str) -> str:
    """
        Normalize whitespace by collapsing multiple spaces and trimming ends

        Args:
            text: Input text

        Returns:
            Text with normalized whitespace
    """
    ## Replace any whitespace run with a single space
    collapsed = _MULTI_SPACE_RE.sub(" ", ensure_str(text))
    return collapsed.strip()


def normalize_case(text: str, mode: str = "lower") -> str:
    """
        Normalize text casing

        Args:
            text: Input text
            mode: Case mode (lower, upper, none)

        Returns:
            Case-normalized text

        Raises:
            DataError: If mode is invalid
    """
    ## Normalize mode
    m = ensure_str(mode).strip().lower()

    if m == "lower":
        return ensure_str(text).lower()

    if m == "upper":
        return ensure_str(text).upper()

    if m == "none":
        return ensure_str(text)

    msg = f"Invalid case mode: {mode}"
    logger.error(msg)
    raise DataError(msg)


def normalize_text(
    text: str,
    to_case: str = "lower",
    remove_accents: bool = False,
    normalize_spaces: bool = True,
) -> str:
    """
        Apply standard normalization pipeline to a text string

        Args:
            text: Raw input text
            to_case: Case normalization mode (lower, upper, none)
            remove_accents: Whether to strip accents
            normalize_spaces: Whether to collapse whitespace

        Returns:
            Normalized text
    """
    ## Normalize unicode first
    out = normalize_unicode(text)

    ## Optionally remove accents
    if remove_accents:
        out = strip_accents(out)

    ## Normalize case
    out = normalize_case(out, mode=to_case)

    ## Optionally normalize spaces
    if normalize_spaces:
        out = normalize_whitespace(out)

    return out


def normalize_corpus(
    texts: Iterable[str],
    to_case: str = "lower",
    remove_accents: bool = False,
    normalize_spaces: bool = True,
) -> list[str]:
    """
        Normalize a list or iterable of texts

        Args:
            texts: Iterable of raw texts
            to_case: Case normalization mode
            remove_accents: Whether to strip accents
            normalize_spaces: Whether to collapse whitespace

        Returns:
            List of normalized texts
    """
    ## Apply normalization to each element
    return [
        normalize_text(
            t,
            to_case=to_case,
            remove_accents=remove_accents,
            normalize_spaces=normalize_spaces,
        )
        for t in texts
    ]
