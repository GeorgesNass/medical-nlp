'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Text normalization helpers for medical documents before segmentation and embeddings."
'''

from __future__ import annotations

import re
from typing import Dict, List, Optional
import unicodedata

from src.core.errors import PipelineError
from src.core.config import get_config

## -----------------------------
## Regex helpers
## -----------------------------
_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
_ACCENT_RE = re.compile(r"\p{Mn}", re.UNICODE)

## -----------------------------
## Public API
## -----------------------------
def normalize_document_text(
    text: str,
    keep_newlines: bool = False,
) -> str:
    """
        Normalize raw document text before downstream processing.

        Strategy:
            - Remove control characters
            - Collapse repeated whitespace
            - Optionally keep newlines (useful for EDA/debug)

        Args:
            text: Raw extracted text
            keep_newlines: If True, preserve newlines while cleaning spaces

        Returns:
            Normalized text

        Raises:
            PipelineError: If normalization fails unexpectedly
    """

    try:
  
        ## Load config
        config = get_config()

        ## Use feature engineering pipeline if enabled
        if config.feature_engineering.enabled:
            return normalize_medical_text(text)
            
        safe_text = text or ""

        ## Remove control characters (common in bad exports)
        safe_text = _CONTROL_CHARS_RE.sub(" ", safe_text)

        ## Normalize whitespace
        if keep_newlines:
            ## Keep newlines but normalize other whitespace
            safe_text = safe_text.replace("\r\n", "\n").replace("\r", "\n")
            safe_text = re.sub(r"[ \t\f\v]+", " ", safe_text)
            safe_text = re.sub(r"\n{3,}", "\n\n", safe_text)
            return safe_text.strip()

        ## Fully collapse whitespace (including newlines)
        return _WHITESPACE_RE.sub(" ", safe_text).strip()

    except Exception as exc:
        raise PipelineError("Failed to normalize document text") from exc

def normalize_label_text(label: str) -> str:
    """
        Normalize a label string.

        Args:
            label: Raw label text

        Returns:
            Normalized label (lowercase, stripped)
    """

    return (label or "").strip().lower()

def normalize_meta_dict(meta: Optional[Dict[str, str]]) -> Dict[str, str]:
    """
        Normalize a metadata dictionary.

        Notes:
            - Ensures keys/values are strings
            - Trims whitespace

        Args:
            meta: Input metadata dict

        Returns:
            Normalized dict (never None)
    """

    if not meta:
        return {}

    normalized: Dict[str, str] = {}
    for k, v in meta.items():
        key = str(k).strip()
        val = str(v).strip()
        if key:
            normalized[key] = val

    return normalized

## ============================================================
## FEATURE ENGINEERING HELPERS
## ============================================================
def remove_accents(text: str) -> str:
    """
        Remove accents from text using unicode normalization

        Args:
            text: Input text

        Returns:
            Accent-free text
    """

    ## Normalize unicode and remove combining characters
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")

def normalize_whitespace(text: str) -> str:
    """
        Normalize whitespace consistently

        Args:
            text: Input text

        Returns:
            Cleaned text
    """

    ## Collapse all whitespace
    return _WHITESPACE_RE.sub(" ", text).strip()

def clean_control_characters(text: str) -> str:
    """
        Remove control characters from text

        Args:
            text: Input text

        Returns:
            Cleaned text
    """

    ## Replace control chars with space
    return _CONTROL_CHARS_RE.sub(" ", text)

def normalize_medical_text(text: str) -> str:
    """
        Apply full normalization pipeline based on config

        High-level workflow:
            1) Remove control characters
            2) Apply lowercase if enabled
            3) Remove accents if enabled
            4) Normalize whitespace

        Args:
            text: Raw input text

        Returns:
            Fully normalized text
    """

    ## Load config once
    config = get_config()

    ## Early exit if disabled
    if not config.feature_engineering.enabled:
        return text

    safe_text = text or ""

    ## control characters
    if config.feature_engineering.text_normalization_enabled:
        safe_text = clean_control_characters(safe_text)

    ## lowercase
    if config.feature_engineering.lowercase_enabled:
        safe_text = safe_text.lower()

    ## accents
    if config.feature_engineering.accent_removal_enabled:
        safe_text = remove_accents(safe_text)

    ## whitespace
    safe_text = normalize_whitespace(safe_text)

    return safe_text