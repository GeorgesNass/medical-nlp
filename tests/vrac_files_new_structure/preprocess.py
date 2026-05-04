'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Text preprocessing utilities for clinical documents (minimal normalization, no content loss)."
'''

from __future__ import annotations

## Standard library imports
import re
from typing import List

## Internal imports
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("preprocess", log_file="preprocess.log")

## ============================================================
## BASIC NORMALIZATION
## ============================================================
def normalize_whitespace(text: str) -> str:
    """
        Normalize whitespace characters

        Rules:
            - Replace multiple spaces with single space
            - Replace newlines and tabs with space
            - Preserve original text meaning

        Args:
            text: Raw text

        Returns:
            Normalized text
    """

    ## Replace newlines and tabs with spaces
    text = text.replace("\n", " ").replace("\t", " ")

    ## Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def to_lowercase(text: str) -> str:
    """
        Convert text to lowercase

        Args:
            text: Input text

        Returns:
            Lowercased text
    """
    
    return text.lower()

## ============================================================
## DE-IDENTIFICATION (OPTIONAL)
## ============================================================
def mask_numbers(text: str) -> str:
    """
        Mask standalone numbers

        Notes:
            - This is optional and conservative
            - Useful for reducing noise in models

        Args:
            text: Input text

        Returns:
            Text with numbers replaced by <NUM>
    """

    ## Replace standalone numbers (not embedded inside words)
    return re.sub(r"\b\d+\b", "<NUM>", text)

def mask_emails(text: str) -> str:
    """
        Mask email addresses

        Args:
            text: Input text

        Returns:
            Text with emails replaced by <EMAIL>
    """

    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    
    return re.sub(email_pattern, "<EMAIL>", text)

def mask_phone_numbers(text: str) -> str:
    """
        Mask phone numbers (basic patterns)

        Args:
            text: Input text

        Returns:
            Text with phone numbers replaced by <PHONE>
    """

    phone_pattern = r"\b(\+?\d{1,3}[\s-]?)?(\(?\d{2,4}\)?[\s-]?){2,4}\d{2,4}\b"
    
    return re.sub(phone_pattern, "<PHONE>", text)

## ============================================================
## TOKENIZATION UTILITIES
## ============================================================
def simple_tokenize(text: str) -> List[str]:
    """
        Tokenize text using whitespace split

        Args:
            text: Input text

        Returns:
            List of tokens
    """
    
    return text.split()

def filter_short_tokens(tokens: List[str], min_length: int = 2) -> List[str]:
    """
        Remove tokens shorter than a given length

        Args:
            tokens: List of tokens
            min_length: Minimum token length

        Returns:
            Filtered token list
    """
    
    return [t for t in tokens if len(t) >= min_length]

## ============================================================
## HIGH-LEVEL PIPELINE
## ============================================================
def preprocess_text(
    text: str,
    lowercase: bool = True,
    mask_numeric: bool = False,
    mask_email: bool = False,
    mask_phone: bool = False,
) -> str:
    """
        Apply configurable preprocessing pipeline

        Design philosophy:
            - Minimal cleaning
            - No semantic content removal
            - All transformations optional

        Steps:
            1) Normalize whitespace
            2) Optional lowercase
            3) Optional masking (numbers, emails, phone)

        Args:
            text: Raw input text
            lowercase: Whether to lowercase
            mask_numeric: Whether to mask numbers
            mask_email: Whether to mask emails
            mask_phone: Whether to mask phone numbers

        Returns:
            Preprocessed text
    """

    if text is None:
        from src.core.errors import log_and_raise_data_error
        log_and_raise_data_error(
            reason="Preprocessing failed: input text is None."
        )

    if not isinstance(text, str):
        from src.core.errors import log_and_raise_data_error
        log_and_raise_data_error(
            reason=f"Preprocessing failed: expected string, got {type(text)}."
        )

    logger.debug("Starting preprocessing")

    ## Step 1: Normalize whitespace
    text = normalize_whitespace(text)

    ## Step 2: Lowercase (optional)
    if lowercase:
        text = to_lowercase(text)

    ## Step 3: Optional masking
    if mask_numeric:
        text = mask_numbers(text)

    if mask_email:
        text = mask_emails(text)

    if mask_phone:
        text = mask_phone_numbers(text)

    if not text.strip():
        from src.core.errors import log_and_raise_data_error
        log_and_raise_data_error(
            reason="Preprocessing produced empty text after normalization."
        )

    return text