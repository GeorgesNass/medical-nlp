'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Tokenization utilities for Clinical NER: simple, spaCy and Hugging Face tokenizers."
'''

from __future__ import annotations

## Standard library imports
from typing import Iterable, List
    
try:
    import spacy  # type: ignore
except Exception as exc:
    msg = "spaCy library is required for spaCy tokenization"
    logger.error(msg)
    raise ConfigurationError(msg) from exc
try:
    from transformers import AutoTokenizer  # type: ignore
except Exception as exc:
    msg = "transformers library is required for HF tokenization"
    logger.error(msg)
    raise ConfigurationError(msg) from exc

## Centralized errors and logging
from src.core.config import get_config
from src.core.errors import ConfigurationError
from src.nlp.normalization import normalize_clinical_text
from src.utils.logging_utils import get_logger
from src.utils.utils import ensure_str

## Module-level logger
logger = get_logger(name="clinical_ner.tokenizer")

def simple_tokenize(text: str) -> List[str]:
    """
        Tokenize text using a simple whitespace strategy

        Args:
            text: Input text

        Returns:
            List of tokens
    """
    
    ## Basic whitespace split
    config = get_config()

    if config.feature_engineering.enabled:
        text = normalize_clinical_text(text)

    return ensure_str(text).split()

def spacy_tokenize(text: str, model: str = "en_core_web_sm") -> List[str]:
    """
        Tokenize text using spaCy

        Args:
            text: Input text
            model: spaCy model name

        Returns:
            List of tokens

        Raises:
            ConfigurationError: If spaCy is not installed or model is missing
    """

    try:
        nlp = spacy.load(model)
    except Exception as exc:
        msg = f"spaCy model not found: {model}"
        logger.error(msg)
        raise ConfigurationError(msg) from exc

    ## Tokenize text
    config = get_config()

    if config.feature_engineering.enabled:
        text = normalize_clinical_text(text)

    doc = nlp(ensure_str(text))
    return [tok.text for tok in doc]

def hf_tokenize(
    text: str,
    model_name: str = "distilbert-base-multilingual-cased",
) -> List[str]:
    """
        Tokenize text using a Hugging Face tokenizer

        Args:
            text: Input text
            model_name: Hugging Face tokenizer name

        Returns:
            List of tokens

        Raises:
            ConfigurationError: If transformers is not installed
    """
    
    ## Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ## Tokenize text
    config = get_config()

    if config.feature_engineering.enabled:
        text = normalize_clinical_text(text)

    tokens = tokenizer.tokenize(ensure_str(text))
    
    return tokens

def batch_tokenize(
    texts: Iterable[str],
    method: str = "simple",
    **kwargs,
) -> List[List[str]]:
    """
        Tokenize a batch of texts with a selected strategy

        Args:
            texts: Iterable of input texts
            method: Tokenization method (simple, spacy, hf)
            **kwargs: Additional tokenizer arguments

        Returns:
            List of token lists

        Raises:
            ConfigurationError: If method is unknown
    """
    
    ## Normalize method
    m = ensure_str(method).strip().lower()

    if m == "simple":
        return [simple_tokenize(t) for t in texts]

    if m == "spacy":
        return [spacy_tokenize(t, **kwargs) for t in texts]

    if m == "hf":
        return [hf_tokenize(t, **kwargs) for t in texts]

    msg = f"Unknown tokenization method: {method}"
    logger.error(msg)
    raise ConfigurationError(msg)

## ============================================================
## TOKEN FEATURES
## ============================================================
def compute_token_features(tokens: List[str]) -> dict:
    """
        Compute token-level features

        Args:
            tokens: List of tokens

        Returns:
            Dictionary with token statistics
    """

    return {
        "token_count": len(tokens),
        "avg_token_length": (
            sum(len(t) for t in tokens) / max(1, len(tokens))
        ),
    }