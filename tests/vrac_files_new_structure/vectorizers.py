'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Vectorization utilities for clinical text: TF-IDF and HashingVectorizer with consistent save/load helpers."
'''

from __future__ import annotations

## Standard library imports
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

## Third-party imports
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

## Internal imports
from src.utils.logging_utils import get_logger
from src.utils.io_utils import ensure_parent_dir, write_json, read_json

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("vectorizers", log_file="vectorizers.log")

## ============================================================
## DATA STRUCTURES
## ============================================================
@dataclass(frozen=True)
class VectorizerBundle:
    """
        Bundle containing a vectorizer and its metadata

        Attributes:
            vectorizer_type: "tfidf" or "hashing"
            params: Vectorizer parameters used for reproducibility
    """

    vectorizer_type: str
    params: Dict[str, Any]

## ============================================================
## TF-IDF VECTORIZER
## ============================================================
def build_tfidf_vectorizer(
    max_features: int = 200_000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.98,
) -> TfidfVectorizer:
    """
        Build a configured TF-IDF vectorizer

        Args:
            max_features: Maximum vocabulary size
            ngram_range: N-gram range
            min_df: Minimum document frequency
            max_df: Maximum document frequency fraction

        Returns:
            Configured TfidfVectorizer
    """

    ## Keep tokenizer default (fast, robust) and rely on preprocessing upstream
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,
    )

def fit_transform_tfidf(
    vectorizer: TfidfVectorizer,
    texts: list[str],
) -> sparse.csr_matrix:
    """
        Fit and transform texts using TF-IDF

        Args:
            vectorizer: TfidfVectorizer instance
            texts: List of input texts

        Returns:
            Sparse TF-IDF matrix
    """

    if not texts:
        from src.core.errors import log_and_raise_data_error
        log_and_raise_data_error(
            reason="Cannot fit TF-IDF: no training documents found (texts is empty)."
        )

    logger.info("Fitting TF-IDF | docs=%d", len(texts))
    X = vectorizer.fit_transform(texts)
    logger.info("TF-IDF matrix shape: %s", X.shape)

    return X


def transform_tfidf(
    vectorizer: TfidfVectorizer,
    texts: list[str],
) -> sparse.csr_matrix:
    """
        Transform texts using a fitted TF-IDF vectorizer

        Args:
            vectorizer: Fitted TfidfVectorizer
            texts: List of input texts

        Returns:
            Sparse TF-IDF matrix
    """

    logger.info("Transforming TF-IDF | docs=%d", len(texts))
    X = vectorizer.transform(texts)
    logger.info("TF-IDF matrix shape: %s", X.shape)
    
    return X

## ============================================================
## HASHING VECTORIZER
## ============================================================
def build_hashing_vectorizer(
    n_features: int = 2**20,
    ngram_range: Tuple[int, int] = (1, 2),
) -> HashingVectorizer:
    """
        Build a configured HashingVectorizer

        Notes:
            - Stateless (no fit)
            - Useful for large-scale streaming datasets

        Args:
            n_features: Number of hashing features
            ngram_range: N-gram range

        Returns:
            Configured HashingVectorizer
    """

    return HashingVectorizer(
        n_features=n_features,
        ngram_range=ngram_range,
        lowercase=False,
        alternate_sign=False,
        norm="l2",
    )

def transform_hashing(
    vectorizer: HashingVectorizer,
    texts: list[str],
) -> sparse.csr_matrix:
    """
        Transform texts using HashingVectorizer

        Args:
            vectorizer: HashingVectorizer instance
            texts: List of input texts

        Returns:
            Sparse matrix
    """

    logger.info("Transforming hashing vectors | docs=%d", len(texts))
    X = vectorizer.transform(texts)
    logger.info("Hashing matrix shape: %s", X.shape)
    
    return X

## ============================================================
## SAVE / LOAD HELPERS
## ============================================================
def _bundle_from_vectorizer(vectorizer: Any, vectorizer_type: str) -> VectorizerBundle:
    """
        Extract minimal metadata bundle from vectorizer

        Args:
            vectorizer: Vectorizer instance
            vectorizer_type: Type string

        Returns:
            VectorizerBundle
    """

    ## Store params for reproducibility
    params = vectorizer.get_params()
    
    return VectorizerBundle(vectorizer_type=vectorizer_type, params=params)

def save_vectorizer_metadata(
    vectorizer: Any,
    vectorizer_type: str,
    output_path: str | Path,
) -> Path:
    """
        Save vectorizer metadata as JSON

        Notes:
            - Actual sklearn object persistence handled by joblib outside this module

        Args:
            vectorizer: Vectorizer instance
            vectorizer_type: "tfidf" or "hashing"
            output_path: JSON output path

        Returns:
            Saved JSON path
    """

    bundle = _bundle_from_vectorizer(vectorizer, vectorizer_type)
    path = write_json({"vectorizer_type": bundle.vectorizer_type, "params": bundle.params}, output_path)
    logger.info("Saved vectorizer metadata: %s", path)
    
    return path

def load_vectorizer_metadata(metadata_path: str | Path) -> VectorizerBundle:
    """
        Load vectorizer metadata from JSON

        Args:
            metadata_path: JSON path

        Returns:
            VectorizerBundle
    """

    payload = read_json(metadata_path)
    return VectorizerBundle(
        vectorizer_type=str(payload["vectorizer_type"]),
        params=dict(payload["params"]),
    )

def create_vectorizer_from_metadata(bundle: VectorizerBundle) -> Any:
    """
        Recreate vectorizer instance from metadata

        Args:
            bundle: VectorizerBundle

        Returns:
            Vectorizer instance

        Raises:
            ValueError: If vectorizer type is unsupported
    """

    if bundle.vectorizer_type == "tfidf":
        return TfidfVectorizer(**bundle.params)

    if bundle.vectorizer_type == "hashing":
        return HashingVectorizer(**bundle.params)

    raise ValueError(f"Unsupported vectorizer_type: {bundle.vectorizer_type}")