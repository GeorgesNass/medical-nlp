'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Embedding utilities for clinical text using sentence-transformers models (optional GPU support)."
'''

from __future__ import annotations

## Standard library imports
from dataclasses import dataclass
from typing import List, Optional

## Third-party imports
import numpy as np

## Internal imports
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("embeddings", log_file="embeddings.log")

## ============================================================
## DATA STRUCTURES
## ============================================================
@dataclass(frozen=True)
class EmbeddingConfig:
    """
        Configuration for embedding model

        Attributes:
            model_name: Sentence-transformers model name
            batch_size: Encoding batch size
            normalize: Whether to L2 normalize embeddings
            use_gpu: Whether to use GPU if available
    """

    model_name: str
    batch_size: int
    normalize: bool
    use_gpu: bool

## ============================================================
## MODEL LOADING
## ============================================================
def load_embedding_model(config: EmbeddingConfig):
    """
        Load sentence-transformers model

        Args:
            config: EmbeddingConfig

        Returns:
            Loaded model instance
    """

    ## Lazy import to avoid heavy dependency at module import time
    from sentence_transformers import SentenceTransformer

    device = "cuda" if config.use_gpu else "cpu"

    logger.info(
        "Loading embedding model=%s | device=%s",
        config.model_name,
        device,
    )

    model = SentenceTransformer(config.model_name, device=device)

    return model

## ============================================================
## ENCODING
## ============================================================
def encode_texts(
    model,
    texts: List[str],
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """
        Encode list of texts into dense embeddings

        Args:
            model: Loaded sentence-transformers model
            texts: List of texts
            batch_size: Batch size
            normalize: Whether to L2 normalize vectors

        Returns:
            Numpy array of shape (n_samples, embedding_dim)
    """

    logger.info("Encoding texts | count=%d", len(texts))

    ## Generate embeddings
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    logger.info("Embeddings shape: %s", embeddings.shape)

    return embeddings

## ============================================================
## HIGH-LEVEL PIPELINE
## ============================================================
def build_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int = 32,
    normalize: bool = True,
    use_gpu: bool = False,
) -> np.ndarray:
    """
        End-to-end embedding generation

        Steps:
            1) Load embedding model
            2) Encode texts
            3) Return embedding matrix

        Args:
            texts: List of raw texts
            model_name: Sentence-transformers model name
            batch_size: Batch size
            normalize: Whether to L2 normalize
            use_gpu: Whether to use GPU

        Returns:
            Embedding matrix (numpy array)
    """

    config = EmbeddingConfig(
        model_name=model_name,
        batch_size=batch_size,
        normalize=normalize,
        use_gpu=use_gpu,
    )

    model = load_embedding_model(config)

    return encode_texts(
        model=model,
        texts=texts,
        batch_size=batch_size,
        normalize=normalize,
    )