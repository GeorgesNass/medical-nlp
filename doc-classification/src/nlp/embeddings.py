'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Embedding backend to encode text segments into vectors (CPU/GPU) for similarity-based labeling."
'''

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.core.config import CONFIG
from src.core.errors import PipelineError
from src.utils.logging_utils import get_logger


## -----------------------------
## Logger
## -----------------------------
logger = get_logger("nlp_embeddings")


## -----------------------------
## Model registry
## -----------------------------
## Notes:
## - We use sentence-transformers for a stable encode() API.
## - Default model ids can be overridden via env vars to avoid code changes.
## - If you have a preferred medical French model, set:
##     EMBEDDING_MODEL_ID_SENTENCE_CAMEMBERT=...
##     EMBEDDING_MODEL_ID_DRBERT=...
_DEFAULT_MODEL_IDS: Dict[str, str] = {
    ## Generic strong French/multilingual sentence embedding baseline
    ## Replace via env var if needed.
    "sentence_camembert": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ## Medical French model placeholder; override recommended
    "drbert": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}


def _resolve_model_id(model_name: str) -> str:
    """
        Resolve the HuggingFace model id for the configured preset name.

        Args:
            model_name: Preset name (e.g., sentence_camembert, drbert)

        Returns:
            HuggingFace model id
    """

    ## Allow override by environment variables
    if model_name == "sentence_camembert":
        override = os.getenv("EMBEDDING_MODEL_ID_SENTENCE_CAMEMBERT", "").strip()
        if override:
            return override

    if model_name == "drbert":
        override = os.getenv("EMBEDDING_MODEL_ID_DRBERT", "").strip()
        if override:
            return override

    ## Fallback to built-in defaults
    return _DEFAULT_MODEL_IDS.get(model_name, _DEFAULT_MODEL_IDS["sentence_camembert"])


def _resolve_device(use_gpu: bool) -> str:
    """
        Resolve device string for sentence-transformers.

        Args:
            use_gpu: Whether GPU should be used

        Returns:
            Device string ('cuda' or 'cpu')
    """

    ## If GPU is requested, attempt to use cuda (if available)
    if use_gpu:
        try:
            import torch  # pylint: disable=import-error
        except Exception:
            return "cpu"

        return "cuda" if torch.cuda.is_available() else "cpu"

    return "cpu"


@dataclass(frozen=True)
class EmbeddingConfig:
    """
        Resolved embedding configuration snapshot.

        Attributes:
            model_id: HuggingFace model id
            device: Device string used by the backend
            batch_size: Batch size for encoding
            normalize: Whether embeddings are L2-normalized
    """

    model_id: str
    device: str
    batch_size: int
    normalize: bool


class EmbeddingBackend:
    """
        Embedding backend wrapper around sentence-transformers.

        Responsibilities:
            - Load the embedding model once
            - Encode a list of texts into vectors
            - Handle CPU/GPU configuration via CONFIG
            - Optionally normalize vectors
    """

    _singleton_model = None
    _singleton_cfg: Optional[EmbeddingConfig] = None

    def __init__(self) -> None:
        """
            Initialize backend using global CONFIG.
        """

        ## Resolve config values from CONFIG
        model_name = str(CONFIG.embeddings.model_name).strip()
        model_id = _resolve_model_id(model_name)
        device = _resolve_device(bool(CONFIG.embeddings.use_gpu))

        ## Defensive parsing for batch size
        try:
            batch_size = int(CONFIG.embeddings.batch_size)
        except Exception as exc:
            raise PipelineError("Invalid embeddings batch size in CONFIG") from exc

        ## Defensive parsing for normalize
        normalize = bool(CONFIG.embeddings.normalize)

        self.cfg = EmbeddingConfig(
            model_id=model_id,
            device=device,
            batch_size=batch_size,
            normalize=normalize,
        )

        ## Lazy-load the model (singleton shared between instances)
        self._ensure_model_loaded()

        ## Expose device usage for CLI/debug
        self.device = device
        self.use_gpu = bool(str(device).startswith("cuda"))

    def _ensure_model_loaded(self) -> None:
        """
            Lazy-load a shared sentence-transformers model.
        """

        ## Reuse singleton if same config
        if (
            self.__class__._singleton_model is not None
            and self.__class__._singleton_cfg is not None
            and self.__class__._singleton_cfg == self.cfg
        ):
            return

        ## Load model via sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:
            raise PipelineError(
                "sentence-transformers is required. Install: pip install sentence-transformers"
            ) from exc

        ## Create and store singleton model
        try:
            logger.info(
                f"Loading embedding model: {self.cfg.model_id} (device={self.cfg.device})"
            )
            model = SentenceTransformer(self.cfg.model_id, device=self.cfg.device)
        except Exception as exc:
            raise PipelineError(f"Failed to load embedding model: {self.cfg.model_id}") from exc

        self.__class__._singleton_model = model
        self.__class__._singleton_cfg = self.cfg

    @property
    def model(self):
        """
            Return the loaded sentence-transformers model (singleton).
        """

        if self.__class__._singleton_model is None:
            raise PipelineError("Embedding model is not loaded")
        return self.__class__._singleton_model

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
            Encode a list of texts into embedding vectors.

            Args:
                texts: List of input texts

            Returns:
                List of embedding vectors (as Python lists)

            Raises:
                PipelineError: If encoding fails
        """

        ## Handle empty input safely
        if not texts:
            return []

        ## Clean inputs to avoid model crashes on None
        clean_texts = [(t or "").strip() for t in texts]
        if not any(clean_texts):
            return []

        ## Perform encoding
        try:
            embeddings = self.model.encode(
                clean_texts,
                batch_size=self.cfg.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.cfg.normalize,
            )
        except TypeError:
            ## Older sentence-transformers may not support normalize_embeddings
            ## We fallback to manual normalization if needed.
            try:
                embeddings = self.model.encode(
                    clean_texts,
                    batch_size=self.cfg.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            except Exception as exc:
                raise PipelineError("Failed to encode texts into embeddings") from exc
        except Exception as exc:
            raise PipelineError("Failed to encode texts into embeddings") from exc

        ## Ensure numpy array
        try:
            emb_array = np.asarray(embeddings, dtype=np.float32)
        except Exception as exc:
            raise PipelineError("Failed to convert embeddings to numpy array") from exc

        ## Manual normalization if required and not provided by backend
        if self.cfg.normalize:
            try:
                norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                emb_array = emb_array / norms
            except Exception as exc:
                raise PipelineError("Failed to normalize embeddings") from exc

        ## Convert to list-of-lists for downstream compatibility
        return emb_array.tolist()
