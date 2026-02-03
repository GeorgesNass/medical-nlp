'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Vector similarity index for segment-level semantic matching and evidence retrieval."
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.config import CONFIG
from src.core.errors import PipelineError  ## errors wired
from src.domain.schema import DocumentSegment
from src.utils.logging_utils import get_logger


## -----------------------------
## Logger
## -----------------------------
logger = get_logger("nlp_similarity_index")


## -----------------------------
## Data structures
## -----------------------------
@dataclass
class SimilarityResult:
    """
        Result of a similarity search for one query segment

        Attributes:
            segment_id: ID of the matched indexed segment
            score: Cosine similarity score
            label: Label associated with the indexed segment
            source_file: Source document filename
            text: Text content of the matched segment
    """

    segment_id: str
    score: float
    label: str
    source_file: str
    text: str


## -----------------------------
## Similarity index
## -----------------------------
class SimilarityIndex:
    """
        In-memory similarity index for document segments

        Design:
            - Stores vectors + metadata in memory
            - Uses cosine similarity
            - One global index, labels are stored as metadata
            - Suitable for small to medium datasets (can be replaced by FAISS later)

        Typical usage:
            - Build index from labeled documents
            - Query with segments from unlabeled documents
            - Retrieve top-k similar segments as evidence
    """

    def __init__(self, normalize_vectors: bool = True) -> None:
        """
            Initialize an empty similarity index

            Args:
                normalize_vectors: Whether vectors are already L2-normalized
        """

        ## Whether vectors are assumed normalized
        self.normalize_vectors = normalize_vectors

        ## Matrix of shape [n_segments, embedding_dim]
        self._vectors: Optional[np.ndarray] = None

        ## Metadata aligned with vectors (same index)
        self._meta: List[Dict[str, str]] = []

    ## -------------------------
    ## Index building
    ## -------------------------
    def add(
        self,
        vectors: List[List[float]],
        segments: List[DocumentSegment],
        labels: List[str],
        source_files: List[str],
    ) -> None:
        """
            Add segments and their vectors to the index

            Args:
                vectors: Embedding vectors (same order as segments)
                segments: Document segments
                labels: Label associated with each segment
                source_files: Source filename for each segment
        """

        ## Sanity checks to avoid silent misalignment
        if not (len(vectors) == len(segments) == len(labels) == len(source_files)):
            raise PipelineError(
                "Vectors, segments, labels, and source_files must have the same length"
            )

        if not vectors:
            logger.warning("No vectors provided to similarity index")
            return

        ## Convert to numpy array for fast math
        try:
            vec_array = np.asarray(vectors, dtype=np.float32)
        except Exception as exc:
            raise PipelineError("Failed to convert vectors to numpy array") from exc

        ## Normalize vectors if required and not already normalized
        if not self.normalize_vectors:
            try:
                norms = np.linalg.norm(vec_array, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                vec_array = vec_array / norms
            except Exception as exc:
                raise PipelineError("Failed to normalize vectors in similarity index") from exc

        ## Append or initialize the vector matrix
        try:
            if self._vectors is None:
                self._vectors = vec_array
            else:
                self._vectors = np.vstack([self._vectors, vec_array])
        except Exception as exc:
            raise PipelineError("Failed to append vectors to similarity index") from exc

        ## Store metadata aligned with vectors
        try:
            for seg, label, src in zip(segments, labels, source_files):
                self._meta.append(
                    {
                        "segment_id": seg.segment_id,
                        "label": label,
                        "source_file": src,
                        "text": seg.text,
                    }
                )
        except Exception as exc:
            raise PipelineError("Failed to append metadata to similarity index") from exc

        logger.info(f"Added {len(vectors)} segments to similarity index")

    ## -------------------------
    ## Querying
    ## -------------------------
    def search(
        self,
        query_vectors: List[List[float]],
        top_k: Optional[int] = None,
    ) -> List[List[SimilarityResult]]:
        """
            Search the index for nearest segments for each query vector

            Args:
                query_vectors: Embedding vectors of query segments
                top_k: Number of top results to return (defaults to CONFIG)

            Returns:
                List of result lists, one list per query vector
        """

        ## Handle empty index or empty queries
        if self._vectors is None or not len(self._meta):
            logger.warning("Similarity index is empty")
            return [[] for _ in query_vectors]

        if not query_vectors:
            return []

        k = top_k or CONFIG.similarity.top_k

        ## Convert queries to numpy
        try:
            query_array = np.asarray(query_vectors, dtype=np.float32)
        except Exception as exc:
            raise PipelineError("Failed to convert query vectors to numpy array") from exc

        ## Normalize queries if needed
        if not self.normalize_vectors:
            try:
                norms = np.linalg.norm(query_array, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                query_array = query_array / norms
            except Exception as exc:
                raise PipelineError("Failed to normalize query vectors") from exc

        ## Compute cosine similarity = dot product (vectors are normalized)
        try:
            scores = np.dot(query_array, self._vectors.T)
        except Exception as exc:
            raise PipelineError("Failed to compute cosine similarity scores") from exc

        all_results: List[List[SimilarityResult]] = []

        ## Process each query independently
        for row_scores in scores:
            ## Get indices of top-k scores (descending)
            top_indices = np.argsort(row_scores)[::-1][:k]

            results: List[SimilarityResult] = []

            for idx in top_indices:
                meta = self._meta[idx]
                results.append(
                    SimilarityResult(
                        segment_id=meta["segment_id"],
                        score=float(row_scores[idx]),
                        label=meta["label"],
                        source_file=meta["source_file"],
                        text=meta["text"],
                    )
                )

            all_results.append(results)

        return all_results

    ## -------------------------
    ## Utilities
    ## -------------------------
    def size(self) -> int:
        """
            Return the number of indexed segments

            Returns:
                Number of segments in the index
        """

        return 0 if self._vectors is None else self._vectors.shape[0]
