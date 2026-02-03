'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Hybrid labeling strategy: similarity-based evidence + optional binary classifiers per label."
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.core.config import LABELS
from src.core.errors import LabelingError
from src.labeling.label_definitions import LabelDefinition
from src.labeling.similarity_labeler import DocumentPredictions, SimilarityLabeler
from src.nlp.similarity_index import SimilarityIndex
from src.utils.logging_utils import get_logger


## -----------------------------
## Logger
## -----------------------------
logger = get_logger("labeling_hybrid_labeler")


## -----------------------------
## Data structures
## -----------------------------
@dataclass(frozen=True)
class HybridLabelerConfig:
    """
        Configuration for the hybrid labeler

        Notes:
            - This first version keeps the architecture ready for ML classifiers
            - For now, it delegates predictions to SimilarityLabeler only

        Attributes:
            use_similarity: Whether similarity-based labeling is enabled
            use_classifiers: Whether binary classifiers are enabled (future)
    """

    use_similarity: bool = True
    use_classifiers: bool = False


## -----------------------------
## Hybrid labeler
## -----------------------------
class HybridLabeler:
    """
        Hybrid labeler combining:
            - Similarity-based evidence labeling (always available)
            - Optional per-label binary classifiers (future extension)

        Current behavior:
            - Uses similarity-based labeling only
            - Provides a stable interface for later adding CAD / RF / Boosting / DL
    """

    def __init__(
        self,
        index: SimilarityIndex,
        label_definitions: Dict[str, LabelDefinition],
        config: Optional[HybridLabelerConfig] = None,
    ) -> None:
        """
            Initialize the hybrid labeler

            Args:
                index: SimilarityIndex built from labeled documents
                label_definitions: LabelDefinition mapping label -> definition/thresholds
                config: Optional HybridLabelerConfig
        """

        self.index = index
        self.label_definitions = label_definitions
        self.config = config or HybridLabelerConfig()

        ## Similarity-based labeler is always available
        self.similarity_labeler = SimilarityLabeler(
            index=self.index,
            label_definitions=self.label_definitions,
        )

        ## Placeholder for future classifiers
        self.classifiers: Dict[str, object] = {}

        ## Validate definitions
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """
            Validate constructor inputs to fail fast in pipeline.
        """

        ## Ensure we have definitions for all expected labels
        missing = [lbl for lbl in LABELS if lbl not in self.label_definitions]
        if missing:
            raise LabelingError(f"Missing label definitions for: {missing}")

    def predict(
        self,
        filename: str,
        segments,
        segment_vectors,
    ) -> DocumentPredictions:
        """
            Predict document labels using hybrid strategy

            Args:
                filename: Document filename
                segments: List of DocumentSegment
                segment_vectors: List of embedding vectors for segments

            Returns:
                DocumentPredictions
        """

        ## Similarity-only path (current stable behavior)
        if self.config.use_similarity and not self.config.use_classifiers:
            return self.similarity_labeler.predict(
                filename=filename,
                segments=segments,
                segment_vectors=segment_vectors,
            )

        ## Hybrid path (reserved for future)
        if self.config.use_similarity and self.config.use_classifiers:
            ## Step 1: similarity prediction (baseline)
            sim_pred = self.similarity_labeler.predict(
                filename=filename,
                segments=segments,
                segment_vectors=segment_vectors,
            )

            ## Step 2: merge classifier outputs (not implemented yet)
            ## For now, we return similarity results unchanged
            logger.warning(
                "Hybrid mode requested (similarity + classifiers), but classifiers are not implemented yet"
            )
            return sim_pred

        ## Invalid configuration
        raise LabelingError("HybridLabeler configuration is invalid (no strategy enabled)")
