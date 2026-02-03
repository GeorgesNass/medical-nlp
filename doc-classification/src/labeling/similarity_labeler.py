'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Similarity-based multi-labeler using segment embeddings and a global similarity index."
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.domain.schema import DocumentSegment
from src.labeling.label_definitions import LabelDefinition, build_label_definitions
from src.nlp.similarity_index import SimilarityIndex, SimilarityResult
from src.utils.logging_utils import get_logger


## -----------------------------
## Logger
## -----------------------------
logger = get_logger("labeling_similarity_labeler")


## -----------------------------
## Results structures
## -----------------------------
@dataclass(frozen=True)
class LabelPrediction:
    """
        Binary prediction result for a single label

        Attributes:
            label: Label name
            predicted: Boolean decision
            score: Best similarity score observed for the label
            evidence: Top evidence match (best SimilarityResult) if available
    """

    label: str
    predicted: bool
    score: float
    evidence: Optional[SimilarityResult]


@dataclass(frozen=True)
class DocumentPredictions:
    """
        Multi-label predictions for a document

        Attributes:
            filename: Document filename
            predictions: Mapping label -> LabelPrediction
    """

    filename: str
    predictions: Dict[str, LabelPrediction]


## -----------------------------
## Similarity labeler
## -----------------------------
class SimilarityLabeler:
    """
        Similarity-based multi-labeler

        Strategy:
            - For each document segment, retrieve nearest labeled segments
            - Aggregate best score per label across all segments
            - Apply per-label thresholds to return TRUE/FALSE

        Notes:
            - This provides evidence (top matching labeled segment) per label
            - Can be combined later with binary classifiers (hybrid)
    """

    def __init__(
        self,
        index: SimilarityIndex,
        label_definitions: Optional[Dict[str, LabelDefinition]] = None,
    ) -> None:
        """
            Initialize the labeler

            Args:
                index: SimilarityIndex built from labeled documents
                label_definitions: Optional label definitions (defaults from CONFIG)
        """

        self.index = index
        self.label_definitions = label_definitions or build_label_definitions()

    def predict(
        self,
        filename: str,
        segments: List[DocumentSegment],
        segment_vectors: List[List[float]],
    ) -> DocumentPredictions:
        """
            Predict multi-label output for a document

            Args:
                filename: Document filename
                segments: Document segments (same order as vectors)
                segment_vectors: Embeddings for each segment

            Returns:
                DocumentPredictions
        """

        ## Validate alignment
        if len(segments) != len(segment_vectors):
            raise ValueError("segments and segment_vectors must have the same length")

        ## Query index for each segment (top-k per segment handled by index)
        results_per_segment = self.index.search(segment_vectors)

        ## Aggregate best score per label
        best_score_by_label: Dict[str, float] = {k: 0.0 for k in self.label_definitions}
        best_evidence_by_label: Dict[str, Optional[SimilarityResult]] = {
            k: None for k in self.label_definitions
        }

        ## For each segment, check retrieved matches and update per-label max
        for seg, matches in zip(segments, results_per_segment):
            _ = seg  ## segment kept for future debugging/evidence expansion

            for match in matches:
                lbl = match.label

                ## Skip unknown labels (defensive)
                if lbl not in best_score_by_label:
                    continue

                ## Keep max similarity per label (best evidence)
                if float(match.score) > float(best_score_by_label[lbl]):
                    best_score_by_label[lbl] = float(match.score)
                    best_evidence_by_label[lbl] = match

        ## Apply thresholds to generate predictions
        predictions: Dict[str, LabelPrediction] = {}

        for lbl, definition in self.label_definitions.items():
            score = float(best_score_by_label.get(lbl, 0.0))
            evidence = best_evidence_by_label.get(lbl)

            ## Binary decision based on per-label threshold
            predicted = bool(score >= float(definition.threshold))

            predictions[lbl] = LabelPrediction(
                label=lbl,
                predicted=predicted,
                score=score,
                evidence=evidence,
            )

        logger.info(f"Predicted labels for file: {filename}")
        return DocumentPredictions(filename=filename, predictions=predictions)
