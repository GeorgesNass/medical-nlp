'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main pipeline orchestration: index building, similarity-based labeling, EDA and CSV export."
'''

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from src.core.config import CONFIG, LABELS
from src.core.errors import (  ## errors wired
    DataError,
    LabelingError,
    ManifestError,
    PipelineError,
)
from src.labeling.similarity_labeler import DocumentPredictions, SimilarityLabeler
from src.labeling.label_definitions import build_label_definitions
from src.nlp.embeddings import EmbeddingBackend
from src.nlp.segmenter import segment_text
from src.nlp.similarity_index import SimilarityIndex
from src.utils.data_utils import export_dicts_to_csv
from src.utils.io_utils import load_text_from_path
from src.utils.logging_utils import get_logger


## ============================================================
## LOGGER
## ============================================================
logger = get_logger("pipeline")


## ============================================================
## SIMPLE MANIFEST STRUCTURE
## ============================================================
@dataclass(frozen=True)
class LabeledManifest:
    """
        Simple container for labeled documents

        Attributes:
            mapping: Dictionary mapping filename -> list of labels
    """

    mapping: Dict[str, List[str]]


## ============================================================
## PIPELINE STEP 1: BUILD SIMILARITY INDEX
## ============================================================
def build_similarity_index_from_labeled(
    labeled_folder: str | Path,
    manifest_path: str | Path,
) -> SimilarityIndex:
    """
        Build a global similarity index from labeled documents

        High-level workflow:
            1) Load labeled documents from disk
            2) Segment documents into overlapping blocks
            3) Encode each block into an embedding
            4) Store vectors + metadata in a similarity index

        Design choice:
            - Each segment is duplicated per label to keep indexing logic simple
            - Label attribution is handled at indexing time, not query time

        Args:
            labeled_folder: Folder containing labeled documents
            manifest_path: JSON file mapping filename -> list of labels

        Returns:
            A populated SimilarityIndex
    """

    ## Resolve and validate labeled folder
    labeled_dir = Path(labeled_folder).expanduser().resolve()
    if not labeled_dir.exists() or not labeled_dir.is_dir():
        raise DataError(f"Invalid labeled folder: {labeled_dir}")

    ## Load manifest (filename -> labels)
    manifest = _load_labeled_manifest(manifest_path)

    ## Initialize embedding backend (GPU/CPU handled internally)
    backend = EmbeddingBackend()

    ## Initialize similarity index
    ## Vectors are assumed normalized if embeddings backend normalizes them
    index = SimilarityIndex(normalize_vectors=bool(CONFIG.embeddings.normalize))

    ## --------------------------------------------------------
    ## Iterate over labeled documents
    ## --------------------------------------------------------
    for filename, labels in manifest.mapping.items():
        file_path = labeled_dir / filename

        ## Defensive checks
        if not file_path.exists():
            logger.warning(f"File listed in manifest not found, skipping: {filename}")
            continue

        ## Keep only known labels
        clean_labels = [lbl for lbl in labels if lbl in LABELS]
        if not clean_labels:
            logger.warning(f"No valid labels for file, skipping: {filename}")
            continue

        ## Load raw text from file
        try:
            text = load_text_from_path(file_path)
        except Exception as exc:
            raise PipelineError(f"Failed to load labeled file: {file_path}") from exc

        if not text.strip():
            logger.warning(f"Empty text extracted, skipping: {filename}")
            continue

        ## Segment document into overlapping blocks
        try:
            segments = segment_text(text)
        except Exception as exc:
            raise PipelineError(f"Segmentation failed for labeled file: {filename}") from exc

        if not segments:
            logger.warning(f"No segments produced, skipping: {filename}")
            continue

        ## Encode each segment into a vector
        try:
            segment_vectors = backend.encode([s.text for s in segments])
        except Exception as exc:
            raise PipelineError(f"Embeddings failed for labeled file: {filename}") from exc

        if not segment_vectors:
            logger.warning(f"No embeddings produced, skipping: {filename}")
            continue

        ## ----------------------------------------------------
        ## Add to index
        ## ----------------------------------------------------
        ## Strategy:
        ##   - One segment can belong to multiple labels
        ##   - We duplicate (segment, vector) per label
        ##   - This simplifies similarity aggregation later
        vectors_to_add: List[List[float]] = []
        segments_to_add: List = []
        labels_to_add: List[str] = []
        sources_to_add: List[str] = []

        for lbl in clean_labels:
            for vec, seg in zip(segment_vectors, segments):
                vectors_to_add.append(vec)
                segments_to_add.append(seg)
                labels_to_add.append(lbl)
                sources_to_add.append(filename)

        try:
            index.add(
                vectors=vectors_to_add,
                segments=segments_to_add,
                labels=labels_to_add,
                source_files=sources_to_add,
            )
        except Exception as exc:
            raise PipelineError(f"Failed to add vectors to index for file: {filename}") from exc

    logger.info(f"Similarity index built with {index.size()} indexed segments")
    return index


## ============================================================
## PIPELINE STEP 2: PREDICT LABELS FOR UNLABELED DOCUMENTS
## ============================================================
def predict_labels_for_unlabeled(
    unlabeled_folder: str | Path,
    index: SimilarityIndex,
) -> List[DocumentPredictions]:
    """
        Predict multi-label outputs for documents without labels

        High-level workflow:
            1) Load each document
            2) Segment into blocks
            3) Encode blocks
            4) Query similarity index
            5) Aggregate max similarity score per label
            6) Apply per-label thresholds

        Args:
            unlabeled_folder: Folder containing unlabeled documents
            index: Pre-built similarity index

        Returns:
            List of DocumentPredictions
    """

    ## Resolve and validate folder
    unlabeled_dir = Path(unlabeled_folder).expanduser().resolve()
    if not unlabeled_dir.exists() or not unlabeled_dir.is_dir():
        raise DataError(f"Invalid unlabeled folder: {unlabeled_dir}")

    ## Initialize embedding backend
    backend = EmbeddingBackend()

    ## Build label definitions (thresholds already resolved in CONFIG)
    try:
        label_definitions = build_label_definitions()
    except Exception as exc:
        raise PipelineError("Failed to build label definitions") from exc

    ## Initialize similarity-based labeler
    try:
        labeler = SimilarityLabeler(
            index=index,
            label_definitions=label_definitions,
        )
    except Exception as exc:
        raise PipelineError("Failed to initialize similarity labeler") from exc

    ## Collect predictions
    predictions: List[DocumentPredictions] = []

    ## Iterate through files
    file_paths = _list_supported_files(unlabeled_dir)
    if not file_paths:
        raise DataError(f"No supported documents found in: {unlabeled_dir}")

    for file_path in file_paths:
        ## Load raw text
        try:
            text = load_text_from_path(file_path)
        except Exception as exc:
            raise PipelineError(f"Failed to load unlabeled file: {file_path}") from exc

        if not text.strip():
            logger.warning(f"Empty text extracted, skipping: {file_path.name}")
            continue

        ## Segment text
        try:
            segments = segment_text(text)
        except Exception as exc:
            raise PipelineError(f"Segmentation failed for unlabeled file: {file_path.name}") from exc

        if not segments:
            logger.warning(f"No segments produced, skipping: {file_path.name}")
            continue

        ## Encode segments
        try:
            segment_vectors = backend.encode([s.text for s in segments])
        except Exception as exc:
            raise PipelineError(f"Embeddings failed for unlabeled file: {file_path.name}") from exc

        if not segment_vectors:
            logger.warning(f"No embeddings produced, skipping: {file_path.name}")
            continue

        ## Predict labels using similarity
        try:
            doc_pred = labeler.predict(
                filename=file_path.name,
                segments=segments,
                segment_vectors=segment_vectors,
            )
        except Exception as exc:
            raise LabelingError(f"Label prediction failed for file: {file_path.name}") from exc

        predictions.append(doc_pred)

    logger.info(f"Predicted labels for {len(predictions)} documents")
    return predictions


## ============================================================
## PIPELINE STEP 3: EXPORT RESULTS
## ============================================================
def export_predictions(
    predictions: List[DocumentPredictions],
    output_csv_name: str = "predictions.csv",
    include_scores: bool = True,
    include_evidence: bool = True,
) -> Path:
    """
        Export predictions to a CSV file in artifacts/exports/

        Args:
            predictions: List of DocumentPredictions
            output_csv_name: Output CSV filename
            include_scores: Include per-label similarity scores
            include_evidence: Include evidence (file, score, text)

        Returns:
            Path to exported CSV file
    """

    ## Configure CSV export
    export_cfg = CsvExportConfig(
        include_scores=include_scores,
        include_evidence=include_evidence,
        evidence_max_chars=250,
    )

    ## Resolve output path
    output_path = CONFIG.paths.exports_dir / output_csv_name

    ## Perform export
    try:
        return export_predictions_to_csv(
            predictions=predictions,
            output_path=output_path,
            config=export_cfg,
        )
    except Exception as exc:
        raise PipelineError(f"CSV export failed: {output_path}") from exc


## ============================================================
## INTERNAL HELPERS
## ============================================================
def _load_labeled_manifest(manifest_path: str | Path) -> LabeledManifest:
    """
        Load labeled manifest JSON

        Expected format:
            {
              "doc1.pdf": ["crh", "analyse_labo"],
              "doc2.txt": ["ordonnance_medicaments"]
            }

        Args:
            manifest_path: Path to JSON manifest

        Returns:
            LabeledManifest
    """

    path = Path(manifest_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise ManifestError(f"Invalid manifest path: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise ManifestError(f"Failed to read manifest JSON: {path}") from exc

    if not isinstance(data, dict):
        raise ManifestError("Manifest JSON must be a dict: filename -> list of labels")

    mapping: Dict[str, List[str]] = {}

    for filename, labels in data.items():
        if not isinstance(filename, str) or not isinstance(labels, list):
            continue

        clean_labels = [
            str(lbl).strip()
            for lbl in labels
            if str(lbl).strip() and str(lbl).strip() in LABELS
        ]

        mapping[filename.strip()] = clean_labels

    return LabeledManifest(mapping=mapping)


def _list_supported_files(folder: Path) -> List[Path]:
    """
        List supported document files in a folder (non-recursive)

        Args:
            folder: Input folder

        Returns:
            List of file paths
    """

    supported = {".txt", ".pdf", ".docx"}
    return [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in supported
    ]
