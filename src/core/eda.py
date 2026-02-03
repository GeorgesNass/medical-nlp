'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Exploratory Data Analysis (EDA) utilities for labeled/unlabeled medical documents and segments."
'''

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.core.config import CONFIG, LABELS, LABEL_KEYWORD_HINTS
from src.nlp.segmenter import segment_text
from src.utils.io_utils import load_text_from_path, normalize_document_text
from src.utils.logging_utils import get_logger


## -----------------------------
## Logger
## -----------------------------
logger = get_logger("core_eda")


## -----------------------------
## EDA results
## -----------------------------
@dataclass(frozen=True)
class EdaSummary:
    """
        EDA summary outputs

        Attributes:
            n_files: Number of processed files
            total_chars: Total characters across documents
            avg_chars: Average characters per document
            total_segments: Total number of segments produced
            avg_segments: Average segments per document
            label_counts: Count of documents per label (if labels provided)
            multi_label_counts: Distribution of number of labels per document
            keyword_hits: Weak keyword hit counts per label (diagnostic)
    """

    n_files: int
    total_chars: int
    avg_chars: float
    total_segments: int
    avg_segments: float
    label_counts: Dict[str, int]
    multi_label_counts: Dict[str, int]
    keyword_hits: Dict[str, int]


## -----------------------------
## Public API
## -----------------------------
def run_eda_on_folder(
    folder_path: str | Path,
    labeled_manifest: Optional[Dict[str, List[str]]] = None,
    output_name: str = "eda_summary.json",
) -> Path:
    """
        Run EDA on all files in a folder and export a JSON summary

        Notes:
            - If labeled_manifest is provided, it must map: filename -> list of labels
            - If not provided, only generic stats are computed

        Args:
            folder_path: Folder containing documents
            labeled_manifest: Optional mapping filename -> list of labels
            output_name: JSON output name

        Returns:
            Path to exported JSON summary
    """

    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid folder path: {folder}")

    file_paths = _list_supported_files(folder)
    if not file_paths:
        raise ValueError(f"No supported files found in: {folder}")

    summary = _compute_eda_summary(file_paths, labeled_manifest=labeled_manifest)

    out_path = CONFIG.paths.reports_dir / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_summary_to_json(summary), f, ensure_ascii=False, indent=2)

    logger.info(f"EDA summary exported to: {out_path}")
    return out_path


## -----------------------------
## Internal helpers
## -----------------------------
def _list_supported_files(folder: Path) -> List[Path]:
    """
        List supported document files in a folder (non-recursive)

        Args:
            folder: Input folder

        Returns:
            List of file paths
    """

    supported = {".txt"}
    paths: List[Path] = []

    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in supported:
            paths.append(p)

    return paths


def _compute_eda_summary(
    file_paths: List[Path],
    labeled_manifest: Optional[Dict[str, List[str]]] = None,
) -> EdaSummary:
    """
        Compute EDA summary for a list of file paths

        Args:
            file_paths: List of file paths
            labeled_manifest: Optional mapping filename -> list of labels

        Returns:
            EdaSummary
    """

    total_chars = 0
    total_segments = 0

    label_counts: Dict[str, int] = {lbl: 0 for lbl in LABELS}
    multi_label_counter: Counter = Counter()
    keyword_hits: Dict[str, int] = {lbl: 0 for lbl in LABELS}

    ## Flatten keyword hints once for faster scanning
    keywords_by_label = _build_keywords_by_label()

    for path in file_paths:
        ## Load raw text
        text = load_text_from_path(path)
        text = normalize_document_text(text)
        text_len = len(text or "")

        total_chars += text_len

        ## Segment text using current CONFIG
        segments = segment_text(text)
        total_segments += len(segments)

        ## If labels provided, update label distributions
        if labeled_manifest is not None:
            labels = labeled_manifest.get(path.name, [])
            labels = [lbl for lbl in labels if lbl in LABELS]

            ## Count per label
            for lbl in labels:
                label_counts[lbl] += 1

            ## Count number of labels per doc
            multi_label_counter[str(len(labels))] += 1

        ## Weak keyword diagnostics (label hints)
        lowered = (text or "").lower()
        for lbl in LABELS:
            hits = 0
            for kw in keywords_by_label.get(lbl, []):
                if kw and kw in lowered:
                    hits += 1
            if hits > 0:
                keyword_hits[lbl] += 1

    n_files = len(file_paths)
    avg_chars = float(total_chars) / float(max(1, n_files))
    avg_segments = float(total_segments) / float(max(1, n_files))

    return EdaSummary(
        n_files=n_files,
        total_chars=total_chars,
        avg_chars=avg_chars,
        total_segments=total_segments,
        avg_segments=avg_segments,
        label_counts=label_counts,
        multi_label_counts=dict(multi_label_counter),
        keyword_hits=keyword_hits,
    )


def _build_keywords_by_label() -> Dict[str, List[str]]:
    """
        Build keyword list per label

        Returns:
            Dict label -> list of keywords
    """

    keywords_by_label: Dict[str, List[str]] = defaultdict(list)

    for lbl in LABELS:
        ## Merge keyword hints from config-level mapping
        for kw in LABEL_KEYWORD_HINTS.get(lbl, []):
            if kw and kw.strip():
                keywords_by_label[lbl].append(kw.strip().lower())

    return dict(keywords_by_label)


def _summary_to_json(summary: EdaSummary) -> Dict[str, object]:
    """
        Convert EdaSummary to a JSON serializable dict

        Args:
            summary: EdaSummary instance

        Returns:
            JSON dict
    """

    return {
        "n_files": summary.n_files,
        "total_chars": summary.total_chars,
        "avg_chars": summary.avg_chars,
        "total_segments": summary.total_segments,
        "avg_segments": summary.avg_segments,
        "label_counts": summary.label_counts,
        "multi_label_counts": summary.multi_label_counts,
        "keyword_hits": summary.keyword_hits,
    }
