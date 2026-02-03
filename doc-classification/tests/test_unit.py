'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unit tests for core utilities (config, loaders, segmenter, similarity index)."
'''

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.core.errors import DataError, ManifestError, PipelineError
from src.domain.schema import DocumentSegment
from src.nlp.segmenter import SegmenterConfig, normalize_text, segment_text, tokenize_words
from src.nlp.similarity_index import SimilarityIndex
from src.utils.io_utils import load_text_from_bytes, load_text_from_path


## ============================================================
## TESTS: LOADERS
## ============================================================
def test_load_text_from_bytes_txt_success() -> None:
    """
        Test TXT decoding from bytes works and returns a string.
    """

    ## Arrange
    content = "Bonjour, ceci est un test.\nLigne 2.".encode("utf-8")

    ## Act
    text = load_text_from_bytes(content=content, file_extension=".txt")

    ## Assert
    assert isinstance(text, str)
    assert "Bonjour" in text


def test_load_text_from_bytes_txt_invalid_ext_raises_data_error() -> None:
    """
        Test byte loader refuses non-txt extensions.
    """

    ## Arrange
    content = b"anything"

    ## Act / Assert
    with pytest.raises(DataError):
        _ = load_text_from_bytes(content=content, file_extension=".pdf")


def test_load_text_from_path_missing_file_raises_data_error(tmp_path: Path) -> None:
    """
        Test missing file path raises DataError.
    """

    ## Arrange
    missing_path = tmp_path / "missing.txt"

    ## Act / Assert
    with pytest.raises(DataError):
        _ = load_text_from_path(missing_path)


def test_load_text_from_path_directory_raises_data_error(tmp_path: Path) -> None:
    """
        Test passing a directory to loader raises DataError.
    """

    ## Arrange
    folder = tmp_path / "folder"
    folder.mkdir(parents=True, exist_ok=True)

    ## Act / Assert
    with pytest.raises(DataError):
        _ = load_text_from_path(folder)


def test_load_text_from_path_txt_success(tmp_path: Path) -> None:
    """
        Test TXT file loading returns content.
    """

    ## Arrange
    p = tmp_path / "doc.txt"
    p.write_text("Patient: Jean Dupont\nAge: 42\n", encoding="utf-8")

    ## Act
    text = load_text_from_path(p)

    ## Assert
    assert "Jean" in text
    assert "Age" in text


## ============================================================
## TESTS: SEGMENTER
## ============================================================
def test_normalize_text_collapses_whitespace() -> None:
    """
        Test normalization collapses multiple whitespace.
    """

    ## Arrange
    raw = "A   B\t\tC\n\nD"

    ## Act
    norm = normalize_text(raw)

    ## Assert
    assert norm == "A B C D"


def test_tokenize_words_basic() -> None:
    """
        Test tokenization extracts word-like tokens.
    """

    ## Arrange
    text = "CRH: Patient Jean Dupont, 42 ans."

    ## Act
    tokens = tokenize_words(text)

    ## Assert
    assert "CRH" in tokens or "crh" in [t.lower() for t in tokens]
    assert "Jean" in tokens
    assert "Dupont" in tokens


def test_segment_text_returns_overlapping_segments() -> None:
    """
        Test segmentation returns multiple overlapping segments.
    """

    ## Arrange
    ## Enough tokens to generate multiple segments with overlap
    text = " ".join([f"mot{i}" for i in range(60)])

    cfg = SegmenterConfig(
        window_size_tokens=20,
        window_overlap_tokens=10,
        min_chars_per_segment=10,
    )

    ## Act
    segments = segment_text(text, config=cfg, segment_id_prefix="t")

    ## Assert
    assert isinstance(segments, list)
    assert len(segments) >= 2

    ## Check segment ids and char offsets are coherent
    assert segments[0].segment_id.startswith("t_")
    assert segments[0].start_char >= 0
    assert segments[0].end_char > segments[0].start_char


def test_segment_text_empty_returns_empty() -> None:
    """
        Test segmentation on empty text returns empty list.
    """

    ## Act
    segments = segment_text("")

    ## Assert
    assert segments == []


## ============================================================
## TESTS: SIMILARITY INDEX
## ============================================================
def _make_segments(n: int) -> List[DocumentSegment]:
    """
        Helper to create dummy segments.

        Args:
            n: number of segments

        Returns:
            List of DocumentSegment
    """

    segments: List[DocumentSegment] = []
    for i in range(n):
        segments.append(
            DocumentSegment(
                segment_id=f"s{i}",
                text=f"segment {i}",
                start_char=0,
                end_char=10,
                meta={},
            )
        )
    return segments


def test_similarity_index_add_misaligned_lengths_raises_pipeline_error() -> None:
    """
        Test add() rejects misaligned metadata lengths.
    """

    ## Arrange
    index = SimilarityIndex(normalize_vectors=True)
    vectors = [[0.1, 0.2], [0.2, 0.3]]
    segments = _make_segments(1)
    labels = ["crh", "cro"]
    sources = ["a.txt", "b.txt"]

    ## Act / Assert
    with pytest.raises(PipelineError):
        index.add(vectors=vectors, segments=segments, labels=labels, source_files=sources)


def test_similarity_index_add_and_search_returns_results() -> None:
    """
        Test index can be built and queried with cosine similarity.
    """

    ## Arrange
    index = SimilarityIndex(normalize_vectors=True)

    ## Create 3 vectors in 2D
    vectors = [
        [1.0, 0.0],  ## label crh
        [0.0, 1.0],  ## label cro
        [1.0, 0.0],  ## label crh (duplicate direction)
    ]
    segments = _make_segments(3)
    labels = ["crh", "cro", "crh"]
    sources = ["f1.txt", "f2.txt", "f3.txt"]

    ## Act: add
    index.add(vectors=vectors, segments=segments, labels=labels, source_files=sources)

    ## Act: search (query close to [1, 0])
    results = index.search(query_vectors=[[1.0, 0.0]], top_k=2)

    ## Assert
    assert len(results) == 1
    assert len(results[0]) == 2

    ## Best match should be a "crh" vector with high score
    assert results[0][0].label in {"crh"}
    assert results[0][0].score == pytest.approx(1.0, abs=1e-6)


def test_similarity_index_empty_search_returns_empty_lists() -> None:
    """
        Test searching an empty index returns empty result lists.
    """

    ## Arrange
    index = SimilarityIndex(normalize_vectors=True)

    ## Act
    results = index.search(query_vectors=[[0.1, 0.2], [0.2, 0.3]], top_k=3)

    ## Assert
    assert results == [[], []]


def test_similarity_index_size() -> None:
    """
        Test size() reflects number of indexed vectors.
    """

    ## Arrange
    index = SimilarityIndex(normalize_vectors=True)
    assert index.size() == 0

    ## Act
    vectors = [[1.0, 0.0]]
    segments = _make_segments(1)
    labels = ["crh"]
    sources = ["f1.txt"]
    index.add(vectors=vectors, segments=segments, labels=labels, source_files=sources)

    ## Assert
    assert index.size() == 1
