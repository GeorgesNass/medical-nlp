'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unit tests for core utilities, NLP components, quality evaluation logic and MeSH indexing and querying using a minimal synthetic dataset."
'''

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.mesh.index_mesh import build_sqlite_fts_index
from src.mesh.query_mesh import browse_tree, lookup_ui, search_mesh
from src.nlp.embeddings import emb_mod, embed_texts
from src.nlp.expand_terms import _find_abbreviation_patterns
from src.nlp.judge_quality import JudgeResult, judge_baseline
from src.nlp.ner_mesh import build_label_dictionary, detect_entities

## ============================================================
## NLP: JUDGE QUALITY
## ============================================================
def test_judge_baseline_exact_match() -> None:
    """
        Accept a candidate when it exactly matches the suggested label

        High-level workflow:
            1) Compare identical medical labels
            2) Validate acceptance verdict and perfect score

        Returns:
            None
    """

    ## Run baseline judge on exact match
    result = judge_baseline(
        candidate_term="Hypertension",
        suggested_label="Hypertension",
    )

    ## Validate output type and expected decision
    assert isinstance(result, JudgeResult)
    assert result.verdict == "accepted"
    assert result.score == 1.0

def test_judge_baseline_reject_low_overlap() -> None:
    """
        Reject a candidate when lexical overlap is very low

        High-level workflow:
            1) Compare semantically unrelated labels
            2) Validate rejection verdict and low score

        Returns:
            None
    """

    ## Run baseline judge on unrelated terms
    result = judge_baseline(
        candidate_term="Hypertension",
        suggested_label="Diabetes Mellitus",
    )

    ## Validate rejection behavior
    assert result.verdict == "rejected"
    assert result.score < 0.3

def test_judge_baseline_empty_candidate() -> None:
    """
        Reject or mark uncertain an empty candidate term

        High-level workflow:
            1) Evaluate baseline judge with empty candidate
            2) Validate robust fallback verdict

        Returns:
            None
    """

    ## Run baseline judge with empty candidate term
    result = judge_baseline(
        candidate_term="",
        suggested_label="Hypertension",
    )

    ## Validate safe output object and conservative verdict
    assert isinstance(result, JudgeResult)
    assert result.verdict in {"rejected", "uncertain"}

## ============================================================
## NLP: ABBREVIATION EXTRACTION
## ============================================================
def test_find_abbreviation_patterns() -> None:
    """
        Extract abbreviation patterns from medical text

        High-level workflow:
            1) Parse a sentence containing long form + abbreviation
            2) Validate extracted pattern structure

        Returns:
            None
    """

    ## Prepare text containing one abbreviation pattern
    text = "Le patient présente une hypertension artérielle (HTA) sévère."

    ## Extract abbreviation patterns
    patterns = _find_abbreviation_patterns(text)

    ## Validate one pattern has been extracted
    assert len(patterns) == 1

    long_form, abbr, start, end = patterns[0]

    ## Validate extracted components
    assert long_form.lower().startswith("hypertension")
    assert abbr == "HTA"
    assert start >= 0
    assert end > start

def test_find_abbreviation_patterns_no_match() -> None:
    """
        Return no abbreviation pattern when none exists

        High-level workflow:
            1) Parse a sentence without abbreviation pattern
            2) Validate empty result

        Returns:
            None
    """

    ## Prepare text without abbreviation syntax
    text = "Le patient présente une hypertension sévère."

    ## Extract abbreviation patterns
    patterns = _find_abbreviation_patterns(text)

    ## Validate no pattern was found
    assert patterns == []

## ============================================================
## NLP: ENTITY DETECTION
## ============================================================
def test_detect_entities_dictionary(tmp_path: Path) -> None:
    """
        Detect entities using a dictionary-based approach

        High-level workflow:
            1) Build a minimal MeSH JSONL resource
            2) Construct label dictionary
            3) Detect entities in plain text
            4) Validate extracted entity metadata

        Args:
            tmp_path: Pytest temporary directory

        Returns:
            None
    """

    ## Create a minimal MeSH JSONL source file
    mesh_jsonl = tmp_path / "mesh_parsed.jsonl"
    mesh_jsonl.write_text(
        '{"ui": "D000002", "preferred_terms": ["Hypertension"], "synonyms": ["HTN"]}\n',
        encoding="utf-8",
    )

    ## Build dictionary-based label resource
    label_dict = build_label_dictionary(mesh_jsonl_path=mesh_jsonl)

    ## Run dictionary NER
    text = "Patient with hypertension."
    entities = detect_entities(
        text=text,
        label_dict=label_dict,
        use_fts_fallback=False,
    )

    ## Validate one entity is found with expected metadata
    assert len(entities) == 1
    assert entities[0].ui == "D000002"
    assert entities[0].method == "dict"

def test_detect_entities_dictionary_no_match(tmp_path: Path) -> None:
    """
        Return empty entity list when no dictionary match is found

        High-level workflow:
            1) Build a minimal dictionary resource
            2) Run detection on unrelated text
            3) Validate empty result list

        Args:
            tmp_path: Pytest temporary directory

        Returns:
            None
    """

    ## Create a minimal MeSH JSONL source file
    mesh_jsonl = tmp_path / "mesh_parsed.jsonl"
    mesh_jsonl.write_text(
        '{"ui": "D000002", "preferred_terms": ["Hypertension"], "synonyms": ["HTN"]}\n',
        encoding="utf-8",
    )

    ## Build dictionary-based label resource
    label_dict = build_label_dictionary(mesh_jsonl_path=mesh_jsonl)

    ## Run NER on a text without any matching concept
    text = "Patient with appendicitis."
    entities = detect_entities(
        text=text,
        label_dict=label_dict,
        use_fts_fallback=False,
    )

    ## Validate no entity is returned
    assert entities == []

## ============================================================
## NLP: EMBEDDINGS
## ============================================================
def test_embed_texts_sentence_transformers_shape() -> None:
    """
        Return consistent embedding shapes using sentence-transformers backend

        High-level workflow:
            1) Encode two short medical texts
            2) Validate output array dimensionality and shape

        Returns:
            None
    """

    ## Prepare two texts for embedding
    texts = ["hypertension", "diabetes mellitus"]

    ## Compute embeddings with sentence-transformers backend
    vectors = embed_texts(texts, backend="sentence_transformers")

    ## Validate embedding matrix structure
    assert isinstance(vectors, np.ndarray)
    assert vectors.ndim == 2
    assert vectors.shape[0] == len(texts)
    assert vectors.shape[1] > 100

def test_embed_texts_fasttext_empty_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
        Handle empty text input safely for FastText backend

        High-level workflow:
            1) Mock FastText model loading
            2) Embed one empty text and one normal text
            3) Validate output shape and zero-vector fallback

        Args:
            monkeypatch: Pytest monkeypatch fixture

        Returns:
            None
    """

    class DummyFT:
        """
            Minimal FastText-like dummy model for tests
        """

        def get_dimension(self) -> int:
            """
                Return dummy vector dimension

                Returns:
                    Embedding dimension
            """

            return 10

        def get_word_vector(self, _) -> np.ndarray:
            """
                Return a constant dummy word vector

                Args:
                    _: Ignored input token

                Returns:
                    Constant vector
            """

            return np.ones(10, dtype="float32")

    ## Patch FastText model loader with dummy implementation
    monkeypatch.setattr(
        emb_mod,
        "_load_fasttext_model",
        lambda _: DummyFT(),
    )

    ## Run embedding with one empty text
    vectors = emb_mod._embed_fasttext(
        ["", "test"],
        model_path=Path("dummy.bin"),
    )

    ## Validate shape and zero-vector handling for empty text
    assert vectors.shape == (2, 10)
    assert np.allclose(vectors[0], 0.0)

def test_embed_texts_sentence_transformers_empty_input() -> None:
    """
        Return empty embeddings safely on empty input

        High-level workflow:
            1) Encode an empty text list
            2) Validate empty embedding matrix

        Returns:
            None
    """

    ## Run embedding on empty input
    vectors = embed_texts([], backend="sentence_transformers")

    ## Validate empty matrix shape
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape[0] == 0

## ============================================================
## TEST FIXTURES
## ============================================================
@pytest.fixture()
def tmp_mesh_jsonl(tmp_path: Path) -> Path:
    """
        Create a minimal mesh_parsed.jsonl file for E2E tests

        High-level workflow:
            1) Build a small synthetic MeSH dataset
            2) Write records to JSONL file
            3) Return generated file path

        Args:
            tmp_path: Pytest temporary directory

        Returns:
            Path to generated JSONL file
    """

    ## Build a tiny synthetic MeSH dataset
    records: List[dict] = [
        {
            "ui": "D000001",
            "preferred_terms": ["Myocardial Infarction"],
            "synonyms": ["Heart Attack", "MI"],
            "tree_numbers": ["C14.280.647"],
            "scope_note": "An infarction of the myocardium.",
            "source": "mesh_xml",
        },
        {
            "ui": "D000002",
            "preferred_terms": ["Hypertension"],
            "synonyms": ["High Blood Pressure", "HTN"],
            "tree_numbers": ["C14.907.489"],
            "scope_note": "Persistently high arterial blood pressure.",
            "source": "mesh_xml",
        },
        {
            "ui": "D000003",
            "preferred_terms": ["Diabetes Mellitus"],
            "synonyms": ["Diabetes", "DM"],
            "tree_numbers": ["C19.246"],
            "scope_note": "A group of metabolic diseases.",
            "source": "mesh_xml",
        },
    ]

    ## Write records to JSONL file
    out = tmp_path / "mesh_parsed.jsonl"
    with open(out, "w", encoding="utf-8") as file_handle:
        for rec in records:
            file_handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return out

@pytest.fixture()
def tmp_sqlite_db(tmp_path: Path) -> Path:
    """
        Provide a SQLite DB output path for tests

        High-level workflow:
            1) Build a temporary SQLite path
            2) Return path without creating content

        Args:
            tmp_path: Pytest temporary directory

        Returns:
            Path to SQLite database file
    """

    return tmp_path / "mesh.db"

@pytest.fixture()
def indexed_db(tmp_mesh_jsonl: Path, tmp_sqlite_db: Path) -> Path:
    """
        Build a SQLite FTS index from the temporary JSONL fixture

        High-level workflow:
            1) Read synthetic JSONL MeSH data
            2) Build SQLite FTS database
            3) Return indexed DB path

        Args:
            tmp_mesh_jsonl: Path to temporary MeSH JSONL input
            tmp_sqlite_db: Path to temporary SQLite database output

        Returns:
            Path to built SQLite database
    """

    ## Build SQLite FTS index from synthetic input
    db_path = build_sqlite_fts_index(
        mesh_jsonl_path=tmp_mesh_jsonl,
        sqlite_db_path=tmp_sqlite_db,
        overwrite=True,
    )

    return db_path

## ============================================================
## E2E TESTS
## ============================================================
def test_e2e_search_mesh(indexed_db: Path) -> None:
    """
        Search MeSH concepts using the FTS index

        High-level workflow:
            1) Query the SQLite FTS index with a known term
            2) Validate at least one result is returned
            3) Validate best result UI matches expected concept

        Args:
            indexed_db: Path to indexed SQLite database

        Returns:
            None
    """

    ## Search for a known MeSH concept
    results = search_mesh(query="Hypertension", limit=5, db_path=indexed_db)

    ## Validate search returned the expected concept
    assert len(results) >= 1
    assert results[0]["ui"] == "D000002"

def test_e2e_search_mesh_no_result(indexed_db: Path) -> None:
    """
        Search with unknown term should return no result

        High-level workflow:
            1) Query the FTS index with an unknown term
            2) Validate no result is returned

        Args:
            indexed_db: Path to indexed SQLite database

        Returns:
            None
    """

    ## Search with an unknown term
    results = search_mesh(query="unknown-term-xyz", limit=5, db_path=indexed_db)

    ## Validate empty result set
    assert results == []

def test_e2e_lookup_ui(indexed_db: Path) -> None:
    """
        Lookup a MeSH concept by its UI

        High-level workflow:
            1) Query the SQLite DB by known UI
            2) Validate returned row and expected fields

        Args:
            indexed_db: Path to indexed SQLite database

        Returns:
            None
    """

    ## Lookup a known UI
    row = lookup_ui("D000001", db_path=indexed_db)

    ## Validate concept identity and preferred term
    assert row["ui"] == "D000001"
    assert "Myocardial Infarction" in row["preferred_terms"]

def test_e2e_lookup_ui_unknown(indexed_db: Path) -> None:
    """
        Lookup with unknown UI should return empty result or None

        High-level workflow:
            1) Query the SQLite DB with a non-existing UI
            2) Validate safe empty-like response

        Args:
            indexed_db: Path to indexed SQLite database

        Returns:
            None
    """

    ## Lookup an unknown UI
    row = lookup_ui("D999999", db_path=indexed_db)

    ## Validate safe response for unknown UI
    assert row in (None, {})

def test_e2e_browse_tree(indexed_db: Path) -> None:
    """
        Browse MeSH concepts by tree prefix

        High-level workflow:
            1) Browse concepts sharing a common tree prefix
            2) Validate multiple expected UIs are returned

        Args:
            indexed_db: Path to indexed SQLite database

        Returns:
            None
    """

    ## Browse by a shared tree prefix
    rows = browse_tree(tree_prefix="C14", limit=10, db_path=indexed_db)

    ## Validate returned UI set
    assert len(rows) >= 2

    uis = {row["ui"] for row in rows}

    assert "D000001" in uis
    assert "D000002" in uis

def test_e2e_browse_tree_no_match(indexed_db: Path) -> None:
    """
        Browse with unmatched tree prefix should return empty list

        High-level workflow:
            1) Browse using a tree prefix absent from the DB
            2) Validate empty result list

        Args:
            indexed_db: Path to indexed SQLite database

        Returns:
            None
    """

    ## Browse with unmatched tree prefix
    rows = browse_tree(tree_prefix="Z99", limit=10, db_path=indexed_db)

    ## Validate no match was found
    assert rows == []