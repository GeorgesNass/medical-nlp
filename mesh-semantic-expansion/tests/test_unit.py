'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unit tests for core utilities, NLP components, and quality evaluation logic."
'''

from pathlib import Path

import numpy as np
import pytest

from src.nlp.judge_quality import JudgeResult, judge_baseline
from src.nlp.ner_mesh import build_label_dictionary, detect_entities
from src.nlp.expand_terms import _find_abbreviation_patterns
from src.nlp.embeddings import embed_texts


## ============================================================
## NLP: JUDGE QUALITY
## ============================================================

def test_judge_baseline_exact_match() -> None:
    """Accept a candidate when it exactly matches the suggested label."""

    result = judge_baseline(
        candidate_term="Hypertension",
        suggested_label="Hypertension",
    )

    assert isinstance(result, JudgeResult)
    assert result.verdict == "accepted"
    assert result.score == 1.0


def test_judge_baseline_reject_low_overlap() -> None:
    """Reject a candidate when lexical overlap is very low."""

    result = judge_baseline(
        candidate_term="Hypertension",
        suggested_label="Diabetes Mellitus",
    )

    assert result.verdict == "rejected"
    assert result.score < 0.3


## ============================================================
## NLP: ABBREVIATION EXTRACTION
## ============================================================

def test_find_abbreviation_patterns() -> None:
    """Extract abbreviation patterns from medical text."""

    text = "Le patient présente une hypertension artérielle (HTA) sévère."
    patterns = _find_abbreviation_patterns(text)

    assert len(patterns) == 1

    long_form, abbr, start, end = patterns[0]
    assert long_form.lower().startswith("hypertension")
    assert abbr == "HTA"
    assert start >= 0
    assert end > start


## ============================================================
## NLP: ENTITY DETECTION
## ============================================================

def test_detect_entities_dictionary(tmp_path: Path) -> None:
    """Detect entities using a dictionary-based approach."""

    mesh_jsonl = tmp_path / "mesh_parsed.jsonl"
    mesh_jsonl.write_text(
        '{"ui": "D000002", "preferred_terms": ["Hypertension"], "synonyms": ["HTN"]}\n',
        encoding="utf-8",
    )

    label_dict = build_label_dictionary(mesh_jsonl_path=mesh_jsonl)

    text = "Patient with hypertension."
    entities = detect_entities(text=text, label_dict=label_dict, use_fts_fallback=False)

    assert len(entities) == 1
    assert entities[0].ui == "D000002"
    assert entities[0].method == "dict"


## ============================================================
## NLP: EMBEDDINGS
## ============================================================

def test_embed_texts_sentence_transformers_shape() -> None:
    """Return consistent embedding shapes using sentence-transformers backend."""

    texts = ["hypertension", "diabetes mellitus"]
    vectors = embed_texts(texts, backend="sentence_transformers")

    assert isinstance(vectors, np.ndarray)
    assert vectors.ndim == 2
    assert vectors.shape[0] == len(texts)
    assert vectors.shape[1] > 100  # model-dependent, but non-trivial


def test_embed_texts_fasttext_empty_safe(monkeypatch) -> None:
    """Handle empty text input safely for FastText backend."""

    ## Monkeypatch FastText loader to avoid loading a real model
    from src.nlp import embeddings as emb_mod

    class DummyFT:
        def get_dimension(self):
            return 10

        def get_word_vector(self, _):
            return np.ones(10, dtype="float32")

    monkeypatch.setattr(emb_mod, "_load_fasttext_model", lambda _: DummyFT())

    vectors = emb_mod._embed_fasttext(["", "test"], model_path=Path("dummy.bin"))

    assert vectors.shape == (2, 10)
    assert np.allclose(vectors[0], 0.0)  # empty text → zero vector
