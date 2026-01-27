'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Quality evaluation utilities (LLM-as-a-Judge / Transformers-based heuristics) for labels and generated outputs."
'''

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger("judge_quality")


## ============================================================
## DATA STRUCTURES
## ============================================================

@dataclass
class JudgeResult:
    """
    Evaluation result.

    Attributes:
        verdict (str): 'accepted' | 'rejected' | 'unsure'
        score (float): Normalized score in [0, 1].
        rationale (str): Short explanation for traceability.
        model (str): Backend model identifier.
    """

    verdict: str
    score: float
    rationale: str
    model: str


## ============================================================
## BASELINE HEURISTIC JUDGE (NO EXTERNAL API)
## ============================================================

def judge_baseline(
    candidate_term: str,
    suggested_label: str,
) -> JudgeResult:
    """
    Lightweight baseline judge without external dependencies.

    It checks:
        - empty values
        - basic lexical overlap (token containment)
        - abbreviation patterns

    Args:
        candidate_term (str): Extracted term from document.
        suggested_label (str): Suggested official label.

    Returns:
        JudgeResult: Baseline evaluation result.
    """

    ct = (candidate_term or "").strip()
    sl = (suggested_label or "").strip()

    if not ct or not sl:
        return JudgeResult(
            verdict="unsure",
            score=0.0,
            rationale="Missing candidate_term or suggested_label.",
            model="baseline",
        )

    ct_low = ct.lower()
    sl_low = sl.lower()

    ## Exact match
    if ct_low == sl_low:
        return JudgeResult(
            verdict="accepted",
            score=1.0,
            rationale="Exact match between candidate and label.",
            model="baseline",
        )

    ## Token containment
    ct_tokens = set(re.findall(r"[a-zà-öø-ÿ]+", ct_low))
    sl_tokens = set(re.findall(r"[a-zà-öø-ÿ]+", sl_low))
    overlap = len(ct_tokens.intersection(sl_tokens))
    denom = max(1, len(ct_tokens))

    overlap_ratio = overlap / denom

    ## Abbreviation hint (candidate looks like "HTA", label has words)
    if ct.isupper() and len(ct) <= 8 and len(sl_tokens) >= 2:
        return JudgeResult(
            verdict="unsure",
            score=0.55,
            rationale="Candidate looks like abbreviation; manual validation recommended.",
            model="baseline",
        )

    if overlap_ratio >= 0.75:
        return JudgeResult(
            verdict="accepted",
            score=min(0.95, overlap_ratio),
            rationale="High lexical overlap between candidate and label.",
            model="baseline",
        )

    if overlap_ratio <= 0.2:
        return JudgeResult(
            verdict="rejected",
            score=max(0.05, overlap_ratio),
            rationale="Low lexical overlap between candidate and label.",
            model="baseline",
        )

    return JudgeResult(
        verdict="unsure",
        score=max(0.25, overlap_ratio),
        rationale="Medium lexical overlap; manual validation recommended.",
        model="baseline",
    )


## ============================================================
## TRANSFORMERS JUDGE (LOCAL, NO LLM API)
## ============================================================

def judge_transformers_cosine(
    candidate_term: str,
    suggested_label: str,
    backend: str = "sentence_transformers",
    threshold_accept: float = 0.75,
    threshold_reject: float = 0.45,
) -> JudgeResult:
    """
    Evaluate similarity using embedding cosine similarity (local judge).

    Notes:
        - Uses src/nlp/embeddings.py backends.
        - Returns accepted/rejected/unsure based on thresholds.

    Args:
        candidate_term (str): Extracted term.
        suggested_label (str): Suggested label.
        backend (str): Embedding backend identifier.
        threshold_accept (float): Accept threshold.
        threshold_reject (float): Reject threshold.

    Returns:
        JudgeResult: Evaluation result.
    """

    from src.nlp.embeddings import embed_texts

    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise ImportError("Missing dependency: numpy required for cosine judge.") from e

    texts = [candidate_term, suggested_label]
    vecs = embed_texts(texts, backend=backend)  # type: ignore

    v1 = vecs[0]
    v2 = vecs[1]

    ## Cosine similarity
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) or 1e-9
    sim = float(np.dot(v1, v2) / denom)

    if sim >= threshold_accept:
        verdict = "accepted"
    elif sim <= threshold_reject:
        verdict = "rejected"
    else:
        verdict = "unsure"

    return JudgeResult(
        verdict=verdict,
        score=max(0.0, min(1.0, sim)),
        rationale=f"Cosine similarity={sim:.4f} using backend={backend}.",
        model=f"transformers_cosine:{backend}",
    )


## ============================================================
## LLM-AS-A-JUDGE (STUB)
## ============================================================

def judge_llm_as_a_judge(
    candidate_term: str,
    suggested_label: str,
    context: str = "",
    provider: str = "openai",
) -> JudgeResult:
    """
    LLM-as-a-Judge stub.

    This function is intentionally a stub to avoid inventing:
        - provider credentials
        - model names
        - API contracts

    Implementation will depend on your chosen provider and prompt template.

    Args:
        candidate_term (str): Extracted term.
        suggested_label (str): Suggested label.
        context (str): Optional context snippet.
        provider (str): LLM provider label.

    Returns:
        JudgeResult: Placeholder result.

    Raises:
        NotImplementedError: Always until integrated with a real LLM call.
    """

    raise NotImplementedError(
        "LLM-as-a-Judge is not implemented yet. "
        "Next step: define prompt schema (Pydantic), provider config (.env), "
        "and implement a safe API client."
    )
