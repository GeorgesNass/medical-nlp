'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Text segmentation utilities to split documents into overlapping blocks for similarity-based labeling."
'''

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from src.core.config import CONFIG
from src.core.errors import PipelineError  ## errors wired
from src.domain.schema import DocumentSegment


## -----------------------------
## Regex helpers
## -----------------------------
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class SegmenterConfig:
    """
        Segmenter configuration

        Attributes:
            window_size_tokens: Token window size for sliding segments
            window_overlap_tokens: Overlap between consecutive segments
            min_chars_per_segment: Minimum characters per segment to keep it
    """

    window_size_tokens: int
    window_overlap_tokens: int
    min_chars_per_segment: int


def get_default_segmenter_config() -> SegmenterConfig:
    """
        Get segmenter configuration from global CONFIG

        Returns:
            SegmenterConfig
    """

    return SegmenterConfig(
        window_size_tokens=CONFIG.segmentation.window_size_tokens,
        window_overlap_tokens=CONFIG.segmentation.window_overlap_tokens,
        min_chars_per_segment=CONFIG.segmentation.min_chars_per_segment,
    )


def normalize_text(text: str) -> str:
    """
        Normalize text for segmentation

        Notes:
            - Collapses whitespace
            - Keeps punctuation (useful for medical patterns)

        Args:
            text: Input text

        Returns:
            Normalized text
    """

    return _WHITESPACE_RE.sub(" ", text or "").strip()


def tokenize_words(text: str) -> List[str]:
    """
        Tokenize text into word-like tokens

        Args:
            text: Input text

        Returns:
            List of tokens
    """

    return _WORD_RE.findall(text or "")


def segment_text(
    text: str,
    config: Optional[SegmenterConfig] = None,
    segment_id_prefix: str = "seg",
) -> List[DocumentSegment]:
    """
        Segment a text into overlapping blocks

        Strategy:
            - Sliding window on word tokens
            - Overlap to preserve continuity across boundaries

        Args:
            text: Input raw text
            config: Optional segmenter config (defaults from CONFIG)
            segment_id_prefix: Prefix used to generate segment IDs

        Returns:
            List of DocumentSegment
    """

    cfg = config or get_default_segmenter_config()

    normalized = normalize_text(text)
    if not normalized:
        return []

    tokens = tokenize_words(normalized)
    if not tokens:
        return []

    window = max(1, int(cfg.window_size_tokens))
    overlap = max(0, int(cfg.window_overlap_tokens))
    step = max(1, window - overlap)

    segments: List[DocumentSegment] = []

    ## Build a mapping token_index -> approximate char position using a scan
    try:
        token_spans = _compute_token_spans(normalized)
    except Exception as exc:
        raise PipelineError("Failed to compute token spans for segmentation") from exc

    seg_idx = 0
    for start in range(0, len(tokens), step):
        end = min(start + window, len(tokens))
        if start >= end:
            break

        try:
            start_char = token_spans[start][0] if start < len(token_spans) else 0
            end_char = (
                token_spans[end - 1][1]
                if (end - 1) < len(token_spans)
                else len(normalized)
            )
        except Exception as exc:
            raise PipelineError("Failed to compute segment char offsets") from exc

        seg_text = normalized[start_char:end_char].strip()
        if len(seg_text) < cfg.min_chars_per_segment:
            continue

        segment = DocumentSegment(
            segment_id=f"{segment_id_prefix}_{seg_idx}",
            text=seg_text,
            start_char=start_char,
            end_char=end_char,
            meta={
                "token_start": str(start),
                "token_end": str(end),
                "window": str(window),
                "overlap": str(overlap),
            },
        )
        segments.append(segment)
        seg_idx += 1

        if end >= len(tokens):
            break

    return segments


def _compute_token_spans(text: str) -> List[tuple[int, int]]:
    """
        Compute approximate token spans (start_char, end_char) for word tokens

        Notes:
            - Uses the same regex tokenization as tokenize_words()
            - Provides char offsets to support evidence extraction later

        Args:
            text: Normalized text

        Returns:
            List of (start_char, end_char) spans
    """

    spans: List[tuple[int, int]] = []
    for match in _WORD_RE.finditer(text):
        spans.append((match.start(), match.end()))
    return spans
