'''
__author__ = "Olivia Tortosa"
__copyright__ = None
__credits__ = ["Olivia Tortosa", "Georges Nassopoulos"]
__version__ = "1.0.0"
__maintainer__ = "Georges Nassopoulos"
__email__ = "olivia.tortosa@gmail.com", "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Analysis group detection from raw laboratory TXT content (biochimie, hemato, coagulation, ionogram, etc.)."
'''

from __future__ import annotations

import re
from typing import Dict

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
## GROUP LABELS
## ============================================================
GROUP_CANONICAL: Dict[str, str] = {
    "biochemistry": "biochemistry",
    "hematology": "hematology",
    "coagulation": "coagulation",
    "ionogram_blood": "ionogram_blood",
    "ionogram_urinary": "ionogram_urinary",
    "cytobacteriology": "cytobacteriology",
    "general": "general",
}

## ============================================================
## REGEX CUES (LIGHTWEIGHT HEURISTICS)
## ============================================================
_GROUP_CUES = {
    "biochemistry": [
        re.compile(r"\b(biochimie|biochimistry|biochemistry)\b", re.IGNORECASE),
        re.compile(r"\b(cr[ée]atinine|ur[ée]e?|glyc[ée]mie|cholest[ée]rol|asat|alat)\b", re.IGNORECASE),
    ],
    "hematology": [
        re.compile(r"\b(h[ée]matologie|hematology|nfs|num[ée]ration)\b", re.IGNORECASE),
        re.compile(r"\b(h[ée]moglobine|plaquettes|leucocytes|h[ée]matocrite|v\.?g\.?m)\b", re.IGNORECASE),
    ],
    "coagulation": [
        re.compile(r"\b(coagulation|h[ée]mostase|hemostasis)\b", re.IGNORECASE),
        re.compile(r"\b(tca|tp\b|inr|fibrinog[èe]ne)\b", re.IGNORECASE),
    ],
    "ionogram_blood": [
        re.compile(r"\b(ionogramme)\b", re.IGNORECASE),
        re.compile(r"\b(natr[ée]mie|kali[ée]mie|chlor[ée]mie|calc[ée]mie)\b", re.IGNORECASE),
        re.compile(r"\b(sang|plasma|s[ée]rum|serum)\b", re.IGNORECASE),
    ],
    "ionogram_urinary": [
        re.compile(r"\b(ionogramme)\b", re.IGNORECASE),
        re.compile(r"\b(natriur[ée]se|kaliur[ée]se|chloruri[ée]se)\b", re.IGNORECASE),
        re.compile(r"\b(urines?|urinary)\b", re.IGNORECASE),
    ],
    "cytobacteriology": [
        re.compile(r"\b(cytobact[ée]riologie|cytobacteriology)\b", re.IGNORECASE),
        re.compile(r"\b(ecbu|leucocyturie|nitrites)\b", re.IGNORECASE),
    ],
}

## ============================================================
## PUBLIC API
## ============================================================
def detect_analysis_group(text: str) -> str:
    """
        Detect analysis group from raw TXT content

        High-level workflow:
            1) Score each group based on regex cue matches
            2) Return best group label
            3) Fallback to "general"

        Args:
            text: Raw TXT content

        Returns:
            Canonical group label
    """

    ## Normalize text for pattern matching
    content = text or ""

    scores: Dict[str, int] = {}
    for group, patterns in _GROUP_CUES.items():
        score = 0

        ## Count cue matches
        for pat in patterns:
            if pat.search(content):
                score += 1

        scores[group] = score

    ## Pick best group
    best_group = "general"
    best_score = 0

    for group, score in scores.items():
        if score > best_score:
            best_group = group
            best_score = score

    return GROUP_CANONICAL.get(best_group, "general")