'''
__author__ = "Olivia Tortosa"
__copyright__ = None
__credits__ = ["Olivia Tortosa", "Georges Nassopoulos"]
__version__ = "1.0.0"
__maintainer__ = "Georges Nassopoulos"
__email__ = "olivia.tortosa@gmail.com", "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "DIM and demographic metadata extraction from raw laboratory TXT content (gender, date of birth)."
'''

from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, Optional, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
## REGEX PATTERNS (ROBUST / MULTI-FORMAT)
## ============================================================
## Gender patterns
_GENDER_PATTERNS = [
    re.compile(r"\b(sexe)\s*[:\-]\s*(m|masculin)\b", flags=re.IGNORECASE),
    re.compile(r"\b(sexe)\s*[:\-]\s*(f|feminin|féminin)\b", flags=re.IGNORECASE),
    re.compile(r"\bsex\s*[:\-]\s*(m|male)\b", flags=re.IGNORECASE),
    re.compile(r"\bsex\s*[:\-]\s*(f|female)\b", flags=re.IGNORECASE),
]

## DOB patterns (common French hospital formats)
_DOB_PATTERNS = [
    re.compile(
        r"\b(n[ée]e?\s*le|date\s*de\s*naissance|d\.?n\.?)\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(birth\s*date|dob)\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",
        flags=re.IGNORECASE,
    ),
]

## ============================================================
## INTERNAL HELPERS
## ============================================================
def _normalize_gender(raw: str) -> str:
    """
        Normalize gender token into canonical value

        Args:
            raw: Raw token extracted from text

        Returns:
            Canonical gender value: "M", "F" or ""
    """

    token = str(raw).strip().lower()

    if token in {"m", "male", "masculin", "homme"}:
        return "M"

    if token in {"f", "female", "feminin", "féminin", "femme"}:
        return "F"

    return ""

def _parse_date_token(token: str) -> Optional[str]:
    """
        Parse a date token and normalize to YYYY-MM-DD

        Supported formats:
            - dd/mm/yyyy
            - dd-mm-yyyy
            - dd.mm.yyyy
            - dd/mm/yy (assumed 19xx/20xx heuristics)

        Args:
            token: Raw date string

        Returns:
            ISO date string (YYYY-MM-DD) or None
    """

    raw = str(token).strip()

    ## Quick normalization
    raw = raw.replace(".", "/").replace("-", "/")

    parts = raw.split("/")
    if len(parts) != 3:
        return None

    try:
        dd = int(parts[0])
        mm = int(parts[1])
        yy = int(parts[2])

        ## Heuristic: expand 2-digit year
        if yy < 100:
            ## 00-29 => 2000-2029, else 1900-1999
            yy = 2000 + yy if yy <= 29 else 1900 + yy

        d = date(yy, mm, dd)
        return d.isoformat()

    except Exception:
        return None

def _extract_first_match(patterns, text: str) -> Optional[Tuple[str, Tuple[Any, ...]]]:
    """
        Extract the first matching pattern groups

        Args:
            patterns: List of compiled regex
            text: Raw TXT content

        Returns:
            Tuple(pattern_name, groups) or None
    """

    for pat in patterns:
        match = pat.search(text)
        if match:
            return pat.pattern, match.groups()

    return None

## ============================================================
## PUBLIC API
## ============================================================
def extract_gender(text: str) -> str:
    """
        Extract gender from raw TXT content

        High-level workflow:
            1) Try explicit gender patterns
            2) Normalize to "M" or "F"

        Args:
            text: Raw TXT content

        Returns:
            "M", "F" or "" if not found
    """

    ## Try to match explicit gender patterns
    for pat in _GENDER_PATTERNS:
        match = pat.search(text)
        if not match:
            continue

        ## Extract last group containing gender token
        groups = match.groups()
        token = groups[-1] if groups else ""
        gender = _normalize_gender(token)

        if gender:
            return gender

    return ""

def extract_dates_dob(text: str) -> str:
    """
        Extract date of birth from raw TXT content

        High-level workflow:
            1) Try explicit DOB patterns (labels: "née le", "date de naissance", etc.)
            2) Parse and normalize into YYYY-MM-DD
            3) Return first valid result

        Args:
            text: Raw TXT content

        Returns:
            ISO date string (YYYY-MM-DD) or "" if not found
    """

    ## First, try labeled patterns
    for pat in _DOB_PATTERNS[:2]:
        match = pat.search(text)
        if match:
            token = match.groups()[-1]
            dob = _parse_date_token(token)
            if dob:
                return dob

    ## Fallback: any date-like token (less reliable)
    for pat in _DOB_PATTERNS[2:]:
        match = pat.search(text)
        if match:
            token = match.groups()[-1]
            dob = _parse_date_token(token)
            if dob:
                return dob

    return ""

def extract_dim_metadata(text: str) -> Dict[str, str]:
    """
        Extract DIM demographic metadata from raw TXT content

        High-level workflow:
            1) Extract gender
            2) Extract date of birth
            3) Return metadata dict

        Args:
            text: Raw TXT content

        Returns:
            Dictionary with keys: gender, dates_dob
    """

    ## Extract gender
    gender = extract_gender(text)

    ## Extract DOB
    dob = extract_dates_dob(text)

    return {
        "gender": gender,
        "dates_dob": dob,
    }