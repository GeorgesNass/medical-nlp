'''
__author__ = "Olivia Tortosa"
__copyright__ = None
__credits__ = ["Olivia Tortosa", "Georges Nassopoulos"]
__version__ = "1.0.0"
__maintainer__ = "Georges Nassopoulos"
__email__ = "olivia.tortosa@gmail.com", "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Time metadata extraction from raw laboratory TXT content (sampling time, report/edition date)."
'''

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Dict, Optional, Tuple

from src.utils.utils import normalize_clinical_text
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
## REGEX PATTERNS (ROBUST / MULTI-FORMAT)
## ============================================================
## Sampling time patterns (French hospital conventions)
_SAMPLING_TIME_PATTERNS = [
    re.compile(
        r"\b(heure\s*de\s*pr[ée]l[èe]vement|heure\s*pr[ée]l[èe]vement|h\.?\s*pr[ée]l[èe]v\.?)\s*[:\-]?\s*([0-2]?\d[:h][0-5]\d)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(pr[ée]l[èe]vement)\s*[:\-]?\s*(?:le\s*)?(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}).{0,40}?([0-2]?\d[:h][0-5]\d)\b",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b([0-2]?\d[:h][0-5]\d)\b",
        flags=re.IGNORECASE,
    ),
]

## Edition/report date patterns
_EDITION_DATE_PATTERNS = [
    re.compile(
        r"\b(date\s*d['’]?[ée]dition|date\s*d['’]?impression|[ée]dit[ée]\s*le|imprim[ée]\s*le|rendu\s*le)\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(compte\s*rendu|rapport)\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",
        flags=re.IGNORECASE,
    ),
]

## ============================================================
## INTERNAL HELPERS
## ============================================================
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
            yy = 2000 + yy if yy <= 29 else 1900 + yy

        d = date(yy, mm, dd)
        return d.isoformat()

    except Exception:
        return None

def _parse_time_token(token: str) -> Optional[str]:
    """
        Parse a time token and normalize to HH:MM

        Supported formats:
            - HH:MM
            - H:MM
            - HHhMM
            - HhMM

        Args:
            token: Raw time string

        Returns:
            Normalized HH:MM or None
    """

    raw = str(token).strip().lower()
    raw = raw.replace("h", ":")

    parts = raw.split(":")
    if len(parts) != 2:
        return None

    try:
        hh = int(parts[0])
        mm = int(parts[1])

        if hh < 0 or hh > 23:
            return None

        if mm < 0 or mm > 59:
            return None

        return f"{hh:02d}:{mm:02d}"

    except Exception:
        return None

def _extract_first_date(text: str) -> str:
    """
        Extract the first valid edition/report date

        Args:
            text: Raw TXT content

        Returns:
            ISO date string or ""
    """

    ## Try labeled patterns first
    for pat in _EDITION_DATE_PATTERNS:
        match = pat.search(text)
        if not match:
            continue

        token = match.groups()[-1]
        parsed = _parse_date_token(token)
        if parsed:
            return parsed

    return ""

def _extract_first_time(text: str) -> str:
    """
        Extract the first valid sampling time

        Args:
            text: Raw TXT content

        Returns:
            HH:MM or ""
    """

    ## Try specific patterns first
    for pat in _SAMPLING_TIME_PATTERNS[:2]:
        match = pat.search(text)
        if not match:
            continue

        groups = match.groups()
        token = groups[-1] if groups else ""
        parsed = _parse_time_token(token)
        if parsed:
            return parsed

    ## Fallback: any time-like token (less reliable)
    for pat in _SAMPLING_TIME_PATTERNS[2:]:
        match = pat.search(text)
        if not match:
            continue

        token = match.groups()[-1]
        parsed = _parse_time_token(token)
        if parsed:
            return parsed

    return ""

## ============================================================
## PUBLIC API
## ============================================================
def extract_sampling_time(text: str) -> str:
    """
        Extract sampling time from raw TXT content

        High-level workflow:
            1) Try labeled sampling time patterns
            2) Normalize to HH:MM
            3) Fallback to any time-like token

        Args:
            text: Raw TXT content

        Returns:
            Sampling time HH:MM or "" if not found
    """

    return _extract_first_time(text)

def extract_dates_edition(text: str) -> str:
    """
        Extract report/edition date from raw TXT content

        High-level workflow:
            1) Try labeled edition date patterns
            2) Normalize to YYYY-MM-DD

        Args:
            text: Raw TXT content

        Returns:
            Edition date YYYY-MM-DD or "" if not found
    """

    return _extract_first_date(text)

def extract_time_metadata(text: str) -> Dict[str, str]:
    """
        Extract time-related metadata from raw TXT content

        High-level workflow:
            1) Extract sampling time
            2) Extract edition date
            3) Return metadata dict

        Args:
            text: Raw TXT content

        Returns:
            Dictionary with keys: sampling_time, dates_edition
    """

    normalized_text = normalize_clinical_text(text)
    
    ## Extract sampling time
    #sampling_time = extract_sampling_time(text)
    sampling_time = extract_sampling_time(normalized_text)

    ## Extract edition date
    #edition_date = extract_dates_edition(text)
    edition_date = extract_dates_edition(normalized_text)

    return {
        "sampling_time": sampling_time,
        "dates_edition": edition_date,
    }