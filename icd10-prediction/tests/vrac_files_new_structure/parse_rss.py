'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Fixed-width RSS parser for ICD10 hospital records. Extracts structured fields and builds a consolidated CSV dataset."
'''

from __future__ import annotations

## Standard library imports
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

## Third-party imports
import pandas as pd

## Internal imports
from src.utils.logging_utils import get_logger
from src.core.errors import (
    log_and_raise_missing_folder,
    log_and_raise_parsing_error,
)

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("parse_rss", log_file="parse_rss.log")

## ============================================================
## DATA STRUCTURES
## ============================================================
@dataclass
class RSSRecord:
    """
        Structured representation of one RSS line

        Attributes:
            admission_id: Hospital admission identifier
            rss_id: RSS unique identifier
            rum_id: RUM identifier
            facility_id: Facility identifier
            birth_date: Patient birth date (raw format)
            sex: Patient sex
            admission_date: Admission date
            discharge_date: Discharge date
            primary_diagnosis_code: Primary ICD10 diagnosis (target)
            related_diagnosis_code: Related diagnosis if present
            associated_diagnosis_codes: Additional diagnosis codes
            procedure_codes: Medical procedure codes
    """

    admission_id: str
    rss_id: str
    rum_id: str
    facility_id: str
    birth_date: Optional[str]
    sex: Optional[str]
    admission_date: Optional[str]
    discharge_date: Optional[str]
    primary_diagnosis_code: str
    related_diagnosis_code: Optional[str]
    associated_diagnosis_codes: List[str]
    procedure_codes: List[str]

## ============================================================
## INTERNAL FIXED-WIDTH PARSING UTILITIES
## ============================================================
def _safe_slice(line: str, start: int, end: int) -> str:
    """
        Safely slice fixed-width string

        Args:
            line: Raw RSS line
            start: Start index
            end: End index

        Returns:
            Stripped substring
    """
    
    ## Ensure slicing does not raise unexpected errors
    return line[start:end].strip()

def _parse_associated_codes(raw_segment: str) -> List[str]:
    """
        Extract associated diagnosis codes from raw segment

        Args:
            raw_segment: Raw string segment

        Returns:
            List of ICD10 codes
    """

    ## Split by whitespace
    tokens = raw_segment.split()

    ## Collect valid ICD-like tokens
    codes: List[str] = []

    for token in tokens:
        ## Basic heuristic:
        ## - ICD10 codes typically start with a letter
        ## - Length >= 3
        if len(token) >= 3 and token[0].isalpha():
            codes.append(token.strip())

    ## Remove duplicates while preserving order
    return list(dict.fromkeys(codes))

## ============================================================
## LINE PARSING LOGIC
## ============================================================
def _parse_line(line: str) -> RSSRecord:
    """
        Parse a single fixed-width RSS line

        Args:
            line: Raw RSS line

        Returns:
            RSSRecord instance

        Raises:
            ParsingError: If critical fields missing
    """

    ## Defensive parsing to isolate failure per line
    try:
        ## Core identifiers
        admission_id = _safe_slice(line, 47, 67)
        rss_id = _safe_slice(line, 27, 47)
        rum_id = _safe_slice(line, 67, 77)
        facility_id = _safe_slice(line, 16, 24)

        ## Patient metadata
        birth_date = _safe_slice(line, 77, 85)
        sex = _safe_slice(line, 85, 86)

        ## Admission / discharge dates
        admission_date = _safe_slice(line, 92, 100)
        discharge_date = _safe_slice(line, 102, 110)

        ## Primary target variable (prediction label)
        primary_diagnosis_code = _safe_slice(line, 140, 148)

        ## Related diagnosis (optional)
        related_diagnosis_code = _safe_slice(line, 148, 156)

        ## Remaining segment after fixed-width zone
        additional_segment = line[192:]

        ## Extract additional diagnosis codes
        associated_codes = _parse_associated_codes(additional_segment)

        ## Procedure codes placeholder (future refinement possible)
        procedure_codes: List[str] = []

        ## Return structured dataclass instance
        return RSSRecord(
            admission_id=admission_id,
            rss_id=rss_id,
            rum_id=rum_id,
            facility_id=facility_id,
            birth_date=birth_date or None,
            sex=sex or None,
            admission_date=admission_date or None,
            discharge_date=discharge_date or None,
            primary_diagnosis_code=primary_diagnosis_code,
            related_diagnosis_code=related_diagnosis_code or None,
            associated_diagnosis_codes=associated_codes,
            procedure_codes=procedure_codes,
        )

    except Exception as exc:
        ## Raise structured parsing error
        log_and_raise_parsing_error(Path("rss_line"), str(exc))

## ============================================================
## PUBLIC API
## ============================================================
def parse_rss_folder(rss_folder: str | Path) -> pd.DataFrame:
    """
        Parse all RSS files inside a folder and return consolidated DataFrame

        Workflow:
            1) Iterate over .rss files
            2) Parse each line as RSSRecord
            3) Aggregate into DataFrame

        Args:
            rss_folder: Folder containing .rss files

        Returns:
            Pandas DataFrame with structured RSS data
    """

    ## Resolve path
    rss_path = Path(rss_folder)

    ## Validate folder existence
    if not rss_path.exists():
        log_and_raise_missing_folder(rss_path)

    ## Storage for parsed records
    records: List[RSSRecord] = []

    ## Iterate through RSS files
    for file_path in sorted(rss_path.glob("*.rss*")):
        logger.info("Parsing RSS file: %s", file_path.name)

        ## Open file safely
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:

                ## Skip empty lines
                if not line.strip():
                    continue

                ## Parse individual line
                record = _parse_line(line)
                records.append(record)

    logger.info("Parsed %d RSS records", len(records))

    ## Convert dataclasses to DataFrame
    return pd.DataFrame([r.__dict__ for r in records])