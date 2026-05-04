'''
__author__ = "Olivia Tortosa"
__copyright__ = None
__credits__ = ["Olivia Tortosa", "Georges Nassopoulos"]
__version__ = "1.0.0"
__maintainer__ = "Georges Nassopoulos"
__email__ = "olivia.tortosa@gmail.com", "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Core analyte extraction logic: segmentation of TXT content and regex-based structured record creation"
'''

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.core.config import AppConfig
from src.core.errors import ValueInterpretationError
from src.parser.interpret_values import interpret_analyte_line
from src.parser.regex_store import ParserResources
from src.utils.logging_utils import get_logger
from src.utils.utils import normalize_text, normalize_clinical_text

logger = get_logger(__name__)

## ============================================================
## SEGMENTATION
## ============================================================
def _segment_text_into_lines(text: str) -> List[str]:
    """
        Segment raw TXT content into candidate lines

        High-level workflow:
            1) Split on newline
            2) Strip whitespace
            3) Remove empty lines

        Args:
            text: Full TXT content

        Returns:
            Cleaned list of lines
    """

    ## Split on line breaks
    raw_lines = text.splitlines()

    ## Strip and filter empty lines
    lines = [line.strip() for line in raw_lines if line.strip()]

    return lines

def _match_analyte_line(
    line: str,
    regex_config: Dict[str, Any],
) -> bool:
    """
        Determine if a line potentially contains an analyte

        High-level workflow:
            1) Iterate over regex patterns
            2) Apply search on current line
            3) Return True if any match

        Args:
            line: Single text line
            regex_config: Regex dictionary

        Returns:
            Boolean indicating potential analyte line
    """

    ## Iterate over configured regex patterns
    for pattern in regex_config.values():

        try:
            if re.search(pattern, line, flags=re.IGNORECASE):
                return True
        except re.error:
            ## Ignore malformed regex patterns safely
            continue

    return False

## ============================================================
## MAIN EXTRACTION FUNCTION
## ============================================================
def extract_analytes_from_text(
    text: str,
    resources: ParserResources,
    config: AppConfig,
    analysis_group: Optional[str] = None,
    gender: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
        Extract structured analyte records from TXT content

        High-level workflow:
            1) Segment text into candidate lines
            2) Detect analyte lines using regex
            3) Interpret each analyte line (with optional metadata)
            4) Accumulate structured records
            5) Return list of dictionaries

        Args:
            text: Raw TXT content
            resources: ParserResources container
            config: AppConfig instance
            analysis_group: Optional analysis group (metadata-derived)
            gender: Optional gender (metadata-derived)

        Returns:
            List of structured analyte records

        Raises:
            ValueInterpretationError: If interpretation fails critically
    """

    ## Segment text
    lines = _segment_text_into_lines(text)
        
    use_fe = getattr(config, "feature_engineering", False)

    records: List[Dict[str, Any]] = []

    ## Iterate through lines
    for line in lines:

        ## Normalize line for matching
        if use_fe:
            normalized_line = normalize_clinical_text(line)
        else:
            normalized_line = normalize_text(line)
            
        ## Detect candidate analyte line
        if not _match_analyte_line(normalized_line, resources.regex):
            continue

        try:
            ## Interpret analyte line (inject metadata to improve norms filtering)
            record = interpret_analyte_line(
                raw_line=line,
                normalized_line=normalized_line,
                resources=resources,
                config=config,
                analysis_group=analysis_group,
                gender=gender,
            )

            if record:
                records.append(record)

        except Exception as exc:
            logger.warning(
                "Failed to interpret line | line=%s | error=%s",
                line,
                str(exc),
            )

    logger.info(
        "Analyte extraction completed | total_records=%s",
        len(records),
    )

    return records