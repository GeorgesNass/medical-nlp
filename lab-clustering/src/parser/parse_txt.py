'''
__author__ = "Olivia Tortosa"
__copyright__ = None
__credits__ = ["Olivia Tortosa", "Georges Nassopoulos"]
__version__ = "1.0.0"
__maintainer__ = "Georges Nassopoulos"
__email__ = "olivia.tortosa@gmail.com", "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main TXT parsing entrypoint: orchestrates segmentation, analyte extraction, interpretation and final DataFrame formatting"
'''

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.core.config import AppConfig
from src.core.errors import ParsingError
from src.parser.extract_analytes import extract_analytes_from_text
from src.parser.format_output import format_structured_output
from src.parser.regex_store import load_parser_resources
from src.metadata.metadata_builder import build_metadata
from src.utils.utils import normalize_clinical_text
from src.utils.io_utils import assert_exists
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
## TXT LOADER
## ============================================================
def _read_txt_file(path: Path) -> str:
    """
        Read raw TXT file content

        High-level workflow:
            1) Ensure file exists
            2) Open with utf-8 encoding
            3) Return full text content

        Args:
            path: Path to TXT file

        Returns:
            Full text content
    """

    ## Ensure file exists
    assert_exists(path, kind="file")

    ## Read text content
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    return content

## ============================================================
## MAIN PARSER ENTRYPOINT
## ============================================================
def parse_txt_file(
    txt_path: Path,
    config: AppConfig,
) -> pd.DataFrame:
    """
        Parse a laboratory TXT file into structured DataFrame

        High-level workflow:
            1) Load parser resources (regex, keywords, norms, conversions)
            2) Read TXT content
            3) Extract analyte-level structured records
            4) Format output to final CSV schema
            5) Return structured DataFrame

        Args:
            txt_path: Path to TXT file
            config: AppConfig instance

        Returns:
            Structured DataFrame ready for CSV export

        Raises:
            ParsingError: If parsing fails
    """

    try:
        ## Load parser resources (regex, keywords, norms, etc.)
        resources = load_parser_resources(config)

        ## Read TXT content
        text_content = _read_txt_file(txt_path)

        use_fe = getattr(config, "feature_engineering", False)

        if use_fe:
            text_content = normalize_clinical_text(text_content)
            
        if not text_content.strip():
            raise ParsingError(
                message="Empty TXT file",
                details={"file": str(txt_path)},
            )

        ## Build metadata (gender, dob, sampling_time, edition_date, analysis_group)
        metadata = build_metadata(
            text=text_content,
            source_file=txt_path.name,
        )

        if use_fe:
            metadata["char_length"] = len(text_content)
            metadata["token_count"] = len(text_content.split())
            
        ## Extract analyte-level records
        records: List[dict] = extract_analytes_from_text(
            text=text_content,
            resources=resources,
            config=config,
        )

        if not records:
            logger.warning("No analytes extracted | file=%s", txt_path.name)

        ## Convert list of dicts to DataFrame
        raw_df = pd.DataFrame(records)

        ## Inject metadata columns into raw_df (same values for all rows)
        if raw_df.empty:
            raw_df = pd.DataFrame([{}])

        for key, value in metadata.items():
            raw_df[key] = value

        ## Apply final formatting to match official schema
        structured_df = format_structured_output(
            df=raw_df,
            source_file=txt_path.name,
            config=config,
        )

        logger.info(
            "Parsing completed | file=%s | rows=%s | cols=%s",
            txt_path.name,
            structured_df.shape[0],
            structured_df.shape[1],
        )

        return structured_df

    except Exception as exc:
        logger.error(
            "Parsing failed | file=%s | error=%s",
            txt_path,
            str(exc),
        )
        logger.debug("Traceback:", exc_info=True)

        raise ParsingError(
            message="Failed to parse TXT file",
            details={
                "file": str(txt_path),
                "error": str(exc),
            },
        )