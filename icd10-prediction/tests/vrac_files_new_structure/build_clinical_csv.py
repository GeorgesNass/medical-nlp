'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Merge RSS structured data with raw clinical record folders to generate one CSV per admission_id."
'''

from __future__ import annotations

## Standard library imports
import json
from pathlib import Path
from typing import Dict, List, Tuple

## Third-party imports
import pandas as pd

## Internal imports
from src.utils.logging_utils import get_logger
from src.core.errors import (
    log_and_raise_missing_folder,
    log_and_raise_missing_file,
)

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("build_clinical_csv", log_file="build_clinical_csv.log")

## ============================================================
## CONSTANTS
## ============================================================
ALLOWED_DOCUMENT_TYPES: Tuple[str, ...] = ( ## Expected labels (document types) extracted from hashtags
    "crh",
    "cro",
    "cra",
    "ordonnance-examen",
    "ordonnance-medicaments",
    "analyse-labo",
    "fiche-patient-admission",
)
HASHTAG_PREFIX = "#" ## Characters used to prefix document types inside filenames
TAG_FILENAME_SEPARATOR = "_" ## Separator used between tags and real filename

## ============================================================
## INTERNAL HELPERS
## ============================================================
def _normalize_document_type(tag: str) -> str:
    """
        Normalize a raw hashtag token into a canonical document type

        Args:
            tag: Raw tag extracted from filename

        Returns:
            Normalized document type string
    """
    
    ## Lowercase for consistency
    normalized = tag.strip().lower()

    ## Convert common separators to hyphen
    normalized = normalized.replace(" ", "-").replace("_", "-")

    ## Keep only allowed types if match
    if normalized in set(ALLOWED_DOCUMENT_TYPES):
        return normalized

    ## Unknown tags are kept as-is (still useful for diagnostics)
    return normalized

def _extract_tags_and_clean_name(filename: str) -> Tuple[List[str], str]:
    """
        Extract document types (hashtags) and clean the file name

        Examples:
            "#crh#cro#ordonnance_medicaments_report.txt"
            -> (["crh", "cro", "ordonnance-medicaments"], "report.txt")

        Rules:
            - Tags are prefixed by '#'
            - Tags may be chained without separators
            - Real file name starts after the first '_' when present
            - If '_' is not present, we remove leading hashtags and keep remainder

        Args:
            filename: Raw filename string

        Returns:
            Tuple (document_types, clean_file_name)
    """

    ## Default outputs
    document_types: List[str] = []
    clean_file_name = filename

    ## Fast path: no hashtag
    if HASHTAG_PREFIX not in filename:
        return document_types, clean_file_name

    ## Split by '_' to isolate tags segment
    if TAG_FILENAME_SEPARATOR in filename:
        tag_part, name_part = filename.split(TAG_FILENAME_SEPARATOR, 1)
        clean_file_name = name_part
    else:
        tag_part = filename
        clean_file_name = filename.lstrip(HASHTAG_PREFIX)

    ## Extract tags from the tag_part: "#a#b#c" -> ["a", "b", "c"]
    raw_tags = [t for t in tag_part.split(HASHTAG_PREFIX) if t.strip()]

    ## Normalize tags
    for t in raw_tags:
        document_types.append(_normalize_document_type(t))

    ## Deduplicate while preserving order
    document_types = list(dict.fromkeys(document_types))

    return document_types, clean_file_name

def _read_text_as_single_line(file_path: Path) -> str:
    """
        Read text file and return content as a single line

        Notes:
            - Content is preserved (no cleaning)
            - Newlines are replaced by spaces for CSV readability

        Args:
            file_path: Path to .txt file

        Returns:
            Text content as a single line
    """
    
    ## Read raw text with tolerant decoding
    text = file_path.read_text(encoding="utf-8", errors="ignore")

    ## Replace line breaks for one-line CSV storage
    return " ".join(text.splitlines()).strip()

def _ensure_required_columns(rss_df: pd.DataFrame) -> None:
    """
        Validate required columns exist in RSS DataFrame

        Args:
            rss_df: DataFrame produced by parse_rss_folder

        Raises:
            KeyError: If required fields are missing
    """
    
    required = {
        "admission_id",
        "primary_diagnosis_code",
    }

    missing = sorted(list(required - set(rss_df.columns)))
    if missing:
        raise KeyError(f"RSS DataFrame missing required columns: {missing}")

def _rss_rows_for_admission(
    rss_df: pd.DataFrame,
    admission_id: str,
) -> List[Dict[str, str]]:
    """
        Extract RSS rows for a given admission_id

        Notes:
            - RSS may contain multiple lines per admission_id
            - We keep all rows and store them per document line

        Args:
            rss_df: Consolidated RSS DataFrame
            admission_id: Admission identifier

        Returns:
            List of RSS row dicts (stringified for safe CSV)
    """
    
    subset = rss_df[rss_df["admission_id"] == admission_id]
    rows: List[Dict[str, str]] = []

    ## Convert each row into serializable dict
    for _, r in subset.iterrows():
        row_dict: Dict[str, str] = {}
        for col in rss_df.columns:
            value = r.get(col)

            ## Store lists as JSON strings if present
            if isinstance(value, list):
                row_dict[col] = json.dumps(value, ensure_ascii=False)
            else:
                row_dict[col] = "" if pd.isna(value) else str(value)

        rows.append(row_dict)

    return rows

## ============================================================
## PUBLIC API
## ============================================================
def build_clinical_records_csv(
    clinical_records_dir: str | Path,
    rss_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
        Generate one CSV per admission_id by merging RSS fields with document files

        Output schema (one row per file):
            - admission_id
            - <all RSS columns...>
            - document_types (json list)
            - file_name
            - text_content

        Folder structure assumption:
            clinical_records_dir/<admission_id>/*.txt

        Args:
            clinical_records_dir: Root folder containing admission subfolders
            rss_df: Consolidated RSS DataFrame
            output_dir: Output folder for per-admission CSVs
    """
    
    ## Resolve and validate paths
    records_root = Path(clinical_records_dir)
    out_dir = Path(output_dir)

    if not records_root.exists():
        log_and_raise_missing_folder(records_root)

    out_dir.mkdir(parents=True, exist_ok=True)

    ## Validate RSS structure
    _ensure_required_columns(rss_df)

    ## Iterate over admission folders
    admission_folders = sorted([p for p in records_root.iterdir() if p.is_dir()])

    logger.info("Found %d admission folders", len(admission_folders))

    for admission_folder in admission_folders:
        admission_id = admission_folder.name.strip()

        ## Skip empty admission_id folder names
        if not admission_id:
            continue

        ## Collect RSS rows for this admission_id
        rss_rows = _rss_rows_for_admission(rss_df, admission_id)

        ## If no RSS match, still export documents for diagnostics
        if not rss_rows:
            logger.warning("No RSS rows found for admission_id=%s", admission_id)

        ## Collect all .txt files
        txt_files = sorted(admission_folder.glob("*.txt"))

        if not txt_files:
            logger.warning("No .txt files found for admission_id=%s", admission_id)
            continue

        ## Build rows: one row per file, enriched with first RSS row (or empty)
        rows_out: List[Dict[str, str]] = []

        ## Choose the first RSS row as default "main" metadata
        rss_main: Dict[str, str] = rss_rows[0] if rss_rows else {}

        for file_path in txt_files:
            ## Extract tags and clean file name from filename only
            doc_types, clean_name = _extract_tags_and_clean_name(file_path.name)

            ## Read file content as one line
            text_content = _read_text_as_single_line(file_path)

            ## Assemble output row
            row: Dict[str, str] = {}

            ## Add admission id first
            row["admission_id"] = admission_id

            ## Add RSS columns (flattened)
            for k, v in rss_main.items():
                row[k] = v

            ## Add document-specific columns
            row["document_types"] = json.dumps(doc_types, ensure_ascii=False)
            row["file_name"] = clean_name
            row["text_content"] = text_content

            rows_out.append(row)

        ## Write one CSV per admission_id
        df_out = pd.DataFrame(rows_out)

        csv_path = out_dir / f"{admission_id}.csv"
        df_out.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info(
            "Exported admission_id=%s | files=%d | csv=%s",
            admission_id,
            len(rows_out),
            csv_path.name,
        )

def build_icd10_consolidated_csv(
    rss_df: pd.DataFrame,
    output_path: str | Path,
    sort_by: str = "admission_date",
) -> None:
    """
        Export a single consolidated CSV from RSS DataFrame

        Notes:
            - Output is stored in data/interim/icd10_csv/
            - Sorting can be done by date or by file/year if added later

        Args:
            rss_df: Consolidated RSS DataFrame
            output_path: Output CSV path
            sort_by: Column to sort by if present
    """
    
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ## Sort only if column exists
    df = rss_df.copy()
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=True)

    df.to_csv(out_path, index=False, encoding="utf-8")

    logger.info("Exported consolidated ICD10 CSV: %s", out_path)