'''
__author__ = "Olivia Tortosa"
__copyright__ = None
__credits__ = ["Olivia Tortosa", "Georges Nassopoulos"]
__version__ = "1.0.0"
__maintainer__ = "Georges Nassopoulos"
__email__ = "olivia.tortosa@gmail.com", "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Interpretation layer: extract analyte name, patient value, unit and reference norms from a matched TXT line"
'''

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from src.core.config import AppConfig
from src.core.errors import ValueInterpretationError
from src.parser.check_norms import compute_status_from_norms
from src.parser.regex_store import ParserResources
from src.parser.unit_conversion import normalize_unit
from src.utils.logging_utils import get_logger
from src.utils.utils import safe_float, safe_strip, normalize_clinical_text

logger = get_logger(__name__)

## ============================================================
## INTERNAL HELPERS
## ============================================================
def _extract_first_number(line: str) -> Optional[float]:
    """
        Extract first numeric value from a text line

        High-level workflow:
            1) Search for number pattern
            2) Normalize decimal separator
            3) Convert safely to float

        Args:
            line: Raw text line

        Returns:
            Float value if found otherwise None
    """

    ## Regex to capture first numeric token
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", line)

    if not match:
        return None

    return safe_float(match.group(0))

def _extract_unit(line: str) -> Optional[str]:
    """
        Extract potential measurement unit from line

        High-level workflow:
            1) Search for alphabetical token after numeric value
            2) Return candidate unit

        Args:
            line: Raw text line

        Returns:
            Unit string if detected otherwise None
    """

    ## Simple heuristic: number followed by unit
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?\s*([a-zA-Z/%µ]+)", line)

    if not match:
        return None

    return safe_strip(match.group(1))

def _select_gender_column(gender: Optional[str]) -> str:
    """
        Select norms column name based on detected gender

        Args:
            gender: Gender string (free text)

        Returns:
            One of: "Homme", "Femme"
            Fallback: "Femme" if unknown
    """

    if not gender:
        return "Femme"

    g = safe_strip(str(gender)).lower()

    if g in {"m", "male", "homme", "h"}:
        return "Homme"

    if g in {"f", "female", "femme"}:
        return "Femme"

    return "Femme"

def _parse_norm_interval(value_str: Any) -> Tuple[Optional[float], Optional[float]]:
    """
        Parse a norms interval string into (min, max)

        Supported patterns:
            - "150-240"
            - "<10"
            - ">3.5"
            - "≤10" / "≥3" (best-effort)
            - single numeric value => treated as max (conservative)

        Args:
            value_str: Raw cell value from norms table

        Returns:
            Tuple (min_value, max_value)
    """

    if value_str is None:
        return None, None

    s = safe_strip(str(value_str))
    if not s:
        return None, None

    ## Normalize decimals
    s = s.replace(",", ".")

    ## Range "a-b"
    m = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)\s*-\s*([-+]?\d+(?:\.\d+)?)\s*$", s)
    if m:
        return safe_float(m.group(1)), safe_float(m.group(2))

    ## Lower than "<x" or "≤x"
    m = re.match(r"^\s*[<≤]\s*([-+]?\d+(?:\.\d+)?)\s*$", s)
    if m:
        return None, safe_float(m.group(1))

    ## Greater than ">x" or "≥x"
    m = re.match(r"^\s*[>≥]\s*([-+]?\d+(?:\.\d+)?)\s*$", s)
    if m:
        return safe_float(m.group(1)), None

    ## Single numeric (fallback)
    m = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)\s*$", s)
    if m:
        return None, safe_float(m.group(1))

    return None, None

def _extract_analyte_name_and_category(
    line: str,
    resources: ParserResources,
) -> Tuple[str, str]:
    """
        Attempt to extract analyte name and category using canonical keywords dictionary

        Notes:
            - resources.keywords is expected to be: {category: {analyte_key: {...}}}
            - We match using:
                1) the stored regex (if provided)
                2) the display_name substring (fallback)

        Args:
            line: Raw text line
            resources: ParserResources

        Returns:
            Tuple (analyte_name, category)
    """

    lowered = (line or "").lower()

    for category, analytes in resources.keywords.items():
        if not isinstance(analytes, dict):
            continue

        for analyte_key, cfg in analytes.items():
            if not isinstance(cfg, dict):
                continue

            ## Try regex match first
            pattern = safe_strip(str(cfg.get("regex", "")))
            if pattern:
                try:
                    if re.search(pattern, line, flags=re.IGNORECASE):
                        return analyte_key, str(category)
                except re.error:
                    pass

            ## Fallback on display name substring
            display_name = safe_strip(str(cfg.get("display_name", analyte_key)))
            if display_name and display_name.lower() in lowered:
                return analyte_key, str(category)

    ## Fallback: return trimmed raw line head with unknown category
    return safe_strip(line[:50]), ""

def _lookup_norms_for_analyte(
    analyte: str,
    category: Optional[str],
    gender: Optional[str],
    resources: ParserResources,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
        Lookup norms interval for a given analyte with category + gender filtering

        High-level workflow:
            1) Load canonical norms table from resources.norms_tables["canonique"]
            2) Filter by categorie if provided
            3) Filter by Type == analyte (case-insensitive)
            4) Select gender column (Homme/Femme)
            5) Parse interval to (min, max)
            6) Return (min, max, metric)

        Args:
            analyte: Detected analyte key
            category: Analysis group (optional)
            gender: Gender (optional)
            resources: ParserResources

        Returns:
            Tuple (norms_min, norms_max, norms_metric)
    """

    if not resources.norms_tables:
        return None, None, None

    norms_df = resources.norms_tables.get("canonique", None)
    if norms_df is None or norms_df.empty:
        return None, None, None

    df = norms_df

    ## Filter by category if available
    if category and "categorie" in df.columns:
        df = df[df["categorie"].astype(str).str.strip().str.lower() == safe_strip(category).lower()]

    ## Filter by analyte name (Type column)
    if "Type" not in df.columns:
        return None, None, None

    df = df[df["Type"].astype(str).str.strip().str.lower() == safe_strip(analyte).lower()]

    if df.empty:
        return None, None, None

    ## Select gender column
    gender_col = _select_gender_column(gender)

    ## Pick first row (stable)
    row = df.iloc[0]

    ## Parse interval from gender cell
    norms_min, norms_max = _parse_norm_interval(row.get(gender_col, None))

    ## Metric from file
    norms_metric = safe_strip(str(row.get("Metric", ""))) or None

    return norms_min, norms_max, norms_metric

## ============================================================
## MAIN INTERPRETATION FUNCTION
## ============================================================
def interpret_analyte_line(
    raw_line: str,
    normalized_line: str,
    resources: ParserResources,
    config: AppConfig,
    analysis_group: Optional[str] = None,
    gender: Optional[str] = None,
) -> Dict[str, Any]:
    """
        Interpret a single analyte line into structured fields

        High-level workflow:
            1) Extract analyte name (+ inferred category)
            2) Extract patient value
            3) Extract unit
            4) Normalize unit
            5) Retrieve reference norms (category + gender)
            6) Compute status (low/normal/high)
            7) Return structured record dictionary

        Args:
            raw_line: Original text line
            normalized_line: Normalized version of line
            resources: ParserResources container
            config: AppConfig instance
            analysis_group: Optional analysis group (metadata-derived)
            gender: Optional gender (metadata-derived)

        Returns:
            Structured record dictionary

        Raises:
            ValueInterpretationError: If interpretation fails critically
    """

    try:
        ## Extract analyte name and inferred category
        analyte, inferred_category = _extract_analyte_name_and_category(raw_line, resources)

        ## Prefer metadata category
        resolved_category = analysis_group if analysis_group else inferred_category

        line_for_extraction = normalize_clinical_text(raw_line) if use_fe else raw_line

        ## Extract patient value
        #patient_value = _extract_first_number(raw_line)
        patient_value = _extract_first_number(line_for_extraction)

        ## Extract unit
        #raw_unit = _extract_unit(raw_line)
        raw_unit = _extract_unit(line_for_extraction)

        ## Normalize unit if possible
        normalized_unit = normalize_unit(
            unit=raw_unit,
            analyte=analyte,
            resources=resources,
        )

        ## Retrieve norms from canonical table
        norms_min, norms_max, norms_metric = _lookup_norms_for_analyte(
            analyte=analyte,
            category=resolved_category,
            gender=gender,
            resources=resources,
        )

        ## Compute status (low/normal/high)
        status = compute_status_from_norms(
            value=patient_value,
            norms_min=norms_min,
            norms_max=norms_max,
        )

        ## Build structured record
        record: Dict[str, Any] = {
            "analyzed_variable": analyte,
            "raw_data_entry": raw_line,
            "structured_data_origin_value": patient_value,
            "structured_data_origin_metric": raw_unit,
            "structured_data_transform_value": patient_value,
            "structured_data_transform_metric": normalized_unit,
            "norms_min": norms_min,
            "norms_max": norms_max,
            "norms_metric": norms_metric,
            "status": status,
            "normalized_text": line_for_extraction if use_fe else None,
            "char_length": len(line_for_extraction) if use_fe else None,
            "token_count": len(line_for_extraction.split()) if use_fe else None,
        }

        return record

    except Exception as exc:
        logger.error(
            "Interpretation failed | line=%s | error=%s",
            raw_line,
            str(exc),
        )
        logger.debug("Traceback:", exc_info=True)

        raise ValueInterpretationError(
            message="Failed to interpret analyte line",
            details={"line": raw_line, "error": str(exc)},
        )