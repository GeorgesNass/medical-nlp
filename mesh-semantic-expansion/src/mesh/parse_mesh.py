'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Parse MeSH XML into a normalized JSONL file for fast querying and downstream NLP pipelines."
'''

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.core.config import get_settings
from src.utils.logging_utils import get_logger


## ============================================================
## LOGGER
## ============================================================

logger = get_logger("parse_mesh")


## ============================================================
## XML PARSING HELPERS
## ============================================================

def _safe_text(element: Optional[ET.Element]) -> str:
    """
    Safely extract text from an XML element.

    Args:
        element (Optional[ET.Element]): XML element.

    Returns:
        str: Element text or empty string.
    """

    if element is None or element.text is None:
        return ""
    return element.text.strip()


def _find_all_text(parent: ET.Element, xpath: str) -> List[str]:
    """
    Extract a list of texts matching an XPath under a parent element.

    Args:
        parent (ET.Element): Parent XML element.
        xpath (str): XPath expression.

    Returns:
        List[str]: Extracted texts (non-empty).
    """

    out: List[str] = []
    for node in parent.findall(xpath):
        txt = _safe_text(node)
        if txt:
            out.append(txt)
    return out


def _parse_tree_numbers(record: ET.Element) -> List[str]:
    """
    Parse tree numbers from a DescriptorRecord.

    Args:
        record (ET.Element): DescriptorRecord XML element.

    Returns:
        List[str]: Tree numbers.
    """

    return _find_all_text(record, ".//TreeNumberList/TreeNumber")


def _parse_terms(record: ET.Element) -> Tuple[List[str], List[str]]:
    """
    Parse descriptor preferred term and synonyms.

    The preferred term typically lives under:
      DescriptorName/String

    All terms (preferred + variants) can be found under:
      ConceptList/Concept/TermList/Term/String

    Args:
        record (ET.Element): DescriptorRecord XML element.

    Returns:
        Tuple[List[str], List[str]]: (preferred_terms, synonyms)
    """

    preferred = _find_all_text(record, ".//DescriptorName/String")
    all_terms = _find_all_text(record, ".//ConceptList/Concept/TermList/Term/String")

    preferred_set = {t.lower() for t in preferred}
    synonyms: List[str] = []
    for t in all_terms:
        if t.lower() not in preferred_set:
            synonyms.append(t)

    ## Deduplicate while preserving order
    seen = set()
    synonyms_deduped: List[str] = []
    for t in synonyms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            synonyms_deduped.append(t)

    return preferred, synonyms_deduped


def _parse_descriptor_record(record: ET.Element) -> Dict:
    """
    Convert a DescriptorRecord into a normalized dictionary.

    Normalized output keys are intentionally minimal and stable:
    - ui: MeSH unique identifier
    - preferred_terms: list of preferred labels
    - synonyms: list of alternative labels
    - tree_numbers: list of tree positions
    - scope_note: definition/description if available

    Args:
        record (ET.Element): DescriptorRecord XML element.

    Returns:
        Dict: Normalized record.
    """

    ui = _safe_text(record.find(".//DescriptorUI"))
    preferred_terms, synonyms = _parse_terms(record)
    tree_numbers = _parse_tree_numbers(record)
    scope_note = _safe_text(record.find(".//ScopeNote"))

    return {
        "ui": ui,
        "preferred_terms": preferred_terms,
        "synonyms": synonyms,
        "tree_numbers": tree_numbers,
        "scope_note": scope_note,
        "source": "mesh_xml",
    }


## ============================================================
## MAIN PUBLIC FUNCTIONS
## ============================================================

def iter_descriptor_records(xml_path: Path) -> Iterable[Dict]:
    """
    Stream MeSH DescriptorRecords from an XML file.

    This uses iterparse to keep memory usage low for large MeSH dumps.

    Args:
        xml_path (Path): Path to MeSH XML file.

    Yields:
        Dict: Normalized descriptor record.
    """

    logger.info(f"Streaming MeSH XML: {xml_path}")

    context = ET.iterparse(str(xml_path), events=("end",))
    for event, elem in context:
        ## DescriptorRecord is the main node of interest
        if elem.tag.endswith("DescriptorRecord"):
            record = _parse_descriptor_record(elem)

            ## Yield only valid UI records
            if record.get("ui"):
                yield record
            else:
                logger.debug("Skipping record without UI.")

            ## Free memory
            elem.clear()


def parse_mesh_xml_to_jsonl(
    xml_path: Path,
    output_jsonl_path: Optional[Path] = None,
    max_records: Optional[int] = None,
) -> Path:
    """
    Parse MeSH XML into a normalized JSONL file.

    Each line in JSONL is a single MeSH descriptor record.

    Args:
        xml_path (Path): Path to MeSH XML input file.
        output_jsonl_path (Optional[Path]): Output JSONL path. If None, uses settings.mesh_parsed_file.
        max_records (Optional[int]): Optional maximum number of records (debug/testing).

    Returns:
        Path: Path to generated JSONL file.

    Raises:
        FileNotFoundError: If the input XML file does not exist.
    """

    if not xml_path.exists():
        raise FileNotFoundError(f"MeSH XML not found: {xml_path}")

    settings = get_settings()
    out_path = output_jsonl_path if output_jsonl_path else settings.mesh_parsed_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Parsing MeSH XML to JSONL: {out_path}")

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for record in iter_descriptor_records(xml_path):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

            if count % 1000 == 0:
                logger.info(f"Parsed {count} records...")

            if max_records is not None and count >= max_records:
                logger.warning(f"Stopping early due to max_records={max_records}.")
                break

    logger.info(f"Completed parsing. Total records: {count}")
    return out_path
