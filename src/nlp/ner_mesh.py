'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Lightweight MeSH entity detection in medical texts using dictionary and SQLite FTS fallback."
'''

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.core.config import get_settings
from src.mesh.query_mesh import search_mesh
from src.utils.logging_utils import get_logger

logger = get_logger("ner_mesh")


## ============================================================
## DATA STRUCTURES
## ============================================================

@dataclass
class DetectedEntity:
    """
    Detected MeSH entity representation.

    Attributes:
        ui (str): Suggested MeSH UI.
        label (str): Matched surface form.
        start (int): Start character index.
        end (int): End character index.
        method (str): Detection method ('dict' or 'fts').
        score (float): Optional confidence score.
    """

    ui: str
    label: str
    start: int
    end: int
    method: str
    score: float = 0.0


## ============================================================
## DICTIONARY BUILDING
## ============================================================

def build_label_dictionary(
    mesh_jsonl_path: Optional[Path] = None,
    max_synonyms: int = 8,
) -> Dict[str, str]:
    """
    Build a lowercase label -> MeSH UI dictionary.

    Notes:
        - Uses preferred terms + limited synonyms.
        - Designed for fast exact matching.
        - Memory-friendly if max_synonyms is kept low.

    Args:
        mesh_jsonl_path (Optional[Path]): Path to mesh_parsed.jsonl.
        max_synonyms (int): Max synonyms kept per UI.

    Returns:
        Dict[str, str]: Mapping from normalized label to MeSH UI.
    """

    import json

    settings = get_settings()
    src_path = mesh_jsonl_path if mesh_jsonl_path else settings.mesh_parsed_file

    if not src_path.exists():
        raise FileNotFoundError(f"MeSH parsed file not found: {src_path}")

    label_to_ui: Dict[str, str] = {}

    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ui = (row.get("ui") or "").strip()
            if not ui:
                continue

            preferred = row.get("preferred_terms", []) or []
            synonyms = (row.get("synonyms", []) or [])[:max_synonyms]

            for term in preferred + synonyms:
                key = term.strip().lower()
                if key and key not in label_to_ui:
                    label_to_ui[key] = ui

    logger.info(f"Label dictionary built: {len(label_to_ui)} entries")
    return label_to_ui


## ============================================================
## ENTITY DETECTION
## ============================================================

def detect_entities(
    text: str,
    label_dict: Optional[Dict[str, str]] = None,
    use_fts_fallback: bool = True,
    max_fts_hits: int = 1,
) -> List[DetectedEntity]:
    """
    Detect MeSH entities in free text.

    Strategy:
        1) Exact match using label dictionary (preferred + synonyms)
        2) Optional fallback using SQLite FTS (term-level)

    Args:
        text (str): Input medical text.
        label_dict (Optional[Dict[str, str]]): Pre-built label dictionary.
        use_fts_fallback (bool): Enable SQLite FTS fallback.
        max_fts_hits (int): Max FTS results per token.

    Returns:
        List[DetectedEntity]: Detected entities.
    """

    detected: List[DetectedEntity] = []
    lowered = text.lower()

    ## --------------------------------------------------------
    ## 1) DICTIONARY MATCHING
    ## --------------------------------------------------------

    if label_dict:
        for label, ui in label_dict.items():
            for match in re.finditer(rf"\b{re.escape(label)}\b", lowered):
                detected.append(
                    DetectedEntity(
                        ui=ui,
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        method="dict",
                        score=1.0,
                    )
                )

    ## --------------------------------------------------------
    ## 2) FTS FALLBACK (TOKEN-LEVEL)
    ## --------------------------------------------------------

    if use_fts_fallback:
        tokens = set(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ\-]{4,}", text))
        for token in tokens:
            try:
                results = search_mesh(query=token, limit=max_fts_hits)
            except Exception:
                continue

            for r in results:
                detected.append(
                    DetectedEntity(
                        ui=r["ui"],
                        label=token,
                        start=-1,
                        end=-1,
                        method="fts",
                        score=float(r.get("score", 0.0)),
                    )
                )

    return detected
