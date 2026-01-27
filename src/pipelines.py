'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "End-to-end pipelines: extract candidates from medical docs, export CSV, apply validation, and build extended MeSH JSON."
'''

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import get_settings
from src.utils.logging_utils import get_logger

logger = get_logger("pipelines")


## ============================================================
## FILE IO HELPERS
## ============================================================

def _list_text_files(docs_dir: Path) -> List[Path]:
    """
    List supported medical document files (text-based) under a folder.

    Args:
        docs_dir (Path): Documents directory.

    Returns:
        List[Path]: List of file paths.
    """

    allowed_ext = {".txt", ".md"}
    files: List[Path] = []
    for p in sorted(docs_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in allowed_ext:
            files.append(p)
    return files


def _read_text(file_path: Path) -> str:
    """
    Read a text file with safe defaults.

    Args:
        file_path (Path): Text file path.

    Returns:
        str: File content.
    """

    return file_path.read_text(encoding="utf-8", errors="ignore")


def _write_csv(rows: List[Dict[str, Any]], output_csv: Path) -> Path:
    """
    Write candidate rows to CSV.

    Args:
        rows (List[Dict[str, Any]]): List of candidate rows as dictionaries.
        output_csv (Path): Output CSV path.

    Returns:
        Path: Written CSV path.
    """

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        ## Still write header-only CSV to keep pipeline deterministic
        header = [
            "doc_id",
            "candidate_term",
            "candidate_type",
            "context_snippet",
            "mesh_ui_suggested",
            "mesh_label_suggested",
            "score",
            "human_validation",
            "human_target_mesh_ui",
            "human_new_entity_label",
            "comment",
        ]
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
        return output_csv

    header = list(rows[0].keys())
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    return output_csv


def _read_csv(input_csv: Path) -> List[Dict[str, str]]:
    """
    Read a CSV into a list of dictionaries.

    Args:
        input_csv (Path): Input CSV path.

    Returns:
        List[Dict[str, str]]: Rows as dicts.
    """

    with open(input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _write_json(output_path: Path, obj: Any) -> Path:
    """
    Write an object to JSON (pretty-printed).

    Args:
        output_path (Path): Output JSON file path.
        obj (Any): JSON-serializable object.

    Returns:
        Path: Written path.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _write_md(output_path: Path, text: str) -> Path:
    """
    Write a markdown report file.

    Args:
        output_path (Path): Output MD path.
        text (str): Markdown content.

    Returns:
        Path: Written path.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


## ============================================================
## CANDIDATE EXTRACTION (BASELINE)
## ============================================================

def _extract_candidates_baseline(
    doc_id: str,
    text: str,
    max_candidates: int = 50,
) -> List[Dict[str, Any]]:
    """
    Baseline candidate extraction from document text.

    Notes:
        - This is intentionally lightweight to keep the pipeline running end-to-end.
        - It can be replaced by src/nlp/expand_terms.py later.
        - Current heuristic: extract non-trivial tokens (length >= 4) and keep unique.

    Args:
        doc_id (str): Document identifier.
        text (str): Document text.
        max_candidates (int): Max candidates per document.

    Returns:
        List[Dict[str, Any]]: Candidate rows.
    """

    import re

    ## Simple tokenization on words + dashes, keep letters only
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ\-]{4,}", text)
    tokens_lower = [t.lower() for t in tokens]

    ## Deduplicate while keeping some ordering
    seen = set()
    uniq: List[str] = []
    for t, tl in zip(tokens, tokens_lower):
        if tl not in seen:
            seen.add(tl)
            uniq.append(t)
        if len(uniq) >= max_candidates:
            break

    rows: List[Dict[str, Any]] = []
    for cand in uniq:
        rows.append(
            {
                "doc_id": doc_id,
                "candidate_term": cand,
                "candidate_type": "baseline_term",
                "context_snippet": "",
                "mesh_ui_suggested": "",
                "mesh_label_suggested": "",
                "score": 0.0,
                "human_validation": "",
                "human_target_mesh_ui": "",
                "human_new_entity_label": "",
                "comment": "",
            }
        )

    return rows


## ============================================================
## PIPELINE: DOCS -> CANDIDATES CSV
## ============================================================

def run_extract_candidates_to_csv(
    docs_dir: str,
    output_csv: Optional[str] = None,
    max_docs: Optional[int] = None,
    enable_faiss: bool = False,
) -> Tuple[Path, int]:
    """
    Run candidate extraction on a folder and export candidates to CSV.

    Args:
        docs_dir (str): Path to folder containing medical documents.
        output_csv (Optional[str]): Optional CSV output path override.
        max_docs (Optional[int]): Optional max number of docs to process.
        enable_faiss (bool): Reserved for future: semantic suggestion via FAISS.

    Returns:
        Tuple[Path, int]: (output_csv_path, total_candidates)

    Raises:
        FileNotFoundError: If docs_dir does not exist.
    """

    settings = get_settings()
    docs_path = Path(docs_dir)

    if not docs_path.exists() or not docs_path.is_dir():
        raise FileNotFoundError(f"Invalid docs_dir: {docs_path}")

    out_csv = Path(output_csv) if output_csv else settings.export_candidates_csv

    files = _list_text_files(docs_path)
    if max_docs is not None:
        files = files[:max_docs]

    logger.info(f"Extracting candidates from docs. Files: {len(files)} | enable_faiss={enable_faiss}")

    all_rows: List[Dict[str, Any]] = []
    for fp in files:
        doc_id = fp.name
        text = _read_text(fp)
        rows = _extract_candidates_baseline(doc_id=doc_id, text=text)
        all_rows.extend(rows)

    _write_csv(all_rows, out_csv)

    logger.info(f"Candidates CSV created: {out_csv} | total={len(all_rows)}")
    return out_csv, len(all_rows)


## ============================================================
## PIPELINE: APPLY VALIDATION (FILTER ACCEPTED)
## ============================================================

def apply_validated_csv(
    validated_csv_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Read validated CSV and keep accepted rows only.

    Expected values (case-insensitive) in 'human_validation':
        - accepted
        - rejected
        - unsure

    Args:
        validated_csv_path (Optional[str]): Optional validated CSV path override.

    Returns:
        List[Dict[str, str]]: Accepted rows only.

    Raises:
        FileNotFoundError: If validated CSV does not exist.
    """

    settings = get_settings()
    csv_path = Path(validated_csv_path) if validated_csv_path else settings.export_candidates_validated_csv

    if not csv_path.exists():
        raise FileNotFoundError(f"Validated CSV not found: {csv_path}")

    rows = _read_csv(csv_path)
    accepted: List[Dict[str, str]] = []

    for r in rows:
        status = (r.get("human_validation") or "").strip().lower()
        if status == "accepted":
            accepted.append(r)

    logger.info(f"Validated CSV loaded: {csv_path} | accepted={len(accepted)} / total={len(rows)}")
    return accepted


## ============================================================
## PIPELINE: BUILD EXTENDED MESH JSON + REPORT
## ============================================================

def build_extended_mesh(
    validated_csv_path: Optional[str] = None,
    output_json_path: Optional[str] = None,
    output_report_path: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Build an extended MeSH JSON artifact from a validated CSV.

    Rules:
        - If human_target_mesh_ui is provided => map to existing UI
        - Else if human_new_entity_label is provided => create new entity
        - Else fallback to candidate_term as new entity label

    Args:
        validated_csv_path (Optional[str]): Path to validated CSV.
        output_json_path (Optional[str]): Output JSON path.
        output_report_path (Optional[str]): Output report path.

    Returns:
        Tuple[Path, Path]: (mesh_extended_json_path, report_diff_md_path)
    """

    settings = get_settings()
    accepted = apply_validated_csv(validated_csv_path)

    out_json = Path(output_json_path) if output_json_path else settings.mesh_extended_json
    out_report = Path(output_report_path) if output_report_path else settings.report_diff_md

    new_entities: List[Dict[str, Any]] = []
    mappings: List[Dict[str, Any]] = []

    for r in accepted:
        candidate_term = (r.get("candidate_term") or "").strip()
        target_ui = (r.get("human_target_mesh_ui") or "").strip()
        new_label = (r.get("human_new_entity_label") or "").strip()
        comment = (r.get("comment") or "").strip()

        if target_ui:
            mappings.append(
                {
                    "candidate_term": candidate_term,
                    "mapped_to_ui": target_ui,
                    "doc_id": r.get("doc_id", ""),
                    "comment": comment,
                }
            )
            continue

        label = new_label if new_label else candidate_term
        if not label:
            continue

        ## A simple "new id" strategy (stable enough for MVP)
        new_entities.append(
            {
                "ui": "",
                "label": label,
                "source_term": candidate_term,
                "doc_id": r.get("doc_id", ""),
                "comment": comment,
            }
        )

    mesh_extended = {
        "meta": {
            "version": settings.app_version,
            "accepted_rows": len(accepted),
            "new_entities": len(new_entities),
            "mappings": len(mappings),
        },
        "mappings_to_existing": mappings,
        "new_entities": new_entities,
    }

    _write_json(out_json, mesh_extended)

    report = (
        "# MeSH Extended Report\n\n"
        f"- Accepted rows: **{len(accepted)}**\n"
        f"- New entities created: **{len(new_entities)}**\n"
        f"- Mappings to existing UI: **{len(mappings)}**\n\n"
        "## Notes\n"
        "- `mappings_to_existing`: accepted terms that point to an existing MeSH UI.\n"
        "- `new_entities`: accepted terms considered as new entries (no existing UI provided).\n"
    )
    _write_md(out_report, report)

    logger.info(f"Extended MeSH JSON created: {out_json}")
    logger.info(f"Report created: {out_report}")

    return out_json, out_report
