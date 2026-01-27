'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "CLI helper utilities for running the project pipelines from Python (used by scripts/tests)."
'''

from pathlib import Path
from typing import Optional, Tuple

from src.core.config import get_settings
from src.mesh.download_mesh import download_mesh
from src.mesh.index_mesh import build_faiss_index, build_sqlite_fts_index
from src.mesh.parse_mesh import parse_mesh_xml_to_jsonl
from src.nlp.embeddings import build_mesh_embeddings
from src.pipelines import build_extended_mesh, run_extract_candidates_to_csv
from src.utils.logging_utils import get_logger

logger = get_logger("utils_cli")


## ============================================================
## HIGH-LEVEL COMMAND WRAPPERS
## ============================================================

def cmd_download_mesh(url: str, overwrite: bool = False) -> Tuple[Path, Path]:
    """
    Download MeSH artifact from an URL.

    Args:
        url (str): MeSH URL (XML/ZIP/etc).
        overwrite (bool): Overwrite existing file.

    Returns:
        Tuple[Path, Path]: (downloaded_path, checksum_path)
    """

    return download_mesh(url=url, overwrite=overwrite)


def cmd_parse_mesh(xml_path: Optional[str] = None, overwrite: bool = False) -> Path:
    """
    Parse MeSH XML into JSONL (mesh_parsed.jsonl).

    Args:
        xml_path (Optional[str]): Optional input XML path.
        overwrite (bool): Overwrite output JSONL if exists.

    Returns:
        Path: Output JSONL path.
    """

    settings = get_settings()
    src = Path(xml_path) if xml_path else _guess_mesh_xml_file(settings.raw_mesh_dir)

    if not src.exists():
        raise FileNotFoundError(f"MeSH XML not found: {src}")

    out = settings.mesh_parsed_file
    if out.exists() and overwrite:
        out.unlink()

    if out.exists() and not overwrite:
        logger.info(f"Mesh parsed JSONL already exists: {out}")
        return out

    return parse_mesh_xml_to_jsonl(xml_path=src, output_jsonl_path=out)


def cmd_index_sqlite(overwrite: bool = False) -> Path:
    """
    Build SQLite FTS index from mesh_parsed.jsonl.

    Args:
        overwrite (bool): Overwrite existing DB.

    Returns:
        Path: SQLite DB path.
    """

    return build_sqlite_fts_index(overwrite=overwrite)


def cmd_build_embeddings(overwrite: bool = False) -> Path:
    """
    Build MeSH embeddings parquet from mesh_parsed.jsonl.

    Args:
        overwrite (bool): Overwrite existing embeddings parquet.

    Returns:
        Path: Embeddings parquet path.
    """

    return build_mesh_embeddings(overwrite=overwrite)


def cmd_index_faiss(overwrite: bool = False) -> Tuple[Path, Path]:
    """
    Build FAISS index from mesh_embeddings.parquet.

    Args:
        overwrite (bool): Overwrite existing FAISS index/map.

    Returns:
        Tuple[Path, Path]: (faiss_index_path, id_map_path)
    """

    return build_faiss_index(overwrite=overwrite)


def cmd_extract_candidates(
    docs_dir: Optional[str] = None,
    output_csv: Optional[str] = None,
    max_docs: Optional[int] = None,
    enable_faiss: bool = False,
) -> Tuple[Path, int]:
    """
    Extract candidate terms from docs and export CSV.

    Args:
        docs_dir (Optional[str]): Documents directory override.
        output_csv (Optional[str]): Output CSV override.
        max_docs (Optional[int]): Max number of docs.
        enable_faiss (bool): Include FAISS suggestions (requires built FAISS index).

    Returns:
        Tuple[Path, int]: (output_csv_path, total_candidates)
    """

    settings = get_settings()
    docs = docs_dir if docs_dir else str(settings.raw_medical_docs_dir)
    return run_extract_candidates_to_csv(
        docs_dir=docs,
        output_csv=output_csv,
        max_docs=max_docs,
        enable_faiss=enable_faiss,
    )


def cmd_build_extended_mesh(
    validated_csv_path: Optional[str] = None,
    output_json_path: Optional[str] = None,
    output_report_path: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Build mesh_extended.json + report_diff.md from a validated CSV.

    Args:
        validated_csv_path (Optional[str]): Path to validated CSV override.
        output_json_path (Optional[str]): Output JSON override.
        output_report_path (Optional[str]): Output report override.

    Returns:
        Tuple[Path, Path]: (mesh_extended_json_path, report_path)
    """

    return build_extended_mesh(
        validated_csv_path=validated_csv_path,
        output_json_path=output_json_path,
        output_report_path=output_report_path,
    )


## ============================================================
## INTERNAL HELPERS
## ============================================================

def _guess_mesh_xml_file(mesh_dir: Path) -> Path:
    """
    Guess a MeSH XML file from a directory.

    Args:
        mesh_dir (Path): Directory containing MeSH artifacts.

    Returns:
        Path: First .xml found in the directory.

    Raises:
        FileNotFoundError: If no XML file is found.
    """

    xml_files = sorted([p for p in mesh_dir.rglob("*.xml") if p.is_file()])
    if not xml_files:
        raise FileNotFoundError(f"No .xml file found in: {mesh_dir}")
    return xml_files[0]
