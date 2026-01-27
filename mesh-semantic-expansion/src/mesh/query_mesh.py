'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "High-level MeSH query utilities over the SQLite FTS index."
'''

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.config import get_settings
from src.utils.logging_utils import get_logger

logger = get_logger("query_mesh")


## ============================================================
## DB HELPERS
## ============================================================

def _resolve_db_path(db_path: Optional[Path] = None) -> Path:
    """
    Resolve SQLite DB path.

    Args:
        db_path (Optional[Path]): Optional override.

    Returns:
        Path: Resolved DB path.
    """

    settings = get_settings()
    return db_path if db_path else settings.data_dir / "interim" / "mesh.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    """
    Create a SQLite connection and configure row_factory.

    Args:
        db_path (Path): SQLite DB path.

    Returns:
        sqlite3.Connection: SQLite connection.

    Raises:
        FileNotFoundError: If database does not exist.
    """

    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}. Please build the index first.")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


## ============================================================
## PUBLIC QUERIES
## ============================================================

def search_mesh(
    query: str,
    limit: int = 10,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Search MeSH using SQLite FTS5.

    Args:
        query (str): FTS query string (supports FTS syntax).
        limit (int): Max number of results.
        db_path (Optional[Path]): Optional SQLite DB path override.

    Returns:
        List[Dict[str, Any]]: List of matching entries.
    """

    resolved_db = _resolve_db_path(db_path)
    conn = _connect(resolved_db)

    ## bm25() provides an FTS ranking score (lower is better)
    rows = conn.execute(
        """
        SELECT m.ui, m.preferred_terms, m.synonyms, m.tree_numbers, m.scope_note,
               bm25(mesh_fts) as score
        FROM mesh_fts
        JOIN mesh m ON m.ui = mesh_fts.ui
        WHERE mesh_fts MATCH ?
        ORDER BY score
        LIMIT ?;
        """,
        (query, limit),
    ).fetchall()

    conn.close()

    results: List[Dict[str, Any]] = []
    for r in rows:
        results.append(
            {
                "ui": r["ui"],
                "preferred_terms": r["preferred_terms"],
                "synonyms": r["synonyms"],
                "tree_numbers": r["tree_numbers"],
                "scope_note": r["scope_note"],
                "score": float(r["score"]),
            }
        )

    return results


def lookup_ui(ui: str, db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Lookup a MeSH entry by its UI.

    Args:
        ui (str): MeSH UI (e.g., D012345).
        db_path (Optional[Path]): Optional SQLite DB path override.

    Returns:
        Dict[str, Any]: MeSH entry fields.

    Raises:
        ValueError: If UI is not found.
    """

    resolved_db = _resolve_db_path(db_path)
    conn = _connect(resolved_db)

    row = conn.execute(
        """
        SELECT ui, preferred_terms, synonyms, tree_numbers, scope_note
        FROM mesh
        WHERE ui = ?;
        """,
        (ui,),
    ).fetchone()

    conn.close()

    if row is None:
        raise ValueError(f"UI not found: {ui}")

    return {
        "ui": row["ui"],
        "preferred_terms": row["preferred_terms"],
        "synonyms": row["synonyms"],
        "tree_numbers": row["tree_numbers"],
        "scope_note": row["scope_note"],
    }


def browse_tree(
    tree_prefix: str,
    limit: int = 50,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Browse MeSH entries by tree prefix.

    Args:
        tree_prefix (str): Tree prefix (e.g., 'C08').
        limit (int): Max number of results.
        db_path (Optional[Path]): Optional SQLite DB path override.

    Returns:
        List[Dict[str, Any]]: Matching entries (minimal fields).
    """

    resolved_db = _resolve_db_path(db_path)
    conn = _connect(resolved_db)

    like_pattern = f"%{tree_prefix}%"
    rows = conn.execute(
        """
        SELECT ui, preferred_terms, tree_numbers
        FROM mesh
        WHERE tree_numbers LIKE ?
        LIMIT ?;
        """,
        (like_pattern, limit),
    ).fetchall()

    conn.close()

    return [
        {
            "ui": r["ui"],
            "preferred_terms": r["preferred_terms"],
            "tree_numbers": r["tree_numbers"],
        }
        for r in rows
    ]
