'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "End-to-end tests for MeSH indexing and querying using a minimal synthetic dataset."
'''

import json
from pathlib import Path
from typing import List

import pytest

from src.mesh.index_mesh import build_sqlite_fts_index
from src.mesh.query_mesh import browse_tree, lookup_ui, search_mesh


## ============================================================
## TEST FIXTURES
## ============================================================

@pytest.fixture()
def tmp_mesh_jsonl(tmp_path: Path) -> Path:
    """Create a minimal mesh_parsed.jsonl file for E2E tests.

    This fixture generates a small synthetic MeSH dataset in JSONL format,
    compatible with the project's indexing pipeline.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        Path: Path to the generated JSONL file.
    """

    records: List[dict] = [
        {
            "ui": "D000001",
            "preferred_terms": ["Myocardial Infarction"],
            "synonyms": ["Heart Attack", "MI"],
            "tree_numbers": ["C14.280.647"],
            "scope_note": "An infarction of the myocardium.",
            "source": "mesh_xml",
        },
        {
            "ui": "D000002",
            "preferred_terms": ["Hypertension"],
            "synonyms": ["High Blood Pressure", "HTN"],
            "tree_numbers": ["C14.907.489"],
            "scope_note": "Persistently high arterial blood pressure.",
            "source": "mesh_xml",
        },
        {
            "ui": "D000003",
            "preferred_terms": ["Diabetes Mellitus"],
            "synonyms": ["Diabetes", "DM"],
            "tree_numbers": ["C19.246"],
            "scope_note": "A group of metabolic diseases.",
            "source": "mesh_xml",
        },
    ]

    out = tmp_path / "mesh_parsed.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return out


@pytest.fixture()
def tmp_sqlite_db(tmp_path: Path) -> Path:
    """Provide a SQLite DB output path for tests.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        Path: Path to the SQLite database file.
    """

    return tmp_path / "mesh.db"


@pytest.fixture()
def indexed_db(tmp_mesh_jsonl: Path, tmp_sqlite_db: Path) -> Path:
    """Build a SQLite FTS index from the temporary JSONL fixture.

    This fixture creates a fresh SQLite database with an FTS5 index using
    the synthetic MeSH JSONL input produced by `tmp_mesh_jsonl`.

    Args:
        tmp_mesh_jsonl (Path): Path to the temporary MeSH JSONL input.
        tmp_sqlite_db (Path): Path to the temporary SQLite database output.

    Returns:
        Path: Path to the built SQLite database.
    """

    db_path = build_sqlite_fts_index(
        mesh_jsonl_path=tmp_mesh_jsonl,
        sqlite_db_path=tmp_sqlite_db,
        overwrite=True,
    )
    return db_path


## ============================================================
## E2E TESTS
## ============================================================

def test_e2e_search_mesh(indexed_db: Path) -> None:
    """Search MeSH concepts using the FTS index.

    This test validates that a known term returns the expected MeSH UI.

    Args:
        indexed_db (Path): Path to the indexed SQLite database.
    """

    results = search_mesh(query="Hypertension", limit=5, db_path=indexed_db)
    assert len(results) >= 1
    assert results[0]["ui"] == "D000002"


def test_e2e_lookup_ui(indexed_db: Path) -> None:
    """Lookup a MeSH concept by its UI.

    This test validates that the lookup returns the expected record and that
    key fields are present in the response.

    Args:
        indexed_db (Path): Path to the indexed SQLite database.
    """

    row = lookup_ui("D000001", db_path=indexed_db)
    assert row["ui"] == "D000001"
    assert "Myocardial Infarction" in row["preferred_terms"]


def test_e2e_browse_tree(indexed_db: Path) -> None:
    """Browse MeSH concepts by tree prefix.

    This test validates that browsing by a shared prefix returns the expected
    UIs from the synthetic dataset.

    Args:
        indexed_db (Path): Path to the indexed SQLite database.
    """

    rows = browse_tree(tree_prefix="C14", limit=10, db_path=indexed_db)
    assert len(rows) >= 2
    uis = {r["ui"] for r in rows}
    assert "D000001" in uis
    assert "D000002" in uis
