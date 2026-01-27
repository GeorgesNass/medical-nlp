'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Build and query a fast MeSH index using SQLite FTS5 (text) and optional FAISS (semantic)."
'''

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import get_settings
from src.utils.logging_utils import get_logger

logger = get_logger("index_mesh")


## ============================================================
## SQLITE (FTS5) INDEX
## ============================================================

def build_sqlite_fts_index(
    mesh_jsonl_path: Optional[Path] = None,
    sqlite_db_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """
    Build a SQLite database with FTS5 index from a MeSH JSONL file.

    Args:
        mesh_jsonl_path (Optional[Path]): Path to mesh_parsed.jsonl.
        sqlite_db_path (Optional[Path]): Output SQLite db path.
        overwrite (bool): If True, overwrite an existing database.

    Returns:
        Path: Path to SQLite database.

    Raises:
        FileNotFoundError: If input JSONL does not exist.
    """

    settings = get_settings()
    src_path = mesh_jsonl_path if mesh_jsonl_path else settings.mesh_parsed_file
    db_path = sqlite_db_path if sqlite_db_path else settings.data_dir / "interim" / "mesh.db"

    if not src_path.exists():
        raise FileNotFoundError(f"MeSH JSONL file not found: {src_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists() and overwrite:
        db_path.unlink()

    if db_path.exists() and not overwrite:
        logger.info(f"SQLite DB already exists: {db_path}")
        return db_path

    logger.info(f"Building SQLite FTS index: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    ## Base table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mesh (
            ui TEXT PRIMARY KEY,
            preferred_terms TEXT,
            synonyms TEXT,
            tree_numbers TEXT,
            scope_note TEXT
        );
        """
    )

    ## FTS table: index text fields
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS mesh_fts
        USING fts5(
            ui,
            preferred_terms,
            synonyms,
            scope_note,
            content='mesh',
            content_rowid='rowid'
        );
        """
    )

    ## Insert rows
    count = 0
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ui = (row.get("ui") or "").strip()
            if not ui:
                continue

            preferred = " | ".join(row.get("preferred_terms", []) or [])
            synonyms = " | ".join(row.get("synonyms", []) or [])
            tree_numbers = " | ".join(row.get("tree_numbers", []) or [])
            scope_note = (row.get("scope_note") or "").strip()

            conn.execute(
                """
                INSERT OR REPLACE INTO mesh(ui, preferred_terms, synonyms, tree_numbers, scope_note)
                VALUES (?, ?, ?, ?, ?);
                """,
                (ui, preferred, synonyms, tree_numbers, scope_note),
            )

            count += 1
            if count % 2000 == 0:
                conn.commit()
                logger.info(f"Inserted {count} records...")

    conn.commit()

    ## Rebuild FTS from content table
    conn.execute("INSERT INTO mesh_fts(mesh_fts) VALUES('rebuild');")
    conn.commit()
    conn.close()

    logger.info(f"SQLite FTS build completed. Records: {count}")
    return db_path


def search_sqlite_fts(
    query: str,
    sqlite_db_path: Optional[Path] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search MeSH via SQLite FTS5.

    Args:
        query (str): Text query (FTS syntax supported).
        sqlite_db_path (Optional[Path]): Path to SQLite database.
        limit (int): Max number of results.

    Returns:
        List[Dict[str, Any]]: Search results with basic fields.
    """

    settings = get_settings()
    db_path = sqlite_db_path if sqlite_db_path else settings.data_dir / "interim" / "mesh.db"

    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}. Build it first.")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT m.ui, m.preferred_terms, m.synonyms, m.tree_numbers,
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
                "fts_score": float(r["score"]),
            }
        )

    return results


## ============================================================
## FAISS INDEX (SEMANTIC SEARCH)
## ============================================================

def _import_faiss():
    """
    Import FAISS with a clear error message if not installed.

    Returns:
        module: faiss module.

    Raises:
        ImportError: If faiss is not installed.
    """

    try:
        import faiss  # type: ignore
        return faiss
    except Exception as e:
        raise ImportError(
            "FAISS is not installed. Please install one of:\n"
            "- faiss-cpu\n"
            "- faiss-gpu\n"
            "Then retry building the FAISS index."
        ) from e


def _load_embeddings_parquet(embeddings_path: Path) -> Tuple[List[str], "Any"]:
    """
    Load MeSH embeddings from a parquet file.

    Expected columns:
    - ui (str)
    - vector (list[float]) OR emb_0..emb_n (float columns)

    Args:
        embeddings_path (Path): Path to mesh_embeddings.parquet.

    Returns:
        Tuple[List[str], Any]: (ui_list, vectors_np)

    Raises:
        FileNotFoundError: If parquet file not found.
        ValueError: If expected columns not found.
    """

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as e:
        raise ImportError("Missing dependency: numpy/pandas required to load embeddings parquet.") from e

    df = pd.read_parquet(embeddings_path)

    if "ui" not in df.columns:
        raise ValueError("Embeddings parquet must contain a 'ui' column.")

    ## Case A: 'vector' column contains list-like vectors
    if "vector" in df.columns:
        vectors = np.vstack(df["vector"].to_list()).astype("float32")
        ui_list = df["ui"].astype(str).to_list()
        return ui_list, vectors

    ## Case B: emb_0..emb_n numeric columns
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("Embeddings parquet must contain 'vector' or 'emb_0..emb_n' columns.")

    vectors = df[emb_cols].to_numpy(dtype="float32")
    ui_list = df["ui"].astype(str).to_list()
    return ui_list, vectors


def build_faiss_index(
    embeddings_parquet_path: Optional[Path] = None,
    faiss_index_path: Optional[Path] = None,
    id_map_path: Optional[Path] = None,
    normalize: bool = True,
    overwrite: bool = False,
) -> Tuple[Path, Path]:
    """
    Build a FAISS index from precomputed MeSH embeddings.

    This function expects embeddings to be generated elsewhere (src/nlp/embeddings.py).

    Args:
        embeddings_parquet_path (Optional[Path]): Path to mesh_embeddings.parquet.
        faiss_index_path (Optional[Path]): Output path for FAISS index file.
        id_map_path (Optional[Path]): Output path for FAISS id->ui mapping JSONL.
        normalize (bool): If True, L2-normalize vectors and use cosine similarity (IndexFlatIP).
        overwrite (bool): If True, overwrite existing files.

    Returns:
        Tuple[Path, Path]: (faiss_index_path, id_map_path)
    """

    settings = get_settings()

    emb_path = embeddings_parquet_path if embeddings_parquet_path else settings.mesh_embeddings_file
    index_path = faiss_index_path if faiss_index_path else settings.data_dir / "interim" / "mesh.faiss"
    map_path = id_map_path if id_map_path else settings.data_dir / "interim" / "mesh_faiss_id_map.jsonl"

    index_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)

    if index_path.exists() and map_path.exists() and not overwrite:
        logger.info("FAISS index and map already exist. Skipping build.")
        return index_path, map_path

    if overwrite:
        if index_path.exists():
            index_path.unlink()
        if map_path.exists():
            map_path.unlink()

    faiss = _import_faiss()

    ui_list, vectors = _load_embeddings_parquet(emb_path)

    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise ImportError("Missing dependency: numpy required for FAISS operations.") from e

    if normalize:
        ## Cosine similarity via dot product on normalized vectors
        faiss.normalize_L2(vectors)

    dim = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dim)

    logger.info(f"Building FAISS index. Vectors: {vectors.shape[0]}, dim: {dim}, normalize: {normalize}")
    index.add(vectors)

    ## Save index
    faiss.write_index(index, str(index_path))

    ## Save id -> ui map
    with open(map_path, "w", encoding="utf-8") as f:
        for idx, ui in enumerate(ui_list):
            f.write(json.dumps({"faiss_id": idx, "ui": ui}, ensure_ascii=False) + "\n")

    logger.info(f"FAISS index saved: {index_path}")
    logger.info(f"FAISS id map saved: {map_path}")

    return index_path, map_path


def _load_id_map(id_map_path: Path) -> List[str]:
    """
    Load FAISS id -> UI map.

    Args:
        id_map_path (Path): Path to jsonl map.

    Returns:
        List[str]: ui_list where index corresponds to FAISS id.
    """

    ui_list: List[str] = []
    with open(id_map_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ui_list.append(obj["ui"])
    return ui_list


def search_faiss(
    query_vector: "Any",
    faiss_index_path: Optional[Path] = None,
    id_map_path: Optional[Path] = None,
    top_k: int = 10,
    normalize_query: bool = True,
) -> List[Dict[str, Any]]:
    """
    Search MeSH via FAISS semantic similarity (requires a query vector).

    Notes:
        - Query vector must already be generated by embeddings backend.
        - If the FAISS index was built with normalized vectors, keep normalize_query=True.

    Args:
        query_vector (Any): Numpy array shaped (dim,) or (1, dim), float32.
        faiss_index_path (Optional[Path]): Path to faiss index file.
        id_map_path (Optional[Path]): Path to id->ui map jsonl.
        top_k (int): Number of neighbors.
        normalize_query (bool): Normalize query vector for cosine similarity.

    Returns:
        List[Dict[str, Any]]: List of {ui, faiss_score}.
    """

    settings = get_settings()
    index_path = faiss_index_path if faiss_index_path else settings.data_dir / "interim" / "mesh.faiss"
    map_path = id_map_path if id_map_path else settings.data_dir / "interim" / "mesh_faiss_id_map.jsonl"

    if not index_path.exists() or not map_path.exists():
        raise FileNotFoundError("FAISS index/map not found. Build it first with build_faiss_index().")

    faiss = _import_faiss()

    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise ImportError("Missing dependency: numpy required for FAISS operations.") from e

    index = faiss.read_index(str(index_path))
    ui_list = _load_id_map(map_path)

    q = query_vector
    if isinstance(q, list):
        q = np.array(q, dtype="float32")

    if q.ndim == 1:
        q = q.reshape(1, -1)

    q = q.astype("float32")

    if normalize_query:
        faiss.normalize_L2(q)

    scores, ids = index.search(q, top_k)

    results: List[Dict[str, Any]] = []
    for faiss_id, score in zip(ids[0].tolist(), scores[0].tolist()):
        if faiss_id < 0:
            continue
        results.append({"ui": ui_list[faiss_id], "faiss_score": float(score)})

    return results


## ============================================================
## HYBRID SEARCH (FTS + FAISS)
## ============================================================

def search_hybrid(
    query: str,
    query_vector: Optional["Any"] = None,
    limit_fts: int = 10,
    top_k_faiss: int = 10,
    enable_faiss: bool = False,
) -> Dict[str, Any]:
    """
    Hybrid search: text search via SQLite FTS + optional semantic search via FAISS.

    Args:
        query (str): Query text.
        query_vector (Optional[Any]): Query embedding vector for FAISS search.
        limit_fts (int): Number of FTS results.
        top_k_faiss (int): Number of FAISS neighbors.
        enable_faiss (bool): If True, include FAISS results.

    Returns:
        Dict[str, Any]: Combined results.
    """

    results: Dict[str, Any] = {
        "query": query,
        "fts": search_sqlite_fts(query=query, limit=limit_fts),
        "faiss": [],
    }

    if enable_faiss:
        if query_vector is None:
            raise ValueError("enable_faiss=True requires a query_vector.")
        results["faiss"] = search_faiss(query_vector=query_vector, top_k=top_k_faiss)

    return results
