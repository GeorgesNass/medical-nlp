'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "FastAPI routes for MeSH search, lookup, and browse endpoints."
'''

from fastapi import APIRouter, HTTPException

from src.core.models import (
    MeshBrowseRequest,
    MeshLookupResponse,
    MeshSearchRequest,
    MeshSearchResult,
)
from src.mesh.query_mesh import browse_tree, lookup_ui, search_mesh
from src.utils.logging_utils import get_logger

logger = get_logger("routes_mesh")

router = APIRouter()


## ============================================================
## ROUTE: SEARCH (FTS)
## ============================================================

@router.post("/search", response_model=list[MeshSearchResult])
def mesh_search(payload: MeshSearchRequest) -> list[MeshSearchResult]:
    """
    Search MeSH using SQLite FTS5.

    Args:
        payload (MeshSearchRequest): Search request payload.

    Returns:
        list[MeshSearchResult]: Ranked results.
    """

    try:
        results = search_mesh(query=payload.query, limit=payload.limit)

        out: list[MeshSearchResult] = []
        for r in results:
            out.append(
                MeshSearchResult(
                    ui=r["ui"],
                    preferred_terms=r["preferred_terms"],
                    synonyms=r["synonyms"],
                    tree_numbers=r["tree_numbers"],
                    score=r["score"],
                )
            )
        return out
    except Exception as e:
        logger.error(f"MeSH search failed: {e}")
        raise HTTPException(status_code=500, detail="MeSH search failed.")


## ============================================================
## ROUTE: LOOKUP (UI)
## ============================================================

@router.get("/lookup/{ui}", response_model=MeshLookupResponse)
def mesh_lookup(ui: str) -> MeshLookupResponse:
    """
    Lookup a MeSH record by UI.

    Args:
        ui (str): MeSH UI.

    Returns:
        MeshLookupResponse: Record details.
    """

    try:
        row = lookup_ui(ui)
        return MeshLookupResponse(**row)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"MeSH lookup failed: {e}")
        raise HTTPException(status_code=500, detail="MeSH lookup failed.")


## ============================================================
## ROUTE: BROWSE (TREE PREFIX)
## ============================================================

@router.post("/browse")
def mesh_browse(payload: MeshBrowseRequest) -> list[dict]:
    """
    Browse MeSH by tree prefix.

    Args:
        payload (MeshBrowseRequest): Browse request payload.

    Returns:
        list[dict]: Minimal browse results.
    """

    try:
        return browse_tree(tree_prefix=payload.tree_prefix, limit=payload.limit)
    except Exception as e:
        logger.error(f"MeSH browse failed: {e}")
        raise HTTPException(status_code=500, detail="MeSH browse failed.")
