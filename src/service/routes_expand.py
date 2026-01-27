'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "FastAPI routes for running semantic expansion pipelines on medical documents."
'''

from fastapi import APIRouter, HTTPException

from src.core.models import ExpandRequest, ExpandResponse
from src.pipelines import run_extract_candidates_to_csv
from src.utils.logging_utils import get_logger

logger = get_logger("routes_expand")

router = APIRouter()


## ============================================================
## ROUTE: EXTRACT CANDIDATES -> CSV
## ============================================================

@router.post("/extract_candidates", response_model=ExpandResponse)
def extract_candidates(payload: ExpandRequest) -> ExpandResponse:
    """
    Run the candidate extraction pipeline on a folder of medical documents.

    This endpoint:
        - Reads medical documents from a directory
        - Extracts candidate terms
        - Exports them to a CSV for human validation

    Args:
        payload (ExpandRequest): Expansion pipeline request.

    Returns:
        ExpandResponse: Pipeline execution summary.
    """

    try:
        output_csv, total = run_extract_candidates_to_csv(
            docs_dir=payload.docs_dir,
            output_csv=payload.output_csv,
            max_docs=payload.max_docs,
            enable_faiss=payload.enable_faiss,
        )

        return ExpandResponse(
            status="ok",
            output_csv=str(output_csv),
            total_candidates=total,
            meta={
                "docs_dir": payload.docs_dir,
                "enable_faiss": payload.enable_faiss,
            },
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Expansion pipeline failed: {e}")
        raise HTTPException(status_code=500, detail="Expansion pipeline failed.")
