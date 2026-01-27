'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Pydantic models for API requests/responses and pipeline artifacts."
'''

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


## ============================================================
## API MODELS: MeSH QUERY
## ============================================================

class MeshSearchRequest(BaseModel):
    """
    Request model for MeSH text search.

    Attributes:
        query (str): Free-text query string.
        limit (int): Max number of results.
    """

    query: str = Field(..., min_length=1, description="Text query for MeSH search.")
    limit: int = Field(default=10, ge=1, le=100, description="Max number of results.")


class MeshSearchResult(BaseModel):
    """
    Single MeSH search result.

    Attributes:
        ui (str): MeSH unique identifier.
        preferred_terms (str): Preferred terms (joined string).
        synonyms (str): Synonyms (joined string).
        tree_numbers (str): Tree numbers (joined string).
        score (float): Search score.
    """

    ui: str
    preferred_terms: str
    synonyms: str
    tree_numbers: str
    score: float


class MeshLookupResponse(BaseModel):
    """
    Response model for MeSH lookup by UI.

    Attributes:
        ui (str): MeSH unique identifier.
        preferred_terms (str): Preferred terms.
        synonyms (str): Synonyms.
        tree_numbers (str): Tree numbers.
        scope_note (str): Scope note / definition.
    """

    ui: str
    preferred_terms: str
    synonyms: str
    tree_numbers: str
    scope_note: str


class MeshBrowseRequest(BaseModel):
    """
    Request model to browse MeSH by tree prefix.

    Attributes:
        tree_prefix (str): Tree prefix (e.g., "C08").
        limit (int): Max number of results.
    """

    tree_prefix: str = Field(..., min_length=1, description="MeSH tree prefix to browse.")
    limit: int = Field(default=50, ge=1, le=500, description="Max number of results.")


## ============================================================
## API MODELS: EXPANSION
## ============================================================

class ExpandRequest(BaseModel):
    """
    Request model for candidate extraction from medical documents.

    Attributes:
        docs_dir (str): Path to medical documents directory.
        output_csv (Optional[str]): Optional output path.
        max_docs (Optional[int]): Optional max number of documents.
        enable_faiss (bool): If True, use FAISS semantic suggestions.
    """

    docs_dir: str = Field(..., description="Directory containing medical documents.")
    output_csv: Optional[str] = Field(default=None, description="Optional output CSV path override.")
    max_docs: Optional[int] = Field(default=None, ge=1, description="Optional max number of docs to process.")
    enable_faiss: bool = Field(default=False, description="Use FAISS for semantic suggestions.")


class ExpandResponse(BaseModel):
    """
    Response model for expansion pipeline.

    Attributes:
        status (str): Status string.
        output_csv (str): Path to generated CSV.
        total_candidates (int): Number of candidates exported.
        meta (Dict[str, Any]): Extra metadata.
    """

    status: str
    output_csv: str
    total_candidates: int
    meta: Dict[str, Any] = Field(default_factory=dict)


## ============================================================
## PIPELINE ARTIFACT MODELS (CSV ROWS)
## ============================================================

class CandidateRow(BaseModel):
    """
    Candidate row format used for CSV export and human validation.

    Attributes:
        doc_id (str): Document identifier.
        candidate_term (str): Extracted candidate term.
        candidate_type (str): e.g. 'synonym', 'abbreviation', 'new_term'.
        context_snippet (str): Short context window from the document.
        mesh_ui_suggested (str): Suggested MeSH UI (optional).
        mesh_label_suggested (str): Suggested MeSH label (optional).
        score (float): Suggestion score.
        human_validation (str): accepted/rejected/unsure (optional).
        human_target_mesh_ui (str): If accepted and mapped to existing UI (optional).
        human_new_entity_label (str): If accepted as new entity label (optional).
        comment (str): Optional human comment.
    """

    doc_id: str
    candidate_term: str
    candidate_type: str
    context_snippet: str = ""
    mesh_ui_suggested: str = ""
    mesh_label_suggested: str = ""
    score: float = 0.0

    human_validation: str = ""
    human_target_mesh_ui: str = ""
    human_new_entity_label: str = ""
    comment: str = ""
