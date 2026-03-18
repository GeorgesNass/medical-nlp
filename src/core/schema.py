'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Pydantic models for MeSH semantic expansion API requests, responses, monitoring, and pipeline artifacts."
'''

from __future__ import annotations

## ============================================================
## STANDARD IMPORTS
## ============================================================
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover
    BaseSettings = BaseModel  # type: ignore[misc, assignment]
    SettingsConfigDict = dict  # type: ignore[misc, assignment]

## ============================================================
## COMMON TYPES
## ============================================================
JobStatusName = Literal["pending", "running", "success", "failed", "cancelled"]
TaskTypeName = Literal[
    "mesh_search",
    "mesh_lookup",
    "mesh_browse",
    "candidate_extraction",
    "semantic_suggestion",
    "csv_export",
    "evaluation",
]
LogLevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
ValidationStatusName = Literal["", "accepted", "rejected", "unsure"]

## ============================================================
## BASE PYDANTIC SCHEMAS
## ============================================================
class BaseSchema(BaseModel):
    """
        Base schema with shared validation and serialization helpers

        Returns:
            A reusable Pydantic base model
    """

    model_config = {
        "extra": "forbid",
        "populate_by_name": True,
        "str_strip_whitespace": True,
    }

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the model to a Python dictionary

            Returns:
                Serialized model as dictionary
        """

        return self.model_dump()

    def to_json(self) -> str:
        """
            Convert the model to a JSON string

            Returns:
                Serialized model as JSON
        """

        return self.model_dump_json()

    def to_record(self) -> dict[str, Any]:
        """
            Convert the model to a row-oriented dictionary

            Returns:
                Flat dictionary representation
        """

        return self.model_dump(mode="json")

    def to_pandas(self) -> Any:
        """
            Convert the model to a one-row pandas DataFrame

            Returns:
                A pandas DataFrame with one row
        """

        import pandas as pd

        return pd.DataFrame([self.to_record()])

class WarningMixin(BaseSchema):
    """
        Mixin exposing warnings in response payloads

        Args:
            warnings: Warning messages list
    """

    warnings: list[str] = Field(default_factory=list)

## ============================================================
## SETTINGS AND DATACLASS CONFIGS
## ============================================================
class EnvSettings(BaseSettings):
    """
        Runtime settings for mesh-semantic-expansion

        Args:
            app_name: Application name
            environment: Runtime environment
            default_search_limit: Default MeSH search limit
            default_browse_limit: Default browse limit
            default_expand_limit: Default document processing limit
            enable_faiss_default: Whether FAISS is enabled by default
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        env_prefix="MESH_EXPANSION_",
        case_sensitive=False,
    )

    app_name: str = "mesh-semantic-expansion"
    environment: str = "dev"
    default_search_limit: int = Field(default=10, ge=1, le=100)
    default_browse_limit: int = Field(default=50, ge=1, le=500)
    default_expand_limit: int = Field(default=500, ge=1)
    enable_faiss_default: bool = False

@dataclass(frozen=True)
class SearchRuntimeConfig:
    """
        Runtime configuration for MeSH search services

        Args:
            default_limit: Default search limit
            max_limit: Maximum search limit
            browse_limit: Default browse limit
    """

    default_limit: int
    max_limit: int
    browse_limit: int

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

@dataclass(frozen=True)
class ExpansionRuntimeConfig:
    """
        Runtime configuration for candidate expansion pipeline

        Args:
            enable_faiss: Whether FAISS semantic suggestions are enabled
            default_docs_dir: Default document directory
            default_output_csv: Default output CSV path
    """

    enable_faiss: bool
    default_docs_dir: str | None = None
    default_output_csv: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

## ============================================================
## COMMON OPERATIONAL SCHEMAS
## ============================================================
class HealthResponse(BaseSchema):
    """
        Healthcheck response schema

        Args:
            status: Service status
            service: Service name
            version: Application version
            timestamp: Response timestamp
    """

    status: str = "ok"
    service: str = "mesh-semantic-expansion"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ErrorResponse(BaseSchema):
    """
        Standard API error schema

        Args:
            error: Normalized error code
            message: Human-readable message
            origin: Component where the error happened
            details: Diagnostic details
            request_id: Optional request correlation id
    """

    error: str
    message: str
    origin: str = "unknown"
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str = "n/a"

class StatusResponse(BaseSchema):
    """
        Generic status response schema

        Args:
            status: Current status
            message: Optional message
            progress: Optional progress value
            metadata: Optional metadata payload
    """

    status: str
    message: str = ""
    progress: float | None = Field(default=None, ge=0.0, le=100.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

class StructuredLogEvent(BaseSchema):
    """
        Structured log schema

        Args:
            level: Log level
            event: Event name
            message: Human-readable message
            logger_name: Logger name
            context: Additional context
            timestamp: Event timestamp
    """

    level: LogLevelName
    event: str
    message: str
    logger_name: str = "mesh-semantic-expansion"
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class MetricPoint(BaseSchema):
    """
        Monitoring metric schema

        Args:
            name: Metric name
            value: Metric value
            unit: Optional metric unit
            tags: Optional metric tags
    """

    name: str
    value: float
    unit: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)

class MonitoringResponse(WarningMixin):
    """
        Monitoring response schema

        Args:
            metrics: Metric points list
            summary: Aggregated summary
    """

    metrics: list[MetricPoint] = Field(default_factory=list)
    summary: dict[str, float] = Field(default_factory=dict)

## ============================================================
## API MODELS: MESH QUERY
## ============================================================
class MeshSearchRequest(BaseSchema):
    """
        Request model for MeSH text search

        Args:
            query: Free-text query string
            limit: Max number of results
    """

    query: str = Field(..., min_length=1, description="Text query for MeSH search.")
    limit: int = Field(default=10, ge=1, le=100, description="Max number of results.")

class MeshSearchResult(BaseSchema):
    """
        Single MeSH search result

        Args:
            ui: MeSH unique identifier
            preferred_terms: Preferred terms joined string
            synonyms: Synonyms joined string
            tree_numbers: Tree numbers joined string
            score: Search score
    """

    ui: str
    preferred_terms: str
    synonyms: str
    tree_numbers: str
    score: float

class MeshLookupResponse(BaseSchema):
    """
        Response model for MeSH lookup by UI

        Args:
            ui: MeSH unique identifier
            preferred_terms: Preferred terms
            synonyms: Synonyms
            tree_numbers: Tree numbers
            scope_note: Scope note or definition
    """

    ui: str
    preferred_terms: str
    synonyms: str
    tree_numbers: str
    scope_note: str

class MeshBrowseRequest(BaseSchema):
    """
        Request model to browse MeSH by tree prefix

        Args:
            tree_prefix: Tree prefix such as C08
            limit: Max number of results
    """

    tree_prefix: str = Field(..., min_length=1, description="MeSH tree prefix to browse.")
    limit: int = Field(default=50, ge=1, le=500, description="Max number of results.")

## ============================================================
## API MODELS: EXPANSION
## ============================================================
class ExpandRequest(BaseSchema):
    """
        Request model for candidate extraction from medical documents

        Args:
            docs_dir: Path to medical documents directory
            output_csv: Optional output path
            max_docs: Optional max number of documents
            enable_faiss: If True, use FAISS semantic suggestions
    """

    docs_dir: str = Field(..., description="Directory containing medical documents.")
    output_csv: Optional[str] = Field(default=None, description="Optional output CSV path override.")
    max_docs: Optional[int] = Field(default=None, ge=1, description="Optional max number of docs to process.")
    enable_faiss: bool = Field(default=False, description="Use FAISS for semantic suggestions.")

class ExpandResponse(WarningMixin):
    """
        Response model for expansion pipeline

        Args:
            status: Status string
            output_csv: Path to generated CSV
            total_candidates: Number of candidates exported
            meta: Extra metadata
    """

    status: str
    output_csv: str
    total_candidates: int = Field(..., ge=0)
    meta: Dict[str, Any] = Field(default_factory=dict)

## ============================================================
## PIPELINE ARTIFACT MODELS (CSV ROWS)
## ============================================================
class CandidateRow(BaseSchema):
    """
        Candidate row format used for CSV export and human validation

        Args:
            doc_id: Document identifier
            candidate_term: Extracted candidate term
            candidate_type: Candidate type
            context_snippet: Short context window from the document
            mesh_ui_suggested: Suggested MeSH UI
            mesh_label_suggested: Suggested MeSH label
            score: Suggestion score
            human_validation: accepted or rejected or unsure
            human_target_mesh_ui: Existing target UI selected by human
            human_new_entity_label: New entity label chosen by human
            comment: Optional human comment
    """

    doc_id: str
    candidate_term: str
    candidate_type: str
    context_snippet: str = ""
    mesh_ui_suggested: str = ""
    mesh_label_suggested: str = ""
    score: float = 0.0
    human_validation: ValidationStatusName = ""
    human_target_mesh_ui: str = ""
    human_new_entity_label: str = ""
    comment: str = ""

    @model_validator(mode="after")
    def validate_human_review_logic(self) -> "CandidateRow":
        """
            Validate human review consistency

            Returns:
                The validated candidate row

            Raises:
                ValueError: If human review fields are inconsistent
        """

        ## Validate acceptance workflow consistency
        if self.human_validation == "accepted":
            if not self.human_target_mesh_ui and not self.human_new_entity_label:
                raise ValueError(
                    "accepted rows must define human_target_mesh_ui or "
                    "human_new_entity_label"
                )

        ## Validate score lower bound for imported rows
        if self.score < 0.0:
            raise ValueError("score must be >= 0.0")

        return self

## ============================================================
## DATASET AND PIPELINE SCHEMAS
## ============================================================
class DatasetRecord(BaseSchema):
    """
        Generic dataset record schema for semantic expansion

        Args:
            record_id: Record identifier
            payload: Raw payload content
            metadata: Optional metadata
    """

    record_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

class DatasetInput(BaseSchema):
    """
        Dataset input schema

        Args:
            name: Dataset name
            records: Dataset records
    """

    name: str
    records: list[DatasetRecord]

    @field_validator("records")
    @classmethod
    def validate_records(cls, value: list[DatasetRecord]) -> list[DatasetRecord]:
        """
            Validate dataset record list

            Args:
                value: Candidate dataset records

            Returns:
                The validated records list

            Raises:
                ValueError: If records list is empty
        """

        if not value:
            raise ValueError("records must contain at least one item")
        return value

class DatasetOutput(BaseSchema):
    """
        Dataset output schema

        Args:
            name: Dataset name
            row_count: Number of rows
            artifacts: Generated artifacts
    """

    name: str
    row_count: int = Field(..., ge=0)
    artifacts: list[str] = Field(default_factory=list)

class PipelineTask(BaseSchema):
    """
        Pipeline task schema

        Args:
            task_id: Task identifier
            task_type: Task type
            status: Task status
            progress: Task progress percentage
            input_payload: Task input payload
            output_payload: Task output payload
    """

    task_id: str
    task_type: TaskTypeName
    status: JobStatusName = "pending"
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)

class PipelineJob(BaseSchema):
    """
        Pipeline job schema

        Args:
            job_id: Job identifier
            status: Job status
            tasks: Job tasks
            progress: Job progress percentage
            metadata: Job metadata
    """

    job_id: str
    status: JobStatusName = "pending"
    tasks: list[PipelineTask] = Field(default_factory=list)
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_progress(self) -> "PipelineJob":
        """
            Validate progress consistency

            Returns:
                The validated pipeline job

            Raises:
                ValueError: If progress is inconsistent
        """

        if self.tasks and self.progress < min(task.progress for task in self.tasks):
            raise ValueError("job progress cannot be below the minimum task progress")
        return self

## ============================================================
## EVALUATION SCHEMAS
## ============================================================
class EvaluationItem(BaseSchema):
    """
        Evaluation item schema for candidate review outcomes

        Args:
            candidate_type: Candidate type
            accepted_count: Accepted row count
            rejected_count: Rejected row count
            unsure_count: Unsure row count
    """

    candidate_type: str
    accepted_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    unsure_count: int = Field(default=0, ge=0)

class EvaluationResponse(WarningMixin):
    """
        Evaluation response schema

        Args:
            items: Per-type evaluation items
            aggregate: Aggregate metrics
    """

    items: list[EvaluationItem] = Field(default_factory=list)
    aggregate: dict[str, float] = Field(default_factory=dict)