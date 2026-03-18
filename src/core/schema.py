'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Domain schemas for the medical document classification pipeline using segment-level similarity and multi-label aggregation."
'''

from __future__ import annotations

## STANDARD IMPORTS
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

## COMMON TYPES
LabelName = Literal[
    "CRH",
    "CRO",
    "CRA",
    "PRESCRIPTION",
    "LAB_RESULTS",
    "ADMISSION_FORM",
]

JobStatusName = Literal[
    "pending",
    "running",
    "success",
    "failed",
]

TaskTypeName = Literal[
    "eda",
    "segment_documents",
    "encode_segments",
    "build_similarity_index",
    "retrieve_similar_segments",
    "aggregate_labels",
    "export_predictions",
]

LogLevelName = Literal[
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]

## ============================================================
## BASE PYDANTIC SCHEMA
## ============================================================
class BaseSchema(BaseModel):
    """
        Base schema providing common serialization helpers

        Design choice:
            All pipeline response models inherit from this schema
            to ensure consistent JSON export and logging behavior.

        Returns:
            A reusable Pydantic schema base class
    """

    model_config = {
        "extra": "forbid",
        "populate_by_name": True,
        "str_strip_whitespace": True,
    }

    def to_dict(self) -> dict[str, Any]:
        """
            Convert model instance to dictionary

            Returns:
                Dictionary representation
        """

        return self.model_dump()

    def to_json(self) -> str:
        """
            Convert model instance to JSON string

            Returns:
                JSON string
        """

        return self.model_dump_json()

## ============================================================
## DOMAIN DATACLASSES
## ============================================================
@dataclass(frozen=True)
class Document:
    """
        Raw clinical document container

        This structure represents the initial input object before
        segmentation and embedding generation.

        Args:
            doc_id: Unique document identifier
            file_name: Source file name
            text: Raw normalized document text
    """

    doc_id: str
    file_name: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        """
            Convert document to dictionary

            Returns:
                Dictionary representation
        """

        return asdict(self)

@dataclass(frozen=True)
class Segment:
    """
        Sliding-window segment extracted from a document

        The segmentation stage divides documents into overlapping
        windows to improve semantic similarity retrieval.

        Args:
            segment_id: Unique segment identifier
            doc_id: Parent document identifier
            text: Segment text
            start: Start character offset
            end: End character offset
    """

    segment_id: str
    doc_id: str
    text: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        """
            Convert segment to dictionary

            Returns:
                Dictionary representation
        """

        return asdict(self)

@dataclass(frozen=True)
class EmbeddingVector:
    """
        Dense embedding vector representation of a segment

        Args:
            segment_id: Segment identifier
            vector: Embedding vector
            dimension: Embedding dimension
    """

    segment_id: str
    vector: list[float]
    dimension: int

    def to_dict(self) -> dict[str, Any]:
        """
            Convert embedding to dictionary

            Returns:
                Dictionary representation
        """

        return asdict(self)

@dataclass(frozen=True)
class SimilarityMatch:
    """
        Similarity match between query and labeled segment

        Used during similarity retrieval.

        Args:
            query_segment_id: Segment from unlabeled document
            reference_segment_id: Segment from labeled corpus
            label: Label associated with reference segment
            score: Cosine similarity score
    """

    query_segment_id: str
    reference_segment_id: str
    label: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        """
            Convert match to dictionary

            Returns:
                Dictionary representation
        """

        return asdict(self)

@dataclass(frozen=True)
class DocumentPrediction:
    """
        Multi-label prediction for a clinical document

        The system performs independent binary decisions for each label.

        Args:
            doc_id: Document identifier
            labels: Predicted labels
            scores: Aggregated similarity scores per label
    """

    doc_id: str
    labels: list[str]
    scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """
            Convert prediction to dictionary

            Returns:
                Dictionary representation
        """

        return asdict(self)

## ============================================================
## EXPLAINABILITY MODELS
## ============================================================
class SimilarityEvidence(BaseSchema):
    """
        Evidence explaining a predicted label

        Args:
            label: Predicted label
            segment_id: Segment responsible for the decision
            matched_segment: Labeled reference segment
            similarity_score: Similarity score
    """

    label: str
    segment_id: str
    matched_segment: str
    similarity_score: float

class PredictionResponse(BaseSchema):
    """
        Full prediction response including explainability

        Args:
            doc_id: Document identifier
            labels: Predicted labels
            scores: Label scores
            evidence: Similarity evidence entries
    """

    doc_id: str
    labels: list[str]
    scores: dict[str, float]
    evidence: list[SimilarityEvidence] = Field(default_factory=list)

## ============================================================
## DATASET SCHEMAS
## ============================================================
class DatasetRecord(BaseSchema):
    """
        Generic dataset record

        Args:
            record_id: Record identifier
            payload: Raw payload
            metadata: Additional metadata
    """

    record_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

class DatasetInput(BaseSchema):
    """
        Dataset input container

        Args:
            name: Dataset name
            records: Dataset records
    """

    name: str
    records: list[DatasetRecord]

    @model_validator(mode="after")
    def validate_records(self) -> "DatasetInput":
        """
            Validate dataset content

            Returns:
                Validated dataset input
        """

        if not self.records:
            raise ValueError("records cannot be empty")

        return self

class DatasetOutput(BaseSchema):
    """
        Dataset export structure

        Args:
            name: Dataset name
            row_count: Number of rows
            artifacts: Exported artifacts
    """

    name: str
    row_count: int
    artifacts: list[str] = Field(default_factory=list)

## ============================================================
## PIPELINE MODELS
## ============================================================
class PipelineTask(BaseSchema):
    """
        Individual pipeline task descriptor

        Args:
            task_id: Unique task identifier
            task_type: Type of pipeline stage
            status: Task status
            progress: Completion percentage
            input_payload: Input metadata
            output_payload: Output metadata
    """

    task_id: str
    task_type: TaskTypeName
    status: JobStatusName = "pending"
    progress: float = Field(default=0.0, ge=0, le=100)
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)

class PipelineJob(BaseSchema):
    """
        Pipeline execution container

        Args:
            job_id: Job identifier
            status: Execution status
            tasks: Pipeline tasks
            progress: Global progress
            metadata: Additional metadata
    """

    job_id: str
    status: JobStatusName = "pending"
    tasks: list[PipelineTask] = Field(default_factory=list)
    progress: float = Field(default=0, ge=0, le=100)
    metadata: dict[str, Any] = Field(default_factory=dict)

## ============================================================
## MONITORING MODELS
## ============================================================
class MetricPoint(BaseSchema):
    """
        Monitoring metric entry

        Args:
            name: Metric name
            value: Metric value
            unit: Optional metric unit
            tags: Metric tags
    """

    name: str
    value: float
    unit: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)

class MonitoringResponse(BaseSchema):
    """
        Monitoring metrics response

        Args:
            metrics: Metric entries
            summary: Aggregated metrics
    """

    metrics: list[MetricPoint] = Field(default_factory=list)
    summary: dict[str, float] = Field(default_factory=dict)

class StructuredLogEvent(BaseSchema):
    """
        Structured logging event

        Args:
            level: Log level
            event: Event name
            message: Log message
            context: Context dictionary
            timestamp: Event timestamp
    """

    level: LogLevelName
    event: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

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
    service: str = "doc-classification"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)