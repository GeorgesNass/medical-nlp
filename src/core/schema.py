'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Canonical data schema for Clinical NER: dataclasses, Pydantic contracts, and strict business validation."
'''

from __future__ import annotations

## STANDARD IMPORTS
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover
    BaseSettings = BaseModel  # type: ignore[misc, assignment]
    SettingsConfigDict = dict  # type: ignore[misc, assignment]

## DOMAIN IMPORTS
from src.core.entities import (
    DictionarySource,
    EntityLabel,
    EntityProvenance,
    NegationStatus,
    ensure_unique_entity_ids,
    normalize_dictionary_source,
    normalize_label,
    validate_temporality,
)
from src.core.errors import DataError
from src.utils.logging_utils import get_logger
from src.utils.utils import (
    ensure_str,
    ensure_str_or_none,
    is_valid_entity_id,
    is_valid_patient_id,
    is_valid_record_id,
    json_dumps,
    load_list_of_dicts_from_json,
    parse_date_to_iso,
)

## ============================================================
## LOGGER AND COMMON TYPES
## ============================================================
logger = get_logger(name="clinical_ner.schema")

JobStatusName = Literal["pending", "running", "success", "failed", "cancelled"]
TaskTypeName = Literal[
    "parse",
    "normalize",
    "predict",
    "postprocess",
    "export",
    "evaluation",
]
LogLevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

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
## SETTINGS AND RUNTIME CONFIG
## ============================================================
@dataclass(frozen=True)
class NerRuntimeConfig:
    """
        Typed runtime configuration for Clinical NER

        Args:
            max_entities_per_record: Safety cap for entities per record
            validate_spans: Whether spans are validated against text
            default_dictionary: Default dictionary source
            default_source: Default provenance source
    """

    max_entities_per_record: int
    validate_spans: bool
    default_dictionary: str
    default_source: str

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

class AppSettings(BaseSettings):
    """
        Settings model for clinical-ner

        Args:
            app_name: Application name
            environment: Runtime environment
            max_entities_per_record: Safety cap for entities per record
            validate_spans: Whether spans are validated
            default_dictionary: Default dictionary source
            default_source: Default provenance source
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        env_prefix="CLINICAL_NER_",
        case_sensitive=False,
    )

    app_name: str = "clinical-ner"
    environment: str = "dev"
    max_entities_per_record: int = Field(default=300, ge=1, le=10000)
    validate_spans: bool = True
    default_dictionary: str = DictionarySource.MESH.value
    default_source: str = EntityProvenance.MANUAL.value

## ============================================================
## DATACLASS DOMAIN MODELS
## ============================================================
@dataclass(slots=True)
class Entity:
    """
        Canonical entity annotation used across the Clinical NER pipeline

        Attributes:
            id: Unique entity identifier inside a record
            text: Surface form extracted from the document
            start: Start character offset in the document
            end: End character offset in the document
            label: Canonical entity label
            concept_id: Ontology or dictionary identifier
            concept_name: Canonical concept name
            dictionary: Dictionary or ontology source
            negation: Negation status of the mention
            temporality: Temporal status
            confidence: Optional confidence score in [0, 1]
            source: Provenance of the entity annotation
            meta: Free metadata container
            normalized_text: Normalized entity text (feature engineering)
            token_count: Number of tokens in entity text
            char_length: Character length of entity text            
    """

    id: str
    text: str
    start: int
    end: int
    label: EntityLabel
    concept_id: str | None = None
    concept_name: str | None = None
    dictionary: DictionarySource = DictionarySource.MESH
    negation: NegationStatus = NegationStatus.UNKNOWN
    temporality: str | None = None
    confidence: float | None = None
    source: EntityProvenance = EntityProvenance.MANUAL
    meta: dict[str, Any] = field(default_factory=dict)
    normalized_text: str | None = None
    token_count: int | None = None
    char_length: int | None = None

    def validate(self) -> None:
        """
            Validate entity internal consistency

            Raises:
                DataError: If any validation rule is violated
        """

        if not is_valid_entity_id(self.id):
            raise DataError(f"Invalid entity id format: {self.id}")

        if not ensure_str(self.text).strip():
            raise DataError(f"Empty entity text for entity id: {self.id}")

        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise DataError(f"Non-integer span for entity id: {self.id}")

        if self.start < 0 or self.end <= self.start:
            raise DataError(
                f"Invalid span for entity id {self.id}: "
                f"start={self.start}, end={self.end}"
            )

        if self.temporality is not None:
            self.temporality = ensure_str(self.temporality).strip().lower()

        if not validate_temporality(self.label, self.temporality):
            raise DataError(
                f"Invalid temporality '{self.temporality}' "
                f"for label '{self.label.value}' (entity id {self.id})"
            )

        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)):
                raise DataError(f"Non-numeric confidence for entity id {self.id}")
            if not 0.0 <= float(self.confidence) <= 1.0:
                raise DataError(f"Confidence out of range for entity id {self.id}")

        if not isinstance(self.meta, dict):
            raise DataError(f"Invalid meta type for entity id {self.id}")

        if self.source in (
            EntityProvenance.MESH_AUTO,
            EntityProvenance.DICT_AUTO,
        ):
            if not ensure_str_or_none(self.concept_id):
                raise DataError(f"Missing concept_id for auto entity id {self.id}")
            if not ensure_str_or_none(self.concept_name):
                raise DataError(f"Missing concept_name for auto entity id {self.id}")

    def to_dict(self) -> dict[str, Any]:
        """
            Convert entity to a JSON-serializable dictionary

            Returns:
                Dictionary representation of the entity
        """

        payload = asdict(self)
        payload["label"] = self.label.value
        payload["dictionary"] = self.dictionary.value
        payload["negation"] = self.negation.value
        payload["source"] = self.source.value

        ## Feature engineering fields
        if self.normalized_text is not None:
            payload["normalized_text"] = self.normalized_text
        if self.token_count is not None:
            payload["token_count"] = self.token_count
        if self.char_length is not None:
            payload["char_length"] = self.char_length
            
        return payload

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Entity":
        """
            Build an Entity instance from a raw dictionary

            Args:
                data: Raw entity dictionary

            Returns:
                Entity instance
        """

        label = normalize_label(ensure_str(data.get("label")))
        dictionary = normalize_dictionary_source(
            ensure_str(data.get("dictionary", DictionarySource.MESH.value))
        )
        neg_raw = ensure_str(data.get("negation", NegationStatus.UNKNOWN.value)).lower()
        negation = next(
            (v for v in NegationStatus if v.value == neg_raw),
            NegationStatus.UNKNOWN,
        )
        src_raw = ensure_str(data.get("source", EntityProvenance.MANUAL.value)).lower()
        source = next(
            (v for v in EntityProvenance if v.value == src_raw),
            EntityProvenance.MANUAL,
        )

        confidence = None if data.get("confidence") is None else float(data["confidence"])
        meta_raw = data.get("meta") or {}
        if not isinstance(meta_raw, dict):
            raise DataError("Invalid meta type (expected dict)")

        ent = Entity(
            id=ensure_str(data.get("id")),
            text=ensure_str(data.get("text")),
            start=int(data.get("start", -1)),
            end=int(data.get("end", -1)),
            label=label,
            concept_id=data.get("concept_id"),
            concept_name=data.get("concept_name"),
            dictionary=dictionary,
            negation=negation,
            temporality=data.get("temporality"),
            confidence=confidence,
            source=source,
            meta=meta_raw,
            normalized_text=data.get("normalized_text"),
            token_count=data.get("token_count"),
            char_length=data.get("char_length"),
        )
        ent.validate()
        return ent

@dataclass(slots=True)
class Record:
    """
        Canonical record container for a single clinical document

        Attributes:
            record_id: Unique record identifier
            patient_id: Patient identifier
            name_document: Document filename or logical name
            type_document: Document type
            text: Raw clinical text
            date_document: ISO date string or None
            entities: List of extracted or annotated entities
    """

    record_id: str
    patient_id: str
    name_document: str
    type_document: str
    text: str
    date_document: str | None = None
    entities: list[Entity] = field(default_factory=list)

    def validate(self, validate_spans: bool = True, max_entities: int = 300) -> None:
        """
            Validate record internal consistency

            Args:
                validate_spans: Whether entity spans are validated against text
                max_entities: Safety limit for number of entities

            Raises:
                DataError: If any validation rule is violated
        """

        if not is_valid_record_id(self.record_id):
            raise DataError(f"Invalid record_id format: {self.record_id}")
        if not is_valid_patient_id(self.patient_id):
            raise DataError(f"Invalid patient_id format: {self.patient_id}")
        if not ensure_str(self.name_document).strip():
            raise DataError(f"Empty name_document for record_id: {self.record_id}")
        if not ensure_str(self.type_document).strip():
            raise DataError(f"Empty type_document for record_id: {self.record_id}")
        if not ensure_str(self.text).strip():
            raise DataError(f"Empty text for record_id: {self.record_id}")

        if self.date_document is not None:
            self.date_document = parse_date_to_iso(self.date_document)

        if len(self.entities) > max_entities:
            raise DataError(
                f"Too many entities ({len(self.entities)}) for record_id "
                f"{self.record_id} (max {max_entities})"
            )

        for ent in self.entities:
            ent.validate()

        if not ensure_unique_entity_ids([e.id for e in self.entities]):
            raise DataError(f"Duplicate entity ids for record_id: {self.record_id}")

        if validate_spans:
            text_len = len(self.text)
            for ent in self.entities:
                if ent.end > text_len:
                    raise DataError(
                        f"Entity span out of bounds for record_id {self.record_id} "
                        f"(entity id {ent.id}, end={ent.end}, text_len={text_len})"
                    )
                if not self.text[ent.start:ent.end]:
                    raise DataError(
                        f"Empty span slice for record_id {self.record_id} "
                        f"(entity id {ent.id})"
                    )

    def to_dict(self) -> dict[str, Any]:
        """
            Convert record to a JSON-serializable dictionary

            Returns:
                Dictionary representation of the record
        """

        payload = asdict(self)
        payload["entities"] = [e.to_dict() for e in self.entities]
        return payload

    def to_csv_row(self) -> dict[str, Any]:
        """
            Convert record to a flat CSV row

            Returns:
                Flat CSV row with entities stored as JSON string
        """

        return {
            "text": self.text,
            "name_document": self.name_document,
            "type_document": self.type_document,
            "patient_id": self.patient_id,
            "record_id": self.record_id,
            "date_document": self.date_document,
            "entities": json_dumps([e.to_dict() for e in self.entities]),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Record":
        """
            Build a Record instance from a raw dictionary

            Args:
                data: Raw record dictionary

            Returns:
                Record instance
        """

        entities_raw = data.get("entities", [])
        if isinstance(entities_raw, str):
            entities_list = load_list_of_dicts_from_json(entities_raw)
        elif isinstance(entities_raw, list):
            entities_list = entities_raw
        else:
            raise DataError("Invalid entities format (expected JSON string or list)")

        rec = Record(
            record_id=ensure_str(data.get("record_id")),
            patient_id=ensure_str(data.get("patient_id")),
            name_document=ensure_str(data.get("name_document")),
            type_document=ensure_str(data.get("type_document")),
            text=ensure_str(data.get("text")),
            date_document=data.get("date_document"),
            entities=[Entity.from_dict(e) for e in entities_list],
        )
        rec.validate(validate_spans=True)
        return rec

## ============================================================
## PYDANTIC DOMAIN CONTRACTS
## ============================================================
class EntityPayload(BaseSchema):
    """
        Pydantic entity payload

        Args:
            id: Unique entity identifier
            text: Surface form extracted from the document
            start: Start character offset
            end: End character offset
            label: Canonical entity label
            concept_id: Ontology or dictionary identifier
            concept_name: Canonical concept name
            dictionary: Dictionary source
            negation: Negation status
            temporality: Temporal status
            confidence: Confidence score
            source: Annotation provenance
            meta: Free metadata
            normalized_text: Normalized entity text (feature engineering)
            token_count: Number of tokens in entity text
            char_length: Character length of entity text            
    """

    id: str
    text: str
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=1)
    label: str
    concept_id: str | None = None
    concept_name: str | None = None
    dictionary: str = DictionarySource.MESH.value
    negation: str = NegationStatus.UNKNOWN.value
    temporality: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str = EntityProvenance.MANUAL.value
    meta: dict[str, Any] = Field(default_factory=dict)
    normalized_text: str | None = None
    token_count: int | None = None
    char_length: int | None = None

    @model_validator(mode="after")
    def validate_entity_payload(self) -> "EntityPayload":
        """
            Validate entity payload business rules

            Returns:
                The validated entity payload
        """

        Entity.from_dict(self.model_dump())
        return self

class RecordPayload(BaseSchema):
    """
        Pydantic record payload

        Args:
            record_id: Record identifier
            patient_id: Patient identifier
            name_document: Document name
            type_document: Document type
            text: Clinical text
            date_document: Optional ISO date
            entities: Extracted or annotated entities
    """

    record_id: str
    patient_id: str
    name_document: str
    type_document: str
    text: str
    date_document: str | None = None
    entities: list[EntityPayload] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_record_payload(self) -> "RecordPayload":
        """
            Validate record payload business rules

            Returns:
                The validated record payload
        """

        Record.from_dict(self.model_dump())
        return self

## ============================================================
## REQUEST AND RESPONSE SCHEMAS
## ============================================================
class PredictRequest(BaseSchema):
    """
        Prediction request schema

        Args:
            record: Input clinical record
            return_normalized: Whether normalized entities are returned
    """

    record: RecordPayload
    return_normalized: bool = True

class PredictResponse(WarningMixin):
    """
        Prediction response schema

        Args:
            record_id: Record identifier
            patient_id: Patient identifier
            entities: Predicted entities
            entity_count: Number of predicted entities
    """

    record_id: str
    patient_id: str
    entities: list[EntityPayload] = Field(default_factory=list)
    entity_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_entity_count(self) -> "PredictResponse":
        """
            Validate response entity count consistency

            Returns:
                The validated prediction response
        """

        if self.entity_count != len(self.entities):
            raise ValueError("entity_count must match len(entities)")
        return self

class BatchPredictRequest(BaseSchema):
    """
        Batch prediction request schema

        Args:
            records: Input clinical records
    """

    records: list[RecordPayload]

    @field_validator("records")
    @classmethod
    def validate_records(cls, value: list[RecordPayload]) -> list[RecordPayload]:
        """
            Validate input record list

            Args:
                value: Candidate input records

            Returns:
                The validated records list
        """

        if not value:
            raise ValueError("records must contain at least one item")
        return value

class BatchPredictResponse(WarningMixin):
    """
        Batch prediction response schema

        Args:
            results: Prediction results
            batch_size: Number of processed records
    """

    results: list[PredictResponse]
    batch_size: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_batch_size(self) -> "BatchPredictResponse":
        """
            Validate batch size consistency

            Returns:
                The validated batch prediction response
        """

        if self.batch_size != len(self.results):
            raise ValueError("batch_size must match len(results)")
        return self

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
    service: str = "clinical-ner"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ErrorResponse(BaseSchema):
    """
        Standard API error response

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
            progress: Optional progress between 0 and 100
            metadata: Optional metadata
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
    logger_name: str = "clinical_ner.schema"
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class MetricPoint(BaseSchema):
    """
        Monitoring metric point schema

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
## DATASET AND PIPELINE SCHEMAS
## ============================================================
class DatasetInput(BaseSchema):
    """
        Dataset input schema

        Args:
            name: Dataset name
            records: Input records
    """

    name: str
    records: list[RecordPayload]

    @field_validator("records")
    @classmethod
    def validate_dataset_records(
        cls, value: list[RecordPayload]
    ) -> list[RecordPayload]:
        """
            Validate dataset records list

            Args:
                value: Candidate dataset records

            Returns:
                The validated dataset records list
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
    def validate_job_progress(self) -> "PipelineJob":
        """
            Validate progress consistency between the job and its tasks

            Returns:
                The validated pipeline job
        """

        if self.tasks and self.progress < min(task.progress for task in self.tasks):
            raise ValueError("job progress cannot be below the minimum task progress")
        return self

## ============================================================
## EVALUATION SCHEMAS
## ============================================================
class EvaluationItem(BaseSchema):
    """
        Entity evaluation item schema

        Args:
            label: Entity label
            precision: Precision score
            recall: Recall score
            f1_score: F1 score
            support: Support count
    """

    label: str
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    support: int = Field(..., ge=0)

class EvaluationResponse(WarningMixin):
    """
        Evaluation response schema

        Args:
            items: Per-label evaluation items
            micro_metrics: Micro-averaged metrics
            macro_metrics: Macro-averaged metrics
    """

    items: list[EvaluationItem] = Field(default_factory=list)
    micro_metrics: dict[str, float] = Field(default_factory=dict)
    macro_metrics: dict[str, float] = Field(default_factory=dict)