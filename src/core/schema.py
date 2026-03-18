'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Pydantic schemas for lab_clustering API: parsing, dataset building, clustering, and export contracts."
'''

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover
    BaseSettings = BaseModel  # type: ignore[misc, assignment]
    SettingsConfigDict = dict  # type: ignore[misc, assignment]

## ============================================================
## COMMON TYPES AND PATTERNS
## ============================================================
DatasetFormat = Literal["long", "wide"]
ClusteringAlgorithm = Literal[
    "kmeans",
    "agglomerative",
    "dbscan",
    "birch",
    "affinity_propagation",
    "kmodes",
]
JobStatusName = Literal["pending", "running", "success", "failed", "cancelled"]

SAFE_FILE_PATTERN = re.compile(r"^[a-zA-Z0-9._/\-]+$")
TXT_FILE_PATTERN = re.compile(r"^[a-zA-Z0-9._/\-]+\.txt$")
CSV_FILE_PATTERN = re.compile(r"^[a-zA-Z0-9._/\-]+\.csv$")
DATASET_FILE_PATTERN = re.compile(
    r"^[a-zA-Z0-9._/\-]+\.(csv|parquet|pkl|pickle)$"
)

## ============================================================
## BASE SCHEMAS
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

        ## Import pandas lazily to avoid a hard dependency at import time
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
## SETTINGS AND CONFIG SCHEMAS
## ============================================================
@dataclass(frozen=True)
class DatasetRuntimeConfig:
    """
        Typed runtime configuration for dataset building

        Args:
            dataset_format: Output dataset format
            overwrite: Whether outputs can be overwritten
            default_export_dir: Default export directory
    """

    dataset_format: str
    overwrite: bool
    default_export_dir: str

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

@dataclass(frozen=True)
class ClusteringRuntimeConfig:
    """
        Typed runtime configuration for clustering runs

        Args:
            algorithm: Clustering algorithm name
            max_workers: Maximum number of workers
            track_with_mlflow: Whether MLflow tracking is enabled
            random_state: Optional random seed
    """

    algorithm: str
    max_workers: int
    track_with_mlflow: bool
    random_state: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

class AppSettings(BaseSettings):
    """
        Settings model for lab_clustering

        Args:
            app_name: Application name
            environment: Runtime environment
            default_dataset_format: Default dataset format
            default_algorithm: Default clustering algorithm
            default_max_workers: Default worker count
            enable_mlflow: Whether MLflow tracking is enabled
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        env_prefix="LAB_CLUSTERING_",
        case_sensitive=False,
    )

    app_name: str = "lab_clustering"
    environment: str = "dev"
    default_dataset_format: DatasetFormat = "wide"
    default_algorithm: ClusteringAlgorithm = "kmeans"
    default_max_workers: int = Field(default=1, ge=1, le=512)
    enable_mlflow: bool = True

class PipelineConfig(BaseSchema):
    """
        Pipeline execution configuration schema

        Args:
            job_name: Pipeline job name
            batch_size: Batch size
            max_workers: Number of workers
            retry_count: Retry count
            overwrite: Whether outputs can be overwritten
    """

    job_name: str = Field(default="lab-clustering-job", min_length=1)
    batch_size: int = Field(default=1, ge=1, le=10000)
    max_workers: int = Field(default=1, ge=1, le=512)
    retry_count: int = Field(default=0, ge=0, le=20)
    overwrite: bool = False

## ============================================================
## COMMON OPERATIONAL SCHEMAS
## ============================================================
class HealthResponse(BaseSchema):
    """
        Healthcheck response payload

        Args:
            status: Service status
            service: Service name
            version: Application version
    """

    status: str = Field(default="ok", min_length=1)
    service: str = Field(default="lab_clustering", min_length=1)
    version: str = Field(default="1.0.0", min_length=1)

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

    error: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    origin: str = Field(default="unknown", min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str = Field(default="n/a", min_length=1)

class StatusResponse(BaseSchema):
    """
        Generic status response schema

        Args:
            status: Current status
            message: Optional message
            progress: Optional progress between 0 and 100
            metadata: Optional metadata
    """

    status: str = Field(..., min_length=1)
    message: str = Field(default="")
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
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    event: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    logger_name: str = Field(default="lab_clustering", min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)

class QueueEvent(BaseSchema):
    """
        Message queue or event bus schema

        Args:
            event_id: Unique event identifier
            event_type: Event type
            source: Event source
            payload: Event payload
    """

    event_id: str = Field(..., min_length=1)
    event_type: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id", "event_type", "source")
    @classmethod
    def validate_safe_values(cls, value: str) -> str:
        """
            Validate safe identifier-like strings

            Args:
                value: Candidate identifier string

            Returns:
                The validated identifier string

            Raises:
                ValueError: If the value contains unsupported characters
        """

        ## Ensure identifiers remain API and filesystem friendly
        if not SAFE_FILE_PATTERN.match(value):
            raise ValueError("value contains unsupported characters")
        return value

class MetricPoint(BaseSchema):
    """
        Monitoring metric point schema

        Args:
            name: Metric name
            value: Metric value
            unit: Optional metric unit
            tags: Optional metric tags
    """

    name: str = Field(..., min_length=1)
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
class DatasetRecord(BaseSchema):
    """
        Generic dataset record schema

        Args:
            record_id: Record identifier
            source_file: Source file name
            text: Main text field
            label: Optional label
            metadata: Optional metadata
    """

    record_id: str = Field(..., min_length=1)
    source_file: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_file")
    @classmethod
    def validate_source_file(cls, value: str) -> str:
        """
            Validate source filename format

            Args:
                value: Candidate source filename

            Returns:
                The validated filename

            Raises:
                ValueError: If the filename format is invalid
        """

        ## Accept txt and csv-like source references
        if not SAFE_FILE_PATTERN.match(value):
            raise ValueError("source_file contains unsupported characters")
        return value

class DatasetInput(BaseSchema):
    """
        Dataset input schema

        Args:
            name: Dataset name
            records: Dataset records
            dataset_format: Dataset format
    """

    name: str = Field(..., min_length=1)
    records: list[DatasetRecord] = Field(default_factory=list)
    dataset_format: DatasetFormat = "wide"

    @field_validator("records")
    @classmethod
    def validate_non_empty_records(
        cls, value: list[DatasetRecord]
    ) -> list[DatasetRecord]:
        """
            Validate that the dataset contains at least one record

            Args:
                value: Dataset records

            Returns:
                The validated records list

            Raises:
                ValueError: If the records list is empty
        """

        ## Prevent empty dataset payloads
        if not value:
            raise ValueError("records must contain at least one item")
        return value

class DatasetOutput(BaseSchema):
    """
        Dataset output schema

        Args:
            name: Dataset name
            dataset_path: Output dataset path
            row_count: Number of rows
            column_count: Number of columns
            artifacts: Generated artifacts list
    """

    name: str = Field(..., min_length=1)
    dataset_path: str = Field(..., min_length=1)
    row_count: int = Field(..., ge=0)
    column_count: int = Field(..., ge=0)
    artifacts: list[str] = Field(default_factory=list)

    @field_validator("dataset_path")
    @classmethod
    def validate_dataset_path(cls, value: str) -> str:
        """
            Validate dataset output path

            Args:
                value: Candidate dataset path

            Returns:
                The validated dataset path

            Raises:
                ValueError: If the dataset path format is invalid
        """

        ## Restrict exported dataset file extensions to supported formats
        if not DATASET_FILE_PATTERN.match(value):
            raise ValueError("dataset_path must end with csv, parquet, pkl or pickle")
        return value

class PipelineTask(BaseSchema):
    """
        Pipeline task schema

        Args:
            task_id: Task identifier
            task_name: Task name
            status: Task status
            progress: Task progress percentage
            input_payload: Task input payload
            output_payload: Task output payload
    """

    task_id: str = Field(..., min_length=1)
    task_name: str = Field(..., min_length=1)
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

    job_id: str = Field(..., min_length=1)
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

            Raises:
                ValueError: If job progress is below the minimum task progress
        """

        ## Keep parent progress coherent with child task progress
        if self.tasks and self.progress < min(task.progress for task in self.tasks):
            raise ValueError("job progress cannot be below the minimum task progress")
        return self

## ============================================================
## PARSING
## ============================================================
class ParseTxtRequest(BaseSchema):
    """
        Request payload for parsing TXT files into structured CSV

        High-level workflow:
            1) Select one or multiple .txt files under data/raw
            2) Parse and structure each file
            3) Export one CSV per source file to data/interim/lab_structured_csv

        Args:
            filenames: List of .txt filenames located in data/raw
            overwrite: Overwrite existing structured CSV if present
    """

    filenames: list[str] = Field(..., min_length=1)
    overwrite: bool = False

    @field_validator("filenames")
    @classmethod
    def validate_txt_filenames(cls, value: list[str]) -> list[str]:
        """
            Validate TXT filenames list

            Args:
                value: Candidate TXT filenames

            Returns:
                The validated TXT filenames list

            Raises:
                ValueError: If the list is empty or contains invalid filenames
        """

        ## Enforce TXT input names only and remove duplicates while preserving order
        cleaned_values: list[str] = []
        for filename in value:
            if not TXT_FILE_PATTERN.match(filename):
                raise ValueError("all filenames must be valid .txt files")
            if filename not in cleaned_values:
                cleaned_values.append(filename)

        if not cleaned_values:
            raise ValueError("filenames must contain at least one item")

        return cleaned_values

class ParseTxtResponse(WarningMixin):
    """
        Response payload after TXT parsing

        Args:
            parsed_files: List of structured CSV filenames created
            failed_files: List of filenames that failed parsing
            message: Human-readable summary message
    """

    parsed_files: list[str] = Field(default_factory=list)
    failed_files: list[str] = Field(default_factory=list)
    message: str = Field(default="")

    @field_validator("parsed_files", "failed_files")
    @classmethod
    def validate_csv_outputs(cls, value: list[str]) -> list[str]:
        """
            Validate generated CSV filenames

            Args:
                value: Candidate generated filenames

            Returns:
                The validated filenames list

            Raises:
                ValueError: If a filename is not a valid CSV file
        """

        ## Restrict parsing outputs to CSV files
        for filename in value:
            if not CSV_FILE_PATTERN.match(filename):
                raise ValueError("generated filenames must be valid .csv files")
        return value

## ============================================================
## DATASET BUILDING
## ============================================================
class BuildDatasetRequest(BaseSchema):
    """
        Request payload for building clustering dataset from structured CSV

        High-level workflow:
            1) Load all (or selected) structured CSVs from data/interim/lab_structured_csv
            2) Build dataset in long and/or wide format
            3) Export to data/interim/datasets (parquet/csv)

        Args:
            structured_csv_files: Optional subset of structured CSV filenames
            dataset_format: "long" or "wide"
            overwrite: Overwrite existing dataset outputs
    """

    structured_csv_files: list[str] | None = None
    dataset_format: DatasetFormat = "wide"
    overwrite: bool = False

    @field_validator("structured_csv_files")
    @classmethod
    def validate_structured_csv_files(
        cls, value: list[str] | None
    ) -> list[str] | None:
        """
            Validate optional structured CSV filenames list

            Args:
                value: Candidate structured CSV filenames

            Returns:
                The validated list or None

            Raises:
                ValueError: If one filename is not a valid CSV file
        """

        ## Accept omitted values, otherwise validate every filename
        if value is None:
            return value

        cleaned_values: list[str] = []
        for filename in value:
            if not CSV_FILE_PATTERN.match(filename):
                raise ValueError(
                    "structured_csv_files must contain valid .csv filenames"
                )
            if filename not in cleaned_values:
                cleaned_values.append(filename)

        return cleaned_values

class BuildDatasetResponse(WarningMixin):
    """
        Response payload after dataset building

        Args:
            dataset_path: Output dataset path
            n_rows: Number of rows in produced dataset
            n_cols: Number of columns in produced dataset
            message: Human-readable summary message
    """

    dataset_path: str = Field(..., min_length=1)
    n_rows: int = Field(..., ge=0)
    n_cols: int = Field(..., ge=0)
    message: str = Field(default="")

    @field_validator("dataset_path")
    @classmethod
    def validate_dataset_output_path(cls, value: str) -> str:
        """
            Validate dataset export path

            Args:
                value: Candidate dataset path

            Returns:
                The validated dataset path

            Raises:
                ValueError: If the dataset path format is invalid
        """

        ## Restrict dataset outputs to supported local file formats
        if not DATASET_FILE_PATTERN.match(value):
            raise ValueError("dataset_path must end with csv, parquet, pkl or pickle")
        return value

## ============================================================
## CLUSTERING
## ============================================================
class ClusteringParams(BaseSchema):
    """
        Clustering hyperparameters payload

        Design choice:
            - Keep a generic dict for algorithm-specific params
            - Validate basic known params optionally in clustering module

        Args:
            algorithm: Clustering algorithm name
            params: Algorithm-specific parameters dictionary
    """

    algorithm: ClusteringAlgorithm
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_known_numeric_params(self) -> "ClusteringParams":
        """
            Validate known numeric clustering parameters when present

            Returns:
                The validated clustering parameters

            Raises:
                ValueError: If one known parameter is out of range
        """

        ## Apply soft validation only to known common parameters
        n_clusters = self.params.get("n_clusters")
        if n_clusters is not None and (
            not isinstance(n_clusters, int) or n_clusters <= 0
        ):
            raise ValueError("params.n_clusters must be a positive integer")

        eps_value = self.params.get("eps")
        if eps_value is not None and (
            not isinstance(eps_value, (int, float)) or eps_value <= 0
        ):
            raise ValueError("params.eps must be a positive number")

        min_samples = self.params.get("min_samples")
        if min_samples is not None and (
            not isinstance(min_samples, int) or min_samples <= 0
        ):
            raise ValueError("params.min_samples must be a positive integer")

        return self

class RunClusteringRequest(BaseSchema):
    """
        Request payload for running clustering on a dataset

        High-level workflow:
            1) Load dataset from data/interim/datasets or data/processed/features
            2) Preprocess: impute, scale, optional reduction
            3) Fit clustering model
            4) Evaluate unsupervised metrics
            5) Track everything with MLflow and export artifacts

        Args:
            dataset_path: Relative path to dataset file
            clustering: Clustering algorithm and params
            preprocess: Preprocessing options dictionary
            overwrite: Overwrite exports if present
    """

    dataset_path: str = Field(..., min_length=1)
    clustering: ClusteringParams
    preprocess: dict[str, Any] = Field(default_factory=dict)
    overwrite: bool = False

    @field_validator("dataset_path")
    @classmethod
    def validate_input_dataset_path(cls, value: str) -> str:
        """
            Validate input dataset path

            Args:
                value: Candidate input dataset path

            Returns:
                The validated input dataset path

            Raises:
                ValueError: If the dataset path format is invalid
        """

        ## Restrict clustering inputs to supported dataset formats
        if not DATASET_FILE_PATTERN.match(value):
            raise ValueError("dataset_path must end with csv, parquet, pkl or pickle")
        return value

    @model_validator(mode="after")
    def validate_preprocess_payload(self) -> "RunClusteringRequest":
        """
            Validate known preprocessing parameters when present

            Returns:
                The validated clustering request

            Raises:
                ValueError: If one preprocess parameter is inconsistent
        """

        ## Validate optional preprocessing numeric parameters
        n_components = self.preprocess.get("n_components")
        if n_components is not None and (
            not isinstance(n_components, int) or n_components <= 0
        ):
            raise ValueError("preprocess.n_components must be a positive integer")

        test_size = self.preprocess.get("test_size")
        if test_size is not None and (
            not isinstance(test_size, (int, float))
            or float(test_size) <= 0
            or float(test_size) >= 1
        ):
            raise ValueError("preprocess.test_size must be between 0 and 1")

        return self

class RunClusteringResponse(WarningMixin):
    """
        Response payload after clustering run

        Args:
            run_id: Unique run identifier
            mlflow_run_id: MLflow run id if tracked
            n_clusters: Number of clusters found
            metrics: Unsupervised metrics dictionary
            exports: List of exported artifact paths
            message: Human-readable summary message
    """

    run_id: str = Field(..., min_length=1)
    mlflow_run_id: str | None = None
    n_clusters: int = Field(..., ge=0)
    metrics: dict[str, Any] = Field(default_factory=dict)
    exports: list[str] = Field(default_factory=list)
    message: str = Field(default="")

    @field_validator("exports")
    @classmethod
    def validate_exports(cls, value: list[str]) -> list[str]:
        """
            Validate export artifact paths

            Args:
                value: Candidate artifact paths

            Returns:
                The validated artifact paths list

            Raises:
                ValueError: If one export path contains unsupported characters
        """

        ## Ensure exported paths remain safe and relative-like
        for export_path in value:
            if not SAFE_FILE_PATTERN.match(export_path):
                raise ValueError("exports contain unsupported path characters")
        return value

## ============================================================
## EXPORTS
## ============================================================
class ExportRequest(BaseSchema):
    """
        Request payload for exporting reports and clustering artifacts

        Args:
            run_id: Run identifier
            export_eda: Whether to export EDA artifacts
            export_clustering: Whether to export clustering artifacts
    """

    run_id: str = Field(..., min_length=1)
    export_eda: bool = True
    export_clustering: bool = True

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        """
            Validate run identifier format

            Args:
                value: Candidate run identifier

            Returns:
                The validated run identifier

            Raises:
                ValueError: If the run identifier contains unsupported characters
        """

        ## Keep run identifiers safe for file paths and API payloads
        if not SAFE_FILE_PATTERN.match(value):
            raise ValueError("run_id contains unsupported characters")
        return value

class ExportResponse(WarningMixin):
    """
        Response payload after export

        Args:
            exports: List of exported artifact paths
            message: Human-readable summary message
    """

    exports: list[str] = Field(default_factory=list)
    message: str = Field(default="")

    @field_validator("exports")
    @classmethod
    def validate_export_paths(cls, value: list[str]) -> list[str]:
        """
            Validate export paths list

            Args:
                value: Candidate export paths

            Returns:
                The validated export paths list

            Raises:
                ValueError: If one path contains unsupported characters
        """

        ## Ensure export paths are safe to log and persist
        for export_path in value:
            if not SAFE_FILE_PATTERN.match(export_path):
                raise ValueError("exports contain unsupported path characters")
        return value