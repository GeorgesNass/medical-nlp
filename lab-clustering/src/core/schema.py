'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Pydantic schemas for lab_clustering API: parsing, dataset building, clustering, and export contracts."
'''

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

## ============================================================
## HEALTHCHECK
## ============================================================
class HealthResponse(BaseModel):
    """
        Healthcheck response payload

        Args:
            status: Service status
            service: Service name
            version: Application version
    """

    status: str = Field(default="ok")
    service: str = Field(default="lab_clustering")
    version: str = Field(default="1.0.0")

## ============================================================
## PARSING
## ============================================================
class ParseTxtRequest(BaseModel):
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

    filenames: List[str] = Field(..., min_length=1)
    overwrite: bool = Field(default=False)

class ParseTxtResponse(BaseModel):
    """
        Response payload after TXT parsing

        Args:
            parsed_files: List of structured CSV filenames created
            failed_files: List of filenames that failed parsing
            message: Human-readable summary message
    """

    parsed_files: List[str] = Field(default_factory=list)
    failed_files: List[str] = Field(default_factory=list)
    message: str = Field(default="")

## ============================================================
## DATASET BUILDING
## ============================================================
DatasetFormat = Literal["long", "wide"]

class BuildDatasetRequest(BaseModel):
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

    structured_csv_files: Optional[List[str]] = Field(default=None)
    dataset_format: DatasetFormat = Field(default="wide")
    overwrite: bool = Field(default=False)

class BuildDatasetResponse(BaseModel):
    """
        Response payload after dataset building

        Args:
            dataset_path: Output dataset path (relative)
            n_rows: Number of rows in produced dataset
            n_cols: Number of columns in produced dataset
            message: Human-readable summary message
    """

    dataset_path: str
    n_rows: int
    n_cols: int
    message: str = Field(default="")

## ============================================================
## CLUSTERING
## ============================================================
ClusteringAlgorithm = Literal[
    "kmeans",
    "agglomerative",
    "dbscan",
    "birch",
    "affinity_propagation",
    "kmodes",
]

class ClusteringParams(BaseModel):
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
    params: Dict[str, Any] = Field(default_factory=dict)

class RunClusteringRequest(BaseModel):
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

    dataset_path: str
    clustering: ClusteringParams
    preprocess: Dict[str, Any] = Field(default_factory=dict)
    overwrite: bool = Field(default=False)

class RunClusteringResponse(BaseModel):
    """
        Response payload after clustering run

        Args:
            run_id: Unique run identifier
            mlflow_run_id: MLflow run id if tracked
            n_clusters: Number of clusters found
            metrics: Unsupervised metrics dictionary
            exports: List of exported artifact paths (relative)
            message: Human-readable summary message
    """

    run_id: str
    mlflow_run_id: Optional[str] = None
    n_clusters: int
    metrics: Dict[str, Any] = Field(default_factory=dict)
    exports: List[str] = Field(default_factory=list)
    message: str = Field(default="")

## ============================================================
## EXPORTS
## ============================================================
class ExportRequest(BaseModel):
    """
        Request payload for exporting reports and clustering artifacts

        Args:
            run_id: Run identifier
            export_eda: Whether to export EDA artifacts
            export_clustering: Whether to export clustering artifacts
    """

    run_id: str
    export_eda: bool = Field(default=True)
    export_clustering: bool = Field(default=True)

class ExportResponse(BaseModel):
    """
        Response payload after export

        Args:
            exports: List of exported artifact paths (relative)
            message: Human-readable summary message
    """

    exports: List[str] = Field(default_factory=list)
    message: str = Field(default="")