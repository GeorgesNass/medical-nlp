'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "FastAPI service layer exposing parsing, dataset building, clustering and export endpoints."
'''

from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI

from src.core.config import AppConfig, build_config
from src.core.errors import (
    ClusteringError,
    DatasetBuildError,
    ParsingError,
)
from src.core.schema import (
    BuildDatasetRequest,
    BuildDatasetResponse,
    ExportRequest,
    ExportResponse,
    HealthResponse,
    ParseTxtRequest,
    ParseTxtResponse,
    RunClusteringRequest,
    RunClusteringResponse,
)
from src.pipelines import (
    build_dataset_pipeline,
    export_pipeline,
    parse_txt_pipeline,
    run_clustering_pipeline,
)
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger(__name__)

## ============================================================
## DEPENDENCIES
## ============================================================
def get_config() -> AppConfig:
    """
        Dependency that builds and returns application configuration

        Returns:
            AppConfig instance
    """

    return build_config()

## ============================================================
## ROUTER
## ============================================================
router = APIRouter()
app = FastAPI(title="Lab Clustering API")
app.include_router(router)

## ============================================================
## HEALTHCHECK
## ============================================================
@router.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    """
        Basic healthcheck endpoint

        Returns:
            HealthResponse payload
    """

    logger.info("Healthcheck endpoint called")

    return HealthResponse()

## ============================================================
## PARSE TXT
## ============================================================
@router.post("/parse", response_model=ParseTxtResponse)
def parse_txt_endpoint(
    request: ParseTxtRequest,
    config: AppConfig = Depends(get_config),
) -> ParseTxtResponse:
    """
        Parse raw TXT laboratory files into structured CSV

        Returns:
            ParseTxtResponse payload
    """

    logger.info(
        "Parse request received | files=%s | overwrite=%s",
        request.filenames,
        request.overwrite,
    )

    try:
        result = parse_txt_pipeline(
            filenames=request.filenames,
            overwrite=request.overwrite,
            config=config,
        )

        return ParseTxtResponse(**result)

    except Exception as exc:
        logger.error("Parsing pipeline failed | error=%s", str(exc))
        raise ParsingError(message="TXT parsing failed", details={"error": str(exc)})

## ============================================================
## BUILD DATASET
## ============================================================
@router.post("/dataset", response_model=BuildDatasetResponse)
def build_dataset_endpoint(
    request: BuildDatasetRequest,
    config: AppConfig = Depends(get_config),
) -> BuildDatasetResponse:
    """
        Build clustering dataset from structured CSV files

        Returns:
            BuildDatasetResponse payload
    """

    logger.info(
        "Build dataset request | format=%s | overwrite=%s",
        request.dataset_format,
        request.overwrite,
    )

    try:
        result = build_dataset_pipeline(
            structured_csv_files=request.structured_csv_files,
            dataset_format=request.dataset_format,
            overwrite=request.overwrite,
            config=config,
        )

        return BuildDatasetResponse(**result)

    except Exception as exc:
        logger.error("Dataset build failed | error=%s", str(exc))
        raise DatasetBuildError(
            message="Dataset construction failed",
            details={"error": str(exc)},
        )

## ============================================================
## RUN CLUSTERING
## ============================================================
@router.post("/cluster", response_model=RunClusteringResponse)
def run_clustering_endpoint(
    request: RunClusteringRequest,
    config: AppConfig = Depends(get_config),
) -> RunClusteringResponse:
    """
        Run clustering pipeline on a prepared dataset

        Returns:
            RunClusteringResponse payload
    """

    logger.info(
        "Clustering request | dataset=%s | algorithm=%s",
        request.dataset_path,
        request.clustering.algorithm,
    )

    try:
        result = run_clustering_pipeline(
            dataset_path=request.dataset_path,
            clustering_params=request.clustering,
            preprocess_params=request.preprocess,
            overwrite=request.overwrite,
            config=config,
        )

        return RunClusteringResponse(**result)

    except Exception as exc:
        logger.error("Clustering pipeline failed | error=%s", str(exc))
        raise ClusteringError(
            message="Clustering pipeline failed",
            details={"error": str(exc)},
        )

## ============================================================
## EXPORT
## ============================================================
@router.post("/export", response_model=ExportResponse)
def export_endpoint(
    request: ExportRequest,
    config: AppConfig = Depends(get_config),
) -> ExportResponse:
    """
        Export clustering and/or EDA artifacts for a given run

        Returns:
            ExportResponse payload
    """

    logger.info(
        "Export request | run_id=%s | export_eda=%s | export_clustering=%s",
        request.run_id,
        request.export_eda,
        request.export_clustering,
    )

    try:
        result = export_pipeline(
            run_id=request.run_id,
            export_eda=request.export_eda,
            export_clustering=request.export_clustering,
            config=config,
        )

        return ExportResponse(**result)

    except Exception as exc:
        logger.error("Export pipeline failed | error=%s", str(exc))
        raise ClusteringError(
            message="Export failed",
            details={"error": str(exc)},
        )