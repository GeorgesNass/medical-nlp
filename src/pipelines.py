'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main pipeline orchestration: parsing, dataset building, clustering execution and export management."
'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.clustering.build_dataset import build_dataset
from src.clustering.export import export_clustering_artifacts
from src.clustering.mlflow_tracking import track_clustering_run
from src.clustering.preprocess import preprocess_dataset
from src.clustering.algorithms import run_clustering_algorithm
from src.core.config import AppConfig
from src.core.errors import (
    ClusteringError,
    DatasetBuildError,
    ParsingError,
    wrap_exception_as,
)
from src.parser.parse_txt import parse_txt_file
from src.utils.io_utils import (
    assert_exists,
    ensure_dir,
    read_csv,
    read_parquet,
    write_csv,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
## PARSE TXT PIPELINE
## ============================================================
def parse_txt_pipeline(
    filenames: List[str],
    overwrite: bool,
    config: AppConfig,
) -> Dict[str, Any]:
    """
        Parse raw TXT files into structured CSV files

        High-level workflow:
            1) Validate raw file existence
            2) Parse each TXT file
            3) Export structured CSV into interim folder
            4) Return summary payload

        Args:
            filenames: List of filenames under data/raw
            overwrite: Overwrite existing CSV
            config: AppConfig instance

        Returns:
            Dictionary for API response
    """

    parsed_files: List[str] = []
    failed_files: List[str] = []

    ## Iterate over each requested TXT file
    for filename in filenames:

        ## Resolve raw file path
        raw_path = config.paths.raw_dir / filename

        try:
            ## Validate raw file existence
            assert_exists(raw_path, kind="file")

            ## Parse TXT file into structured DataFrame
            structured_df = parse_txt_file(raw_path, config)

            ## Build output CSV path in interim folder
            output_path = config.paths.interim_structured_dir / (
                Path(filename).stem + ".csv"
            )

            ## Skip existing file if overwrite disabled
            if output_path.exists() and not overwrite:
                logger.info("Skipping existing file: %s", output_path.name)
                parsed_files.append(output_path.name)
                continue

            ## Persist structured CSV
            write_csv(structured_df, output_path)
            parsed_files.append(output_path.name)

        except ParsingError as exc:
            logger.error("Parsing failed for file=%s | error=%s", filename, str(exc))
            failed_files.append(filename)

        except Exception as exc:
            logger.error("Parsing failed for file=%s | error=%s", filename, str(exc))
            failed_files.append(filename)

    return {
        "parsed_files": parsed_files,
        "failed_files": failed_files,
        "message": "Parsing completed",
    }

## ============================================================
## BUILD DATASET PIPELINE
## ============================================================
def build_dataset_pipeline(
    structured_csv_files: Optional[List[str]],
    dataset_format: str,
    overwrite: bool,
    config: AppConfig,
) -> Dict[str, Any]:
    """
        Build clustering dataset from structured CSV files

        Args:
            structured_csv_files: Optional subset of structured CSV files
            dataset_format: long or wide
            overwrite: Overwrite existing dataset
            config: AppConfig instance

        Returns:
            Dictionary for API response
    """

    try:
        ## Build dataset from structured CSV inputs
        dataset_df = build_dataset(
            structured_csv_files=structured_csv_files,
            dataset_format=dataset_format,
            config=config,
        )

        ## Resolve dataset output path
        output_path = config.paths.interim_datasets_dir / (
            f"dataset_{dataset_format}.parquet"
        )

        ## Write dataset if needed
        if output_path.exists() and not overwrite:
            logger.info("Dataset already exists: %s", output_path.name)
        else:
            ensure_dir(output_path.parent)
            dataset_df.to_parquet(output_path, index=False)

        return {
            "dataset_path": str(output_path),
            "n_rows": int(dataset_df.shape[0]),
            "n_cols": int(dataset_df.shape[1]),
            "message": "Dataset built successfully",
        }

    except DatasetBuildError:
        raise

    except Exception as exc:
        logger.error("Dataset build pipeline failed | error=%s", str(exc))
        raise wrap_exception_as(
            exc=exc,
            exc_type=DatasetBuildError,
            message="Dataset build failed",
            details={
                "dataset_format": dataset_format,
                "structured_csv_files": structured_csv_files or [],
            },
        )

## ============================================================
## CLUSTERING PIPELINE
## ============================================================
def run_clustering_pipeline(
    dataset_path: str,
    clustering_params: Any,
    preprocess_params: Dict[str, Any],
    overwrite: bool,
    config: AppConfig,
) -> Dict[str, Any]:
    """
        Execute full clustering pipeline

        Args:
            dataset_path: Path to dataset file
            clustering_params: Clustering parameters object
            preprocess_params: Preprocessing options
            overwrite: Overwrite artifacts
            config: AppConfig instance

        Returns:
            Dictionary for API response
    """

    try:
        ## Validate dataset existence
        path = assert_exists(dataset_path, kind="file")

        ## Load dataset according to extension
        if path.suffix == ".parquet":
            df = read_parquet(path)
        else:
            df = read_csv(path)

        ## Preprocess dataset (imputation, scaling, optional PCA)
        processed_df, preprocessing_metadata = preprocess_dataset(
            df,
            preprocess_params,
        )

        ## Execute clustering algorithm
        clustering_result = run_clustering_algorithm(
            processed_df,
            clustering_params,
        )

        ## Track experiment in MLflow
        mlflow_run_id = track_clustering_run(
            clustering_params=clustering_params,
            preprocess_params=preprocess_params,
            metrics=clustering_result["metrics"],
            config=config,
        )

        ## Export clustering artifacts
        exports = export_clustering_artifacts(
            clustering_result=clustering_result,
            dataset=df,
            config=config,
            overwrite=overwrite,
        )

        return {
            "run_id": config.runtime.run_id,
            "mlflow_run_id": mlflow_run_id,
            "n_clusters": clustering_result["n_clusters"],
            "metrics": clustering_result["metrics"],
            "exports": exports,
            "message": "Clustering completed successfully",
        }

    except ClusteringError:
        raise

    except Exception as exc:
        logger.error("Clustering pipeline failed | error=%s", str(exc))
        raise wrap_exception_as(
            exc=exc,
            exc_type=ClusteringError,
            message="Clustering execution failed",
            details={
                "dataset_path": dataset_path,
                "algorithm": getattr(clustering_params, "algorithm", "unknown"),
            },
        )

## ============================================================
## EXPORT PIPELINE
## ============================================================
def export_pipeline(
    run_id: str,
    export_eda: bool,
    export_clustering: bool,
    config: AppConfig,
) -> Dict[str, Any]:
    """
        Manage export operations for a given run

        Args:
            run_id: Runtime identifier
            export_eda: Export EDA artifacts
            export_clustering: Export clustering artifacts
            config: AppConfig instance

        Returns:
            Dictionary for API response
    """

    exports: List[str] = []

    ## Handle EDA export flag
    if export_eda:
        exports.append("EDA export handled via core.eda module")

    ## Handle clustering export flag
    if export_clustering:
        exports.append("Clustering artifacts already exported during run")

    return {
        "exports": exports,
        "message": f"Export completed for run_id={run_id}",
    }