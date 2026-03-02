'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main CLI entry point for lab_clustering (parse TXT, build dataset, cluster, export, run EDA, run API)."
'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import uvicorn

from src.core.config import AppConfig, build_config
from src.core.eda import run_dataset_eda, run_structured_eda
from src.core.errors import (
    ClusteringError,
    ConfigurationError,
    DatasetBuildError,
    LabClusteringError,
    ParsingError,
)
from src.pipelines import (
    build_dataset_pipeline,
    parse_txt_pipeline,
    run_clustering_pipeline,
)
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("main")

## ============================================================
## CLI ARGUMENTS
## ============================================================
def _build_parser() -> argparse.ArgumentParser:
    """
        Build argument parser for CLI usage

        High-level workflow:
            1) Define action flags (parse, dataset, cluster, eda, api)
            2) Define optional path overrides
            3) Define API options

        Returns:
            Configured ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description="Unsupervised clustering and analysis of laboratory data (lab_clustering)."
    )

    ## Main actions
    parser.add_argument(
        "--parse-txt",
        action="store_true",
        help="Parse raw TXT files from data/raw and export structured CSV to data/interim/lab_structured_csv.",
    )
    parser.add_argument(
        "--build-dataset",
        action="store_true",
        help="Build dataset from structured CSV files and export to data/interim/datasets.",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Run clustering on a dataset (preprocess + fit + metrics + MLflow + exports).",
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Run basic EDA on structured CSVs and/or datasets.",
    )
    parser.add_argument(
        "--run-api",
        action="store_true",
        help="Run FastAPI service (uvicorn).",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run parse-txt -> build-dataset -> cluster -> eda in sequence.",
    )

    ## Parse options
    parser.add_argument(
        "--txt-files",
        type=str,
        default="",
        help="Comma-separated filenames under data/raw (default: all .txt in data/raw).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if present.",
    )

    ## Dataset options
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="wide",
        choices=["wide", "long"],
        help="Dataset format to build (wide or long).",
    )

    ## Clustering options
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="Path to dataset (default: data/interim/datasets/dataset_wide.parquet).",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="kmeans",
        choices=["kmeans", "agglomerative", "dbscan", "birch"],
        help="Clustering algorithm.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="Number of clusters (for kmeans/agglomerative/birch).",
    )
    parser.add_argument(
        "--apply-pca",
        action="store_true",
        help="Apply PCA for dimensionality reduction.",
    )
    parser.add_argument(
        "--pca-n-components",
        type=int,
        default=2,
        help="Number of PCA components if --apply-pca is enabled.",
    )

    ## API options
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API host (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API port (default: 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (dev mode).",
    )

    return parser

## ============================================================
## PATH RESOLUTION HELPERS
## ============================================================
def _parse_txt_files_arg(arg: str) -> list[str]:
    """
        Parse --txt-files argument into list

        High-level workflow:
            1) Split on commas
            2) Strip whitespace
            3) Remove empty tokens

        Args:
            arg: Raw CLI argument string

        Returns:
            List of filenames
    """

    if not arg.strip():
        return []

    return [x.strip() for x in arg.split(",") if x.strip()]

def _default_dataset_path(config: AppConfig, dataset_format: str) -> Path:
    """
        Resolve default dataset path based on format

        Args:
            config: AppConfig instance
            dataset_format: Dataset format

        Returns:
            Path to default dataset
    """

    return config.paths.interim_datasets_dir / f"dataset_{dataset_format}.parquet"

## ============================================================
## MAIN EXECUTION
## ============================================================
def main() -> None:
    """
        Main CLI entry point

        Workflow notes:
            - parse-txt parses raw TXT reports into structured analyte-level CSV files
            - build-dataset builds wide/long datasets from structured CSV files
            - cluster runs preprocessing and unsupervised clustering and exports artifacts with MLflow tracking
            - eda exports basic diagnostics for structured data and datasets
            - run-api starts FastAPI server via uvicorn
    """

    try:
        config = build_config()

        parser = _build_parser()
        args = parser.parse_args()

        ## Decide which workflow to run
        if not any(
            [
                args.parse_txt,
                args.build_dataset,
                args.cluster,
                args.eda,
                args.run_api,
                args.run_all,
            ]
        ):
            parser.print_help()
            return

        ## Parse file list
        txt_files = _parse_txt_files_arg(args.txt_files)

        ## Resolve dataset path
        dataset_path = (
            Path(args.dataset_path).expanduser().resolve()
            if args.dataset_path.strip()
            else _default_dataset_path(config, args.dataset_format)
        )

        ## RUN ALL
        if args.run_all:
            logger.info("Running full pipeline: parse-txt -> build-dataset -> cluster -> eda")

            ## Parse TXT
            if not txt_files:
                txt_files = [p.name for p in config.paths.raw_dir.glob("*.txt")]

            parse_result = parse_txt_pipeline(
                filenames=txt_files,
                overwrite=bool(args.overwrite),
                config=config,
            )

            ## Build dataset
            dataset_result = build_dataset_pipeline(
                structured_csv_files=None,
                dataset_format=args.dataset_format,
                overwrite=bool(args.overwrite),
                config=config,
            )

            ## Run clustering
            clustering_params = {
                "algorithm": args.algorithm,
                "params": {"n_clusters": int(args.n_clusters)},
            }
            preprocess_params = {
                "impute_strategy": "median",
                "apply_pca": bool(args.apply_pca),
                "pca_n_components": int(args.pca_n_components),
            }

            run_clustering_pipeline(
                dataset_path=dataset_result["dataset_path"],
                clustering_params=type(
                    "TmpParams",
                    (),
                    {
                        "algorithm": clustering_params["algorithm"],
                        "model_dump": lambda self: clustering_params["params"],
                    },
                )(),
                preprocess_params=preprocess_params,
                overwrite=bool(args.overwrite),
                config=config,
            )

            ## EDA (structured + dataset)
            structured_paths = list(config.paths.interim_structured_dir.glob("*.csv"))
            run_structured_eda([str(p) for p in structured_paths], config=config)

            run_dataset_eda(str(dataset_path), config=config)

            logger.info("Full pipeline completed")
            logger.info("Parsed files: %s", parse_result.get("parsed_files", []))
            return

        ## PARSE TXT
        if args.parse_txt:
            if not txt_files:
                txt_files = [p.name for p in config.paths.raw_dir.glob("*.txt")]

            parse_txt_pipeline(
                filenames=txt_files,
                overwrite=bool(args.overwrite),
                config=config,
            )
            logger.info("TXT parsing completed")

        ## BUILD DATASET
        if args.build_dataset:
            build_dataset_pipeline(
                structured_csv_files=None,
                dataset_format=args.dataset_format,
                overwrite=bool(args.overwrite),
                config=config,
            )
            logger.info("Dataset build completed")

        ## CLUSTER
        if args.cluster:
            clustering_params = {
                "algorithm": args.algorithm,
                "params": {"n_clusters": int(args.n_clusters)},
            }
            preprocess_params = {
                "impute_strategy": "median",
                "apply_pca": bool(args.apply_pca),
                "pca_n_components": int(args.pca_n_components),
            }

            run_clustering_pipeline(
                dataset_path=str(dataset_path),
                clustering_params=type(
                    "TmpParams",
                    (),
                    {
                        "algorithm": clustering_params["algorithm"],
                        "model_dump": lambda self: clustering_params["params"],
                    },
                )(),
                preprocess_params=preprocess_params,
                overwrite=bool(args.overwrite),
                config=config,
            )
            logger.info("Clustering completed")

        ## EDA
        if args.eda:
            structured_paths = list(config.paths.interim_structured_dir.glob("*.csv"))
            if structured_paths:
                run_structured_eda([str(p) for p in structured_paths], config=config)

            if dataset_path.exists():
                run_dataset_eda(str(dataset_path), config=config)

            logger.info("EDA completed")

        ## RUN API
        if args.run_api:
            logger.info(
                "Starting API server | host=%s port=%d reload=%s",
                args.host,
                args.port,
                bool(args.reload),
            )
            uvicorn.run(
                "src.core.service:app",
                host=args.host,
                port=args.port,
                reload=bool(args.reload),
            )

    except (ConfigurationError, ParsingError, DatasetBuildError, ClusteringError, LabClusteringError) as exc:
        print(f"\nERROR: {exc}\n")
        sys.exit(2)


if __name__ == "__main__":
    main()