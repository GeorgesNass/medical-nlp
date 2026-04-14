'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main CLI entry point for icd10_prediction: parsing, dataset build, training, EDA and API."
'''

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import uvicorn

from src.utils.logging_utils import get_logger
from src.core.data_consistency import run_data_consistency
from src.core.errors import (
    ConfigurationError,
    DataError,
    ModelError,
    ParsingError,
    PipelineError,
)
from src.core.config import build_config
from src.core.eda import (
    compute_label_distribution,
    compute_text_length_stats,
    plot_label_distribution,
)
from src.pipelines import (
    run_clinical_csv_build,
    run_rss_parsing,
    run_training_pipeline,
)

## ============================================================
## CONSTANTS
## ============================================================
APP_VERSION = "1.0.0"
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_PLATFORM_ERROR = 2

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

        Returns:
            Configured ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description="ICD10 prediction from clinical records (icd10_prediction).",
        add_help=True,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {APP_VERSION}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and log intended actions without executing workflows.",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration loading and resolved default paths, then exit.",
    )

    ## Main action flags
    parser.add_argument(
        "--parse-rss",
        action="store_true",
        help="Parse all raw RSS files and export a consolidated CSV.",
    )
    parser.add_argument(
        "--build-clinical-csv",
        action="store_true",
        help="Build one CSV per admission_id by merging RSS data with clinical_records files.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train baseline model (vectorize + train + export metrics).",
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Run basic EDA on per-admission CSV files (label distribution + text length stats).",
    )
    parser.add_argument(
        "--run-api",
        action="store_true",
        help="Run FastAPI service (uvicorn).",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run parse-rss -> build-clinical-csv -> train -> eda in sequence.",
    )

    ## Paths overrides (defaults from config)
    parser.add_argument(
        "--rss-dir",
        type=str,
        default="",
        help="Path to data/raw/icd10/ (folder containing .rss files).",
    )
    parser.add_argument(
        "--clinical-records-dir",
        type=str,
        default="",
        help="Path to data/raw/clinical_records/ (folder containing admission subfolders).",
    )
    parser.add_argument(
        "--clinical-csv-dir",
        type=str,
        default="",
        help="Path to data/interim/clinical_records_csv/ (folder containing per-admission CSVs).",
    )

    ## Outputs
    parser.add_argument(
        "--rss-output-csv",
        type=str,
        default="",
        help="Path to consolidated RSS CSV output (default: data/interim/icd10_csv/icd10_structured.csv).",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="",
        help="Path to save trained model (default: artifacts/models/model.joblib).",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="",
        help="Path to save metrics JSON (default: artifacts/reports/metrics.json).",
    )
    parser.add_argument(
        "--eda-plot-output",
        type=str,
        default="",
        help="Path to save EDA plot (default: artifacts/exports/eda/label_distribution.png).",
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
## HELPERS
## ============================================================
def _build_summary(
    action: str,
    success: bool,
    start: float,
    details: Optional[dict] = None,
) -> dict:
    """
        Build standardized execution summary

        Args:
            action: Executed action name
            success: Execution status
            start: Monotonic start timestamp
            details: Optional structured details

        Returns:
            Standardized summary dictionary
    """

    return {
        "action": action,
        "success": success,
        "duration_seconds": round(time.monotonic() - start, 3),
        "details": details or {},
    }

def _resolve_default_paths(config) -> dict:
    """
        Resolve default project paths from configuration

        Args:
            config: Project configuration object

        Returns:
            Dictionary containing resolved default paths
    """

    return {
        "default_rss_dir": config.paths.raw_dir / "icd10",
        "default_clinical_records_dir": config.paths.raw_dir / "clinical_records",
        "default_clinical_csv_dir": config.paths.interim_dir / "clinical_records_csv",
        "default_rss_output_csv": config.paths.interim_dir / "icd10_csv" / "icd10_structured.csv",
        "default_model_output": config.paths.artifacts_dir / "models" / "model.joblib",
        "default_metrics_output": config.paths.artifacts_dir / "reports" / "metrics.json",
        "default_eda_plot_output": config.paths.artifacts_dir / "exports" / "eda" / "label_distribution.png",
    }

def _resolve_cli_path(path_value: str, fallback: Path) -> Path:
    """
        Resolve CLI path override or fallback path

        Args:
            path_value: Raw CLI path string
            fallback: Default fallback path

        Returns:
            Resolved Path instance
    """

    return Path(path_value).expanduser().resolve() if path_value.strip() else fallback

def _load_rss_dataframe_from_csv(rss_output_csv: Path) -> pd.DataFrame:
    """
        Load consolidated RSS CSV with defensive validation

        Args:
            rss_output_csv: Path to consolidated RSS CSV file

        Returns:
            Loaded RSS DataFrame

        Raises:
            DataError: If the file is missing, empty or invalid
    """

    try:
        if not rss_output_csv.exists():
            from src.core.errors import log_and_raise_missing_file

            log_and_raise_missing_file(
                rss_output_csv,
                reason=(
                    "Run: python main.py --parse-rss "
                    "(this file is required for --build-clinical-csv)."
                ),
            )

        if rss_output_csv.stat().st_size == 0:
            from src.core.errors import log_and_raise_data_error

            log_and_raise_data_error(
                reason=(
                    f"RSS consolidated CSV is empty: {rss_output_csv} | "
                    "Run: python main.py --parse-rss and verify the RSS parser output."
                )
            )

        rss_df = pd.read_csv(rss_output_csv)

        if rss_df.empty:
            from src.core.errors import log_and_raise_data_error

            log_and_raise_data_error(
                reason=(
                    f"RSS consolidated CSV has no rows: {rss_output_csv} | "
                    "Parsing may have produced headers but no data."
                )
            )

        return rss_df

    except pd.errors.EmptyDataError as exc:
        from src.core.errors import log_and_raise_data_error

        log_and_raise_data_error(
            reason=f"RSS consolidated CSV has no columns to parse: {rss_output_csv} | {str(exc)}"
        )

## ============================================================
## MAIN EXECUTION
## ============================================================
def main() -> int:
    """
        Main CLI entry point

        Workflow notes:
            - parse-rss builds a consolidated structured dataset from fixed-width RSS files
            - build-clinical-csv merges RSS metadata with raw clinical documents per admission_id
            - train runs baseline vectorization + training + metrics export
            - eda exports basic dataset diagnostics (label distribution + text length stats)
            - run-api starts FastAPI server via uvicorn

        Returns:
            Standardized process exit code
    """

    start_time = time.monotonic()

    try:
        config = build_config()
        default_paths = _resolve_default_paths(config)

        parser = _build_parser()
        args = parser.parse_args()

        if args.validate_config:
            logger.info("Configuration validation succeeded")
            logger.info(
                "Resolved defaults | rss_dir=%s | clinical_records_dir=%s | clinical_csv_dir=%s",
                default_paths["default_rss_dir"],
                default_paths["default_clinical_records_dir"],
                default_paths["default_clinical_csv_dir"],
            )
            logger.info(
                "Summary | %s",
                _build_summary(
                    action="validate-config",
                    success=True,
                    start=start_time,
                    details={"config_loaded": True},
                ),
            )
            return EXIT_SUCCESS

        ## Decide which workflow to run
        if not any(
            [
                args.parse_rss,
                args.build_clinical_csv,
                args.train,
                args.eda,
                args.run_api,
                args.run_all,
            ]
        ):
            parser.print_help()
            logger.info(
                "Summary | %s",
                _build_summary(
                    action="help",
                    success=True,
                    start=start_time,
                ),
            )
            return EXIT_SUCCESS

        ## Resolve paths (CLI overrides config defaults)
        rss_dir = _resolve_cli_path(args.rss_dir, default_paths["default_rss_dir"])
        clinical_records_dir = _resolve_cli_path(
            args.clinical_records_dir,
            default_paths["default_clinical_records_dir"],
        )
        clinical_csv_dir = _resolve_cli_path(
            args.clinical_csv_dir,
            default_paths["default_clinical_csv_dir"],
        )
        rss_output_csv = _resolve_cli_path(
            args.rss_output_csv,
            default_paths["default_rss_output_csv"],
        )
        model_output = _resolve_cli_path(
            args.model_output,
            default_paths["default_model_output"],
        )
        metrics_output = _resolve_cli_path(
            args.metrics_output,
            default_paths["default_metrics_output"],
        )
        eda_plot_output = _resolve_cli_path(
            args.eda_plot_output,
            default_paths["default_eda_plot_output"],
        )

        ## ============================================================
        ## DATA CONSISTENCY CHECK
        ## ============================================================
        if config.data_consistency.enabled:
            consistency_result = run_data_consistency(
                data={
                    "text": "icd10_run",
                    "labels": ["A00"],
                },
                strict=config.data_consistency.strict_mode,
            )

            logger.info(f"Consistency OK: {consistency_result['is_consistent']}")

            if not consistency_result["is_consistent"] and config.data_consistency.strict_mode:
                raise DataError(
                    message="Data consistency failed before pipeline",
                    error_code="data_consistency_error",
                    details={"issues": consistency_result["issues"]},
                    origin="main",
                    http_status=400,
                    is_retryable=False,
                )

        if args.dry_run:
            logger.info(
                "Dry-run | parse_rss=%s | build_clinical_csv=%s | train=%s | eda=%s | run_api=%s | run_all=%s",
                bool(args.parse_rss),
                bool(args.build_clinical_csv),
                bool(args.train),
                bool(args.eda),
                bool(args.run_api),
                bool(args.run_all),
            )
            logger.info(
                "Dry-run paths | rss_dir=%s | clinical_records_dir=%s | clinical_csv_dir=%s",
                rss_dir,
                clinical_records_dir,
                clinical_csv_dir,
            )
            logger.info(
                "Summary | %s",
                _build_summary(
                    action="dry-run",
                    success=True,
                    start=start_time,
                ),
            )
            return EXIT_SUCCESS

        ## RUN ALL
        if args.run_all:
            logger.info("Running full pipeline: parse-rss -> build-clinical-csv -> train -> eda")

            rss_df = run_rss_parsing(
                rss_folder=rss_dir,
                output_csv=rss_output_csv,
            )

            run_clinical_csv_build(
                clinical_records_dir=clinical_records_dir,
                rss_df=rss_df,
                output_dir=clinical_csv_dir,
            )

            run_training_pipeline(
                clinical_csv_dir=clinical_csv_dir,
                model_output_path=model_output,
                metrics_output_path=metrics_output,
            )

            dist = compute_label_distribution(clinical_csv_dir)
            plot_label_distribution(dist, eda_plot_output, top_k=20)
            stats = compute_text_length_stats(clinical_csv_dir)

            logger.info("EDA stats: %s", stats)
            logger.info("Full pipeline completed")
            logger.info(
                "Summary | %s",
                _build_summary(
                    action="run-all",
                    success=True,
                    start=start_time,
                ),
            )
            return EXIT_SUCCESS

        ## PARSE RSS
        rss_df = None
        if args.parse_rss:
            rss_df = run_rss_parsing(
                rss_folder=rss_dir,
                output_csv=rss_output_csv,
            )
            logger.info("RSS parsing completed")

        ## BUILD CLINICAL CSV
        if args.build_clinical_csv:
            ## If rss_df not produced in this run, load from consolidated CSV
            if rss_df is None:
                rss_df = _load_rss_dataframe_from_csv(rss_output_csv)

            run_clinical_csv_build(
                clinical_records_dir=clinical_records_dir,
                rss_df=rss_df,
                output_dir=clinical_csv_dir,
            )
            logger.info("Clinical CSV build completed")

        ## TRAIN
        if args.train:
            run_training_pipeline(
                clinical_csv_dir=clinical_csv_dir,
                model_output_path=model_output,
                metrics_output_path=metrics_output,
            )
            logger.info("Training completed")

        ## EDA
        if args.eda:
            dist = compute_label_distribution(clinical_csv_dir)
            plot_label_distribution(dist, eda_plot_output, top_k=20)
            stats = compute_text_length_stats(clinical_csv_dir)
            logger.info("EDA stats: %s", stats)
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

        logger.info(
            "Summary | %s",
            _build_summary(
                action="run",
                success=True,
                start=start_time,
            ),
        )
        return EXIT_SUCCESS

    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user")
        logger.warning(
            "Summary | %s",
            _build_summary(
                action="interrupt",
                success=False,
                start=start_time,
            ),
        )
        return EXIT_FAILURE

    except (ConfigurationError, DataError, ParsingError, PipelineError, ModelError) as exc:
        logger.error("Known application error: %s", exc)
        logger.error(
            "Summary | %s",
            _build_summary(
                action="known-error",
                success=False,
                start=start_time,
                details={"error": str(exc)},
            ),
        )
        return EXIT_PLATFORM_ERROR

    except Exception as exc:
        logger.exception("Unhandled exception: %s", exc)
        logger.error(
            "Summary | %s",
            _build_summary(
                action="unhandled-exception",
                success=False,
                start=start_time,
                details={"error": str(exc)},
            ),
        )
        return EXIT_FAILURE

## ============================================================
## ENTRYPOINT
## ============================================================
if __name__ == "__main__":
    sys.exit(main())