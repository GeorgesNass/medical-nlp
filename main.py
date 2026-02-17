'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main CLI entry point for icd10_prediction (parse RSS, build clinical CSV, train/eval, export, run EDA, run API)."
'''

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import sys
import uvicorn

from src.utils.logging_utils import get_logger
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
## LOGGER
## ============================================================
logger = get_logger("main", log_file="main.log")

## ============================================================
## CLI ARGUMENTS
## ============================================================
def _build_parser() -> argparse.ArgumentParser:
    """
        Build argument parser for CLI usage

        Returns:
            Configured ArgumentParser
    """

    parser = argparse.ArgumentParser(description="ICD10 prediction from clinical records (icd10_prediction).",)

    ## Main action flags
    parser.add_argument(
        "--parse-rss", action="store_true",
        help="Parse all raw RSS files and export a consolidated CSV.",)
    parser.add_argument(
        "--build-clinical-csv", action="store_true",
        help="Build one CSV per admission_id by merging RSS data with clinical_records files.",)
    parser.add_argument(
            "--train", action="store_true",
            help="Train baseline model (vectorize + train + export metrics).",)
    parser.add_argument(
        "--eda", action="store_true",
        help="Run basic EDA on per-admission CSV files (label distribution + text length stats).",)
    parser.add_argument(
        "--run-api", action="store_true",
        help="Run FastAPI service (uvicorn).",)
    parser.add_argument(
        "--run-all", action="store_true",
        help="Run parse-rss -> build-clinical-csv -> train -> eda in sequence.",)

    ## Paths overrides (defaults from config)
    parser.add_argument(
        "--rss-dir", type=str, default="",
        help="Path to data/raw/icd10/ (folder containing .rss files).",)
    parser.add_argument(
        "--clinical-records-dir", type=str, default="",
        help="Path to data/raw/clinical_records/ (folder containing admission subfolders).",)
    parser.add_argument(
        "--clinical-csv-dir", type=str, default="",
        help="Path to data/interim/clinical_records_csv/ (folder containing per-admission CSVs).",)

    ## Outputs
    parser.add_argument(
        "--rss-output-csv", type=str, default="",
        help="Path to consolidated RSS CSV output (default: data/interim/icd10_csv/icd10_structured.csv).",)
    parser.add_argument(
        "--model-output", type=str, default="",
        help="Path to save trained model (default: artifacts/models/model.joblib).",)
    parser.add_argument(
        "--metrics-output", type=str, default="",
        help="Path to save metrics JSON (default: artifacts/reports/metrics.json).",)
    parser.add_argument(
        "--eda-plot-output", type=str, default="",
        help="Path to save EDA plot (default: artifacts/exports/eda/label_distribution.png).",)

    ## API options
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="API host (default: 0.0.0.0).",)
    parser.add_argument(
        "--port", type=int, default=8000,
        help="API port (default: 8000).",)
    parser.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload (dev mode).",)

    return parser

## ============================================================
## MAIN EXECUTION
## ============================================================
def main() -> None:
    """
        Main CLI entry point

        Workflow notes:
            - parse-rss builds a consolidated structured dataset from fixed-width RSS files
            - build-clinical-csv merges RSS metadata with raw clinical documents per admission_id
            - train runs baseline vectorization + training + metrics export
            - eda exports basic dataset diagnostics (label distribution + text length stats)
            - run-api starts FastAPI server via uvicorn
    """

    try:
        config = build_config()

        ## Default paths from project structure
        default_rss_dir = config.paths.raw_dir / "icd10"
        default_clinical_records_dir = config.paths.raw_dir / "clinical_records"
        default_clinical_csv_dir = config.paths.interim_dir / "clinical_records_csv"
        default_rss_output_csv = config.paths.interim_dir / "icd10_csv" / "icd10_structured.csv"

        default_model_output = config.paths.artifacts_dir / "models" / "model.joblib"
        default_metrics_output = config.paths.artifacts_dir / "reports" / "metrics.json"
        default_eda_plot_output = config.paths.artifacts_dir / "exports" / "eda" / "label_distribution.png"

        parser = _build_parser()
        args = parser.parse_args()

        ## Decide which workflow to run
        if not any([args.parse_rss, args.build_clinical_csv, args.train, args.eda, args.run_api, args.run_all]):
            parser.print_help()
            return

        ## Resolve paths (CLI overrides config defaults)
        rss_dir = Path(args.rss_dir).expanduser().resolve() if args.rss_dir.strip() else default_rss_dir
        clinical_records_dir = (
            Path(args.clinical_records_dir).expanduser().resolve()
            if args.clinical_records_dir.strip()
            else default_clinical_records_dir
        )
        clinical_csv_dir = (
            Path(args.clinical_csv_dir).expanduser().resolve()
            if args.clinical_csv_dir.strip()
            else default_clinical_csv_dir
        )

        rss_output_csv = (
            Path(args.rss_output_csv).expanduser().resolve()
            if args.rss_output_csv.strip()
            else default_rss_output_csv
        )
        model_output = (
            Path(args.model_output).expanduser().resolve()
            if args.model_output.strip()
            else default_model_output
        )
        metrics_output = (
            Path(args.metrics_output).expanduser().resolve()
            if args.metrics_output.strip()
            else default_metrics_output
        )
        eda_plot_output = (
            Path(args.eda_plot_output).expanduser().resolve()
            if args.eda_plot_output.strip()
            else default_eda_plot_output
        )

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
            return

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
                try:
                    if not rss_output_csv.exists():
                        from src.core.errors import log_and_raise_missing_file
                        log_and_raise_missing_file(
                            rss_output_csv,
                            reason="Run: python main.py --parse-rss (this file is required for --build-clinical-csv).",
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

                except pd.errors.EmptyDataError as exc:
                    from src.core.errors import log_and_raise_data_error
                    log_and_raise_data_error(
                        reason=f"RSS consolidated CSV has no columns to parse: {rss_output_csv} | {str(exc)}"
                    )

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
            logger.info("Starting API server | host=%s port=%d reload=%s", args.host, args.port, bool(args.reload))
            uvicorn.run(
                "src.core.service:app",
                host=args.host,
                port=args.port,
                reload=bool(args.reload),
            )

    except (ConfigurationError, DataError, ParsingError, PipelineError, ModelError) as exc:
        print(f"\nERROR: {exc}\n")
        sys.exit(2)
        
if __name__ == "__main__":
    main()