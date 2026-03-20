'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main CLI entry point for doc-classification: index building, prediction, export and EDA."
'''

from __future__ import annotations

## Standard library
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

## Project imports
from src.core.config import CONFIG
from src.core.eda import run_eda_on_folder
from src.pipeline import (
    build_similarity_index_from_labeled,
    export_predictions,
    predict_labels_for_unlabeled,
)
from src.utils.logging_utils import get_logger

## ============================================================
## CONSTANTS
## ============================================================
APP_VERSION = "1.0.0"
EXIT_SUCCESS = 0
EXIT_FAILURE = 1

logger = get_logger("doc_classification.main")

## ============================================================
## ARG PARSER
## ============================================================
def _build_parser() -> argparse.ArgumentParser:
    """
        Build CLI parser

        Returns:
            ArgumentParser instance
    """

    parser = argparse.ArgumentParser(
        description="Document classification pipeline",
        add_help=True,
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {APP_VERSION}")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-config", action="store_true")

    ## Actions
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--eda", action="store_true")
    parser.add_argument("--run-all", action="store_true")

    ## Paths
    parser.add_argument("--labeled-dir", type=str, default=str(CONFIG.paths.labeled_dir))
    parser.add_argument("--unlabeled-dir", type=str, default=str(CONFIG.paths.unlabeled_dir))
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(CONFIG.paths.data_dir / "labeled_manifest.json"),
    )

    ## Outputs
    parser.add_argument("--output-csv", type=str, default="predictions.csv")
    parser.add_argument("--include-scores", action="store_true")
    parser.add_argument("--include-evidence", action="store_true")

    ## EDA
    parser.add_argument("--eda-folder", type=str, default="")
    parser.add_argument("--eda-output", type=str, default="eda_summary.json")

    return parser

## ============================================================
## VALIDATION
## ============================================================
def _validate_runtime() -> dict:
    """
        Validate runtime

        Returns:
            Summary dict
    """

    return {
        "cwd": str(Path.cwd()),
        "config_paths": str(CONFIG.paths),
        "python": sys.executable,
    }

def _build_summary(action: str, success: bool, start: float, details: Optional[dict] = None) -> dict:
    """
        Build execution summary

        Args:
            action: Action name
            success: Status
            start: Start time
            details: Optional details

        Returns:
            Summary dict
    """

    return {
        "action": action,
        "success": success,
        "duration_seconds": round(time.monotonic() - start, 3),
        "details": details or {},
    }

## ============================================================
## MAIN
## ============================================================
def main() -> int:
    """
        Main CLI entry point

        Returns:
            Exit code
    """

    start_time = time.monotonic()

    parser = _build_parser()
    args = parser.parse_args()

    try:
        ## Validate runtime
        runtime = _validate_runtime()

        if args.validate_config:
            logger.info("Config OK | %s", runtime)
            logger.info("Summary | %s", _build_summary("validate-config", True, start_time))
            return EXIT_SUCCESS

        ## Resolve paths
        labeled_dir = Path(args.labeled_dir).expanduser().resolve()
        unlabeled_dir = Path(args.unlabeled_dir).expanduser().resolve()
        manifest_path = Path(args.manifest).expanduser().resolve()

        if not any([args.build_index, args.predict, args.export, args.eda, args.run_all]):
            parser.print_help()
            return EXIT_SUCCESS

        if args.dry_run:
            logger.info("Dry-run | no execution")
            logger.info("Summary | %s", _build_summary("dry-run", True, start_time))
            return EXIT_SUCCESS

        ## ====================================================
        ## EDA
        ## ====================================================
        if args.eda:
            eda_folder = (
                Path(args.eda_folder).expanduser().resolve()
                if args.eda_folder.strip()
                else (labeled_dir if labeled_dir.exists() else unlabeled_dir)
            )

            logger.info("Running EDA on %s", eda_folder)

            run_eda_on_folder(
                folder_path=eda_folder,
                labeled_manifest=None,
                output_name=args.eda_output,
            )

            logger.info("EDA finished")

            if not (args.build_index or args.predict or args.run_all):
                logger.info("Summary | %s", _build_summary("eda", True, start_time))
                return EXIT_SUCCESS

        ## ====================================================
        ## PIPELINE
        ## ====================================================
        predictions = None
        index = None

        if args.run_all:
            logger.info("Running full pipeline")

            index = build_similarity_index_from_labeled(
                labeled_folder=labeled_dir,
                manifest_path=manifest_path,
            )

            predictions = predict_labels_for_unlabeled(
                unlabeled_folder=unlabeled_dir,
                index=index,
            )

            export_path = export_predictions(
                predictions=predictions,
                output_csv_name=args.output_csv,
                include_scores=args.include_scores,
                include_evidence=args.include_evidence,
            )

            logger.info("Export completed: %s", export_path)
            logger.info("Summary | %s", _build_summary("run-all", True, start_time))
            return EXIT_SUCCESS

        if args.build_index:
            logger.info("Building index")
            index = build_similarity_index_from_labeled(
                labeled_folder=labeled_dir,
                manifest_path=manifest_path,
            )

        if args.predict:
            if index is None:
                logger.info("Index missing -> building first")
                index = build_similarity_index_from_labeled(
                    labeled_folder=labeled_dir,
                    manifest_path=manifest_path,
                )

            predictions = predict_labels_for_unlabeled(
                unlabeled_folder=unlabeled_dir,
                index=index,
            )

        if args.export:
            if predictions is None:
                raise ValueError("Export requires predictions")

            export_path = export_predictions(
                predictions=predictions,
                output_csv_name=args.output_csv,
                include_scores=args.include_scores,
                include_evidence=args.include_evidence,
            )

            logger.info("Export completed: %s", export_path)

        logger.info("Summary | %s", _build_summary("run", True, start_time))
        return EXIT_SUCCESS

    except KeyboardInterrupt:
        logger.warning("Interrupted")
        logger.warning("Summary | %s", _build_summary("interrupt", False, start_time))
        return EXIT_FAILURE

    except Exception as exc:
        logger.exception("Error: %s", exc)
        logger.error("Summary | %s", _build_summary("error", False, start_time))
        return EXIT_FAILURE

## ============================================================
## ENTRYPOINT
## ============================================================
if __name__ == "__main__":
    sys.exit(main())