'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main CLI entry point for the doc-classification project (build index, predict labels, export CSV, run EDA)."
'''

from __future__ import annotations

import argparse
from pathlib import Path

from src.core.config import CONFIG
from src.core.eda import run_eda_on_folder
from src.pipeline import (
    build_similarity_index_from_labeled,
    export_predictions,
    predict_labels_for_unlabeled,
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

        Returns:
            Configured ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description="Medical document classification and topic labeling (doc-classification).",
    )

    ## Main action flags
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build similarity index from labeled documents + manifest JSON.",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Predict labels for unlabeled documents using the similarity index.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export last predictions to CSV (requires --predict).",
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Run EDA on a folder (labeled or unlabeled).",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run build-index + predict + export in sequence.",
    )

    ## Paths (defaults from CONFIG)
    parser.add_argument(
        "--labeled-dir",
        type=str,
        default=str(CONFIG.paths.labeled_dir),
        help="Folder containing labeled documents.",
    )
    parser.add_argument(
        "--unlabeled-dir",
        type=str,
        default=str(CONFIG.paths.unlabeled_dir),
        help="Folder containing unlabeled documents.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(CONFIG.paths.data_dir / "labeled_manifest.json"),
        help="JSON manifest mapping labeled filenames to their labels.",
    )

    ## Outputs
    parser.add_argument(
        "--output-csv",
        type=str,
        default="predictions.csv",
        help="Output CSV filename (written to artifacts/exports/).",
    )
    parser.add_argument(
        "--include-scores",
        action="store_true",
        help="Include per-label similarity score columns in CSV export.",
    )
    parser.add_argument(
        "--include-evidence",
        action="store_true",
        help="Include per-label evidence columns in CSV export.",
    )

    ## EDA options
    parser.add_argument(
        "--eda-folder",
        type=str,
        default="",
        help="Folder path for EDA. If empty, uses --labeled-dir when available, else --unlabeled-dir.",
    )
    parser.add_argument(
        "--eda-output",
        type=str,
        default="eda_summary.json",
        help="EDA JSON output filename (written to artifacts/reports/).",
    )

    return parser


## ============================================================
## MAIN EXECUTION
## ============================================================
def main() -> None:
    """
        Main CLI entry point

        Workflow notes:
            - build-index reads labeled docs and manifest to create an in-memory index
            - predict runs similarity labeling on unlabeled docs
            - export writes a CSV with TRUE/FALSE per label, plus optional details
            - run-all executes build-index -> predict -> export
            - eda computes basic corpus stats and exports a JSON summary
    """

    parser = _build_parser()
    args = parser.parse_args()

    ## --------------------------------------------------------
    ## Resolve folders and paths
    ## --------------------------------------------------------
    labeled_dir = Path(args.labeled_dir).expanduser().resolve()
    unlabeled_dir = Path(args.unlabeled_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()

    ## --------------------------------------------------------
    ## Decide which workflow to run
    ## --------------------------------------------------------
    if not any([args.build_index, args.predict, args.export, args.eda, args.run_all]):
        ## No flags provided -> show help and exit cleanly
        parser.print_help()
        return

    ## --------------------------------------------------------
    ## EDA can be run independently
    ## --------------------------------------------------------
    if args.eda:
        ## Choose folder for EDA (explicit overrides default)
        if args.eda_folder.strip():
            eda_folder = Path(args.eda_folder).expanduser().resolve()
        else:
            ## Default: labeled dir if it exists, else unlabeled dir
            eda_folder = labeled_dir if labeled_dir.exists() else unlabeled_dir

        logger.info(f"Running EDA on folder: {eda_folder}")
        run_eda_on_folder(
            folder_path=eda_folder,
            labeled_manifest=None,
            output_name=args.eda_output,
        )
        logger.info("EDA finished")

        ## If user only asked for EDA, we can exit now
        if not (args.build_index or args.predict or args.run_all):
            return

    ## --------------------------------------------------------
    ## Build index / Predict / Export (in-memory for now)
    ## --------------------------------------------------------
    predictions = None

    if args.run_all:
        ## Run full pipeline
        logger.info("Running full pipeline: build-index -> predict -> export")

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
            include_scores=bool(args.include_scores),
            include_evidence=bool(args.include_evidence),
        )

        logger.info(f"Export completed: {export_path}")
        return

    ## Build index if requested
    index = None
    if args.build_index:
        logger.info("Building similarity index from labeled documents")
        index = build_similarity_index_from_labeled(
            labeled_folder=labeled_dir,
            manifest_path=manifest_path,
        )
        logger.info("Index build finished")

    ## Predict if requested
    if args.predict:
        ## If predict requested without build-index, we still need an index
        if index is None:
            logger.info("Index not built in this run, building index first (required for predict)")
            index = build_similarity_index_from_labeled(
                labeled_folder=labeled_dir,
                manifest_path=manifest_path,
            )

        logger.info("Predicting labels for unlabeled documents")
        predictions = predict_labels_for_unlabeled(
            unlabeled_folder=unlabeled_dir,
            index=index,
        )
        logger.info("Prediction finished")

    ## Export if requested
    if args.export:
        if predictions is None:
            raise ValueError("--export requires predictions. Use --predict or --run-all first.")

        export_path = export_predictions(
            predictions=predictions,
            output_csv_name=args.output_csv,
            include_scores=bool(args.include_scores),
            include_evidence=bool(args.include_evidence),
        )

        logger.info(f"Export completed: {export_path}")


if __name__ == "__main__":
    main()
