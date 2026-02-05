'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "CLI entry point for Clinical NER: configuration loading and pipeline execution."
'''

from __future__ import annotations

## Standard library imports
import argparse
import sys
from pathlib import Path

## Core config and pipeline
from src.core.config import ProjectConfig
from src.pipeline import run_pipeline

## Centralized errors and logging
from src.core.errors import ClinicalNERError
from src.utils.logging_utils import get_logger


## Module-level logger
logger = get_logger(name="clinical_ner.main")


def build_arg_parser() -> argparse.ArgumentParser:
    """
        Build CLI argument parser for Clinical NER

        Returns:
            Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Clinical NER pipeline execution")

    ## Input options
    parser.add_argument(
        "--labeled-csv",
        type=str,
        default=None,
        help="Path to labeled CSV input",
    )
    parser.add_argument(
        "--unlabeled-texts",
        type=str,
        default=None,
        help="Path to folder containing raw .txt documents",
    )

    ## Output option
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to output CSV file",
    )

    ## Project root override
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Override project root directory",
    )

    return parser


def main() -> None:
    """
        Main CLI entry point

        Returns:
            None
    """
    ## Parse CLI arguments
    parser = build_arg_parser()
    args = parser.parse_args()

    ## Normalize project root as Path when provided
    project_root: Path | None = None
    if args.project_root:
        project_root = Path(args.project_root).expanduser().resolve()

    try:
        ## Load project configuration
        cfg = ProjectConfig.from_env(project_root=project_root)

        ## Run pipeline
        output_path = run_pipeline(
            cfg=cfg,
            labeled_csv_path=args.labeled_csv,
            unlabeled_texts_dir=args.unlabeled_texts,
            output_csv_path=args.output_csv,
        )

        logger.info("Output written to: %s", output_path)

    except ClinicalNERError as exc:
        ## Known application error
        logger.error("Clinical NER failed: %s", exc)
        sys.exit(1)

    except Exception:
        ## Unexpected error
        logger.exception("Unexpected error occurred")
        sys.exit(2)


if __name__ == "__main__":
    main()
