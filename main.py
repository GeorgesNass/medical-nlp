'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "CLI entry point for Clinical NER: configuration loading, validation and pipeline execution."
'''

from __future__ import annotations

## Standard library
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

## Core config and pipeline
from src.core.data_consistency import run_data_consistency
from src.core.config import ProjectConfig
from src.pipeline import run_pipeline

## Errors and logging
from src.core.errors import ClinicalNERError
from src.utils.logging_utils import get_logger

## ============================================================
## CONSTANTS
## ============================================================
APP_VERSION = "1.0.0"
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_PLATFORM_ERROR = 2

logger = get_logger(name="clinical_ner.main")

## ============================================================
## ARG PARSER
## ============================================================
def build_arg_parser() -> argparse.ArgumentParser:
    """
        Build CLI argument parser for Clinical NER

        Returns:
            Configured ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description="Clinical NER pipeline execution",
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
        help="Validate inputs without executing pipeline",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit",
    )

    ## Input
    parser.add_argument("--labeled-csv", type=str, default=None)
    parser.add_argument("--unlabeled-texts", type=str, default=None)

    ## Output
    parser.add_argument("--output-csv", type=str, default=None)

    ## Project root
    parser.add_argument("--project-root", type=str, default=None)

    return parser

## ============================================================
## VALIDATION
## ============================================================
def _validate_inputs(
    labeled_csv: Optional[str],
    unlabeled_texts: Optional[str],
) -> None:
    """
        Validate minimal input consistency

        Args:
            labeled_csv: Path to labeled CSV
            unlabeled_texts: Path to raw texts

        Raises:
            ValueError: If both inputs are missing
    """

    ## At least one input must be provided
    if not labeled_csv and not unlabeled_texts:
        raise ValueError(
            "At least one input must be provided: --labeled-csv or --unlabeled-texts"
        )

def _validate_runtime_config(project_root: Optional[Path]) -> dict:
    """
        Validate runtime environment

        Args:
            project_root: Optional project root

        Returns:
            Validation summary
    """

    return {
        "cwd": str(Path.cwd()),
        "project_root": str(project_root) if project_root else None,
        "python": sys.executable,
    }

## ============================================================
## EXECUTION SUMMARY
## ============================================================
def _build_summary(
    action: str,
    success: bool,
    start: float,
    details: Optional[dict] = None,
) -> dict:
    """
        Build execution summary

        Args:
            action: Executed action
            success: Execution status
            start: Start timestamp
            details: Optional details

        Returns:
            Summary dictionary
    """

    return {
        "action": action,
        "success": success,
        "duration_seconds": round(time.monotonic() - start, 3),
        "details": details or {},
    }

## ============================================================
## MAIN LOGIC
## ============================================================
def main() -> int:
    """
        Main CLI entry point

        Returns:
            Exit code
    """

    start_time = time.monotonic()

    parser = build_arg_parser()
    args = parser.parse_args()

    ## Normalize project root
    project_root: Optional[Path] = None
    if args.project_root:
        project_root = Path(args.project_root).expanduser().resolve()

    try:
        ## Validate runtime
        runtime_summary = _validate_runtime_config(project_root)

        if args.validate_config:
            logger.info("Config validation OK | %s", runtime_summary)
            logger.info("Summary | %s", _build_summary("validate-config", True, start_time))
            return EXIT_SUCCESS

        ## Validate inputs
        _validate_inputs(args.labeled_csv, args.unlabeled_texts)

        if args.dry_run:
            logger.info("Dry-run | pipeline would execute")
            logger.info(
                "Summary | %s",
                _build_summary("dry-run", True, start_time, runtime_summary),
            )
            return EXIT_SUCCESS

        ## Load config
        cfg = ProjectConfig.from_env(project_root=project_root)

        ## ============================================================
        ## DATA CONSISTENCY CHECK
        ## ============================================================
        if cfg.data_consistency.enabled:
            consistency_result = run_data_consistency(
                data={
                    "text": "ner_run",
                    "entities": [
                        {"start": 0, "end": 3, "label": "TEST"}
                    ],
                },
                strict=cfg.data_consistency.strict_mode,
            )

            logger.info(f"Consistency OK: {consistency_result['is_consistent']}")

            if not consistency_result["is_consistent"] and cfg.data_consistency.strict_mode:
                raise ClinicalNERError(
                    message="Data consistency failed before pipeline",
                    error_code="data_consistency_error",
                    details={"issues": consistency_result["issues"]},
                    origin="main",
                    http_status=400,
                    is_retryable=False,
                )

        ## Run pipeline
        output_path = run_pipeline(
            cfg=cfg,
            labeled_csv_path=args.labeled_csv,
            unlabeled_texts_dir=args.unlabeled_texts,
            output_csv_path=args.output_csv,
        )

        logger.info("Output written to: %s", output_path)

        logger.info(
            "Summary | %s",
            _build_summary(
                "run-pipeline",
                True,
                start_time,
                {"output": str(output_path)},
            ),
        )

        return EXIT_SUCCESS

    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user")
        logger.warning("Summary | %s", _build_summary("interrupt", False, start_time))
        return EXIT_FAILURE

    except ClinicalNERError as exc:
        logger.error("ClinicalNERError | %s", exc)
        logger.error("Summary | %s", _build_summary("error", False, start_time))
        return EXIT_PLATFORM_ERROR

    except Exception as exc:
        logger.exception("Unhandled exception: %s", exc)
        logger.error("Summary | %s", _build_summary("exception", False, start_time))
        return EXIT_FAILURE

## ============================================================
## ENTRY POINT
## ============================================================
if __name__ == "__main__":
    sys.exit(main())