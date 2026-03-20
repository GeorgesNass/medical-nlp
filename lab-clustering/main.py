'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main CLI entry point for lab-clustering: UI, API, chat, loop, ingest and evaluation."
'''

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn

from src.core.config import config
from src.core.errors import AutonomousAIPlatformError, DependencyError, log_structured_error
from src.core.streamlit_app import run_streamlit_app
from src.pipeline import run_chat, run_evaluation, run_loop
from src.utils.env_utils import _get_env_int
from src.utils.logging_utils import get_logger
from src.utils.safe_utils import _safe_json, _safe_str
from src.utils.validation_utils import _must_be_non_empty

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
## ARG PARSER
## ============================================================
def _build_parser() -> argparse.ArgumentParser:
    """
        Build CLI argument parser

        Returns:
            Configured ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description="Lab clustering CLI launcher",
        add_help=True,
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {APP_VERSION}")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-config", action="store_true")

    ## Execution
    parser.add_argument("--run-ui", action="store_true")
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--ingest", action="store_true")

    ## Runtime
    parser.add_argument("--prefer-local", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--export", action="store_true")

    ## Inputs
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--answer", type=str, default="")
    parser.add_argument("--input-dir", type=str, default="")

    ## API
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")

    parser.add_argument("--internal-ui", action="store_true", help=argparse.SUPPRESS)

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

def _print_json(payload: Dict[str, Any]) -> None:
    """
        Print structured JSON payload

        Args:
            payload: Dictionary payload

        Returns:
            None
    """

    logger.info("CLI_OUTPUT | %s", _safe_json(payload))

def _run_streamlit_subprocess() -> None:
    """
        Launch Streamlit UI subprocess

        Raises:
            DependencyError if Streamlit missing
    """

    main_path = Path(__file__).resolve()
    cmd = ["streamlit", "run", str(main_path), "--", "--internal-ui"]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise DependencyError(
            message="streamlit is not installed",
            error_code="dependency_error",
            details={"cmd": cmd},
            origin="main",
            cause=exc,
            http_status=500,
            is_retryable=False,
        ) from exc

def _run_api(host: str, port: int, reload: bool) -> None:
    """
        Start FastAPI server

        Args:
            host: API host
            port: API port
            reload: Dev reload flag
    """

    uvicorn.run("src.core.mcp_server:app", host=host, port=int(port), reload=bool(reload))

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
        ## Validate config
        if args.validate_config:
            logger.info("Config OK")
            logger.info("Summary | %s", _build_summary("validate-config", True, start_time))
            return EXIT_SUCCESS

        ## Internal UI
        if args.internal_ui:
            run_streamlit_app()
            return EXIT_SUCCESS

        if not any([args.run_ui, args.run_api, args.chat, args.loop, args.evaluate, args.ingest]):
            parser.print_help()
            return EXIT_SUCCESS

        if args.dry_run:
            logger.info("Dry-run | no execution")
            logger.info("Summary | %s", _build_summary("dry-run", True, start_time))
            return EXIT_SUCCESS

        prefer_local = bool(args.prefer_local)
        use_gpu = bool(args.use_gpu)

        ## UI
        if args.run_ui:
            _run_streamlit_subprocess()
            return EXIT_SUCCESS

        ## API
        if args.run_api:
            _run_api(args.host, args.port, args.reload)
            return EXIT_SUCCESS

        ## Chat
        if args.chat:
            text = _must_be_non_empty(args.prompt, "prompt")
            result = run_chat(text, prefer_local=prefer_local, use_gpu=use_gpu)
            _print_json(result)

        ## Loop
        if args.loop:
            text = _must_be_non_empty(args.prompt, "prompt")
            result = run_loop(text, prefer_local=prefer_local, use_gpu=use_gpu, export=bool(args.export))
            _print_json(result)

        ## Evaluate
        if args.evaluate:
            q = _must_be_non_empty(args.query, "query")
            a = _must_be_non_empty(args.answer, "answer")
            report = run_evaluation(
                q,
                a,
                use_llm_judge=True,
                prefer_local=prefer_local,
                use_gpu=use_gpu,
                export=bool(args.export),
            )
            _print_json(report)

        ## Ingest
        if args.ingest:
            from src.orchestrator.retrieval import ingest_folder

            root = Path(args.input_dir).expanduser().resolve() if args.input_dir else config.paths.data_raw_dir

            result = ingest_folder(
                folder=str(root),
                prefer_local=prefer_local,
                use_gpu=use_gpu,
            )
            _print_json(result)

        logger.info("Summary | %s", _build_summary("run", True, start_time))
        return EXIT_SUCCESS

    except KeyboardInterrupt:
        logger.warning("Interrupted")
        logger.warning("Summary | %s", _build_summary("interrupt", False, start_time))
        return EXIT_FAILURE

    except AutonomousAIPlatformError as exc:
        log_structured_error(exc, request=None, include_traceback=False)

        payload = exc.to_payload()

        logger.error(
            "CLI_ERROR | code=%s | message=%s | details=%s",
            payload.error_code,
            payload.message,
            _safe_json(payload.details),
        )

        return EXIT_PLATFORM_ERROR

    except Exception as exc:
        logger.error(
            "CLI_UNHANDLED_EXCEPTION | type=%s | message=%s",
            exc.__class__.__name__,
            _safe_str(exc),
        )
        return EXIT_FAILURE

## ============================================================
## ENTRYPOINT
## ============================================================
if __name__ == "__main__":
    sys.exit(main())