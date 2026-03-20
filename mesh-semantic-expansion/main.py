'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main entry point for the MeSH Semantic Expansion API (FastAPI) and routes registration."
'''

from __future__ import annotations

import argparse
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Optional

import uvicorn
from fastapi import FastAPI

from src.core.config import get_settings
from src.service.routes_expand import router as expand_router
from src.service.routes_mesh import router as mesh_router
from src.utils.logging_utils import get_logger

## ============================================================
## CONSTANTS
## ============================================================
APP_VERSION = "1.0.0"
EXIT_SUCCESS = 0
EXIT_FAILURE = 1

## ============================================================
## LOGGER INITIALIZATION
## ============================================================
logger = get_logger("main")

## ============================================================
## APPLICATION SETTINGS
## ============================================================
settings = get_settings()

## ============================================================
## LIFESPAN EVENTS
## ============================================================
@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """
        Run startup and shutdown lifecycle hooks

        Yields:
            Control back to FastAPI runtime
    """

    logger.info("Starting MeSH Semantic Expansion API.")
    logger.info("Environment: %s", settings.environment)
    yield
    logger.info("Shutting down MeSH Semantic Expansion API.")

## ============================================================
## FASTAPI APP INITIALIZATION
## ============================================================
app = FastAPI(
    title="MeSH Semantic Expansion API",
    description=(
        "API for downloading, querying and extending MeSH vocabulary "
        "using medical documents, embeddings and NLP pipelines."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)

## ============================================================
## ROUTERS REGISTRATION
## ============================================================
app.include_router(mesh_router, prefix="/mesh", tags=["MeSH"])
app.include_router(expand_router, prefix="/expand", tags=["Semantic Expansion"])

## ============================================================
## ROUTE: HEALTHCHECK
## ============================================================
@app.get("/healthcheck")
def healthcheck() -> Dict[str, str]:
    """
        Return service health status

        Returns:
            Dictionary with service status and version
    """

    logger.debug("Healthcheck endpoint called")

    return {
        "status": "ok",
        "service": "mesh_semantic_expansion",
        "version": settings.app_version,
    }

## ============================================================
## CLI PARSER
## ============================================================
def _build_parser() -> argparse.ArgumentParser:
    """
        Build CLI parser for API launcher

        Returns:
            ArgumentParser instance
    """

    parser = argparse.ArgumentParser(
        description="Run MeSH Semantic Expansion API",
        add_help=True,
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {APP_VERSION}")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-config", action="store_true")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")

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
            Summary dictionary
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
        Main entrypoint for API launcher

        Returns:
            Exit code
    """

    start_time = time.monotonic()
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.validate_config:
            logger.info("Config OK | env=%s version=%s", settings.environment, settings.app_version)
            logger.info("Summary | %s", _build_summary("validate-config", True, start_time))
            return EXIT_SUCCESS

        if args.dry_run:
            logger.info("Dry-run | host=%s port=%s reload=%s", args.host, args.port, args.reload)
            logger.info("Summary | %s", _build_summary("dry-run", True, start_time))
            return EXIT_SUCCESS

        logger.info(
            "Starting API | host=%s port=%d reload=%s",
            args.host,
            args.port,
            args.reload,
        )

        uvicorn.run(
            "main:app",
            host=args.host,
            port=int(args.port),
            reload=bool(args.reload),
        )

        logger.info("Summary | %s", _build_summary("run-api", True, start_time))
        return EXIT_SUCCESS

    except KeyboardInterrupt:
        logger.warning("Interrupted")
        logger.warning("Summary | %s", _build_summary("interrupt", False, start_time))
        return EXIT_FAILURE

    except Exception as exc:
        logger.exception("Unhandled error: %s", exc)
        logger.error("Summary | %s", _build_summary("error", False, start_time))
        return EXIT_FAILURE

## ============================================================
## ENTRYPOINT
## ============================================================
if __name__ == "__main__":
    sys.exit(main())