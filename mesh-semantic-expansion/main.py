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
import pandas as pd
from pathlib import Path
from typing import AsyncIterator, Dict, Optional

import uvicorn
from fastapi import FastAPI

from src.core.config import get_config, get_settings
from src.core.data_consistency import run_data_consistency
from src.core.data_quality import run_data_quality
from src.core.data_drift import run_data_drift
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

    ## DATA CONSISTENCY CHECK
    data = {
        "text": "startup_check",
        "embeddings": [0.1] * settings.data_consistency.min_embedding_dim,
    }

    consistency_result = run_data_consistency(
        data=data,
        strict=settings.data_consistency.strict_mode,
    )

    logger.info(f"Consistency OK: {consistency_result['is_consistent']}")

    if not consistency_result["is_consistent"] and settings.data_consistency.strict_mode:
        raise RuntimeError("Data consistency failed at startup")

$    ## DATA QUALITY CHECK
    if settings.runtime.anomaly_detection_enabled:

        quality_result = run_data_quality(
            terms=["startup_check"],
            scores=[0.5],
            method=settings.runtime.anomaly_method,
            z_threshold=settings.runtime.z_threshold,
            iqr_multiplier=settings.runtime.iqr_multiplier,
            strict=settings.runtime.anomaly_strict_mode,
        )

        logger.info(f"Data quality score: {quality_result['score']}")
        
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
    parser.add_argument("--mode", type=str, default="", help="Optional mode (e.g. drift)")
    parser.add_argument("--ref", type=str, default="", help="Reference dataset for drift")
    parser.add_argument("--current", type=str, default="", help="Current dataset for drift")
    parser.add_argument("--features", action="store_true", help="Enable feature engineering pipeline",)
    
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
    config = get_config()
 
    try:
    
        ## Feature engineering CLI toggle
        if args.features:
            config.feature_engineering.enabled = True
            logger.info("Feature engineering enabled via CLI")
        
        if config.feature_engineering.enabled:
            logger.info("Feature engineering pipeline is ACTIVE")   
            
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

        ## DATA CONSISTENCY CHECK
        data = {
            "text": "api_launch",
            "embeddings": [0.1] * settings.data_consistency.min_embedding_dim,
        }

        consistency_result = run_data_consistency(
            data=data,
            strict=settings.data_consistency.strict_mode,
        )

        logger.info(f"Consistency OK: {consistency_result['is_consistent']}")

        if not consistency_result["is_consistent"] and settings.data_consistency.strict_mode:
            raise RuntimeError("Data consistency failed at API launch")

        ## DATA QUALITY CHECK
        if settings.runtime.anomaly_detection_enabled:

            quality_result = run_data_quality(
                terms=["api_launch"],
                scores=[0.5],
                method=settings.runtime.anomaly_method,
                z_threshold=settings.runtime.z_threshold,
                iqr_multiplier=settings.runtime.iqr_multiplier,
                strict=settings.runtime.anomaly_strict_mode,
            )

            logger.info(f"Data quality score: {quality_result['score']}")

        ## DATA DRIFT CHECK
        if args.mode == "drift":
            """
                Run data drift detection for MeSH datasets

                High-level workflow:
                    1) Validate reference and current dataset paths
                    2) Load both datasets from CSV
                    3) Run drift detection pipeline
                    4) Log drift score and Evidently report
                    5) Return success without starting API

                Args:
                    None

                Returns:
                    EXIT_SUCCESS if drift execution succeeds
            """

            if not args.ref or not args.current:
                raise ValueError("Drift mode requires --ref and --current")

            ref_path = Path(args.ref)
            cur_path = Path(args.current)

            if not ref_path.exists() or not cur_path.exists():
                raise ValueError("Drift datasets not found")

            df_ref = pd.read_csv(ref_path)
            df_cur = pd.read_csv(cur_path)

            drift_result = run_data_drift(
                df_ref=df_ref,
                df_current=df_cur,
                strict=settings.runtime.drift_strict_mode,
            )

            logger.info("Drift score | %s", drift_result["drift_score"])

            if "evidently_report" in drift_result:
                logger.info("Evidently report | %s", drift_result["evidently_report"])

            if drift_result["errors"] > 0:
                raise RuntimeError("Data drift detected")

            logger.info(
                "Summary | %s",
                _build_summary(
                    "drift",
                    True,
                    start_time,
                    {"drift_score": drift_result["drift_score"]},
                ),
            )

            return EXIT_SUCCESS
            
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