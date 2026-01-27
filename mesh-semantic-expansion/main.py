'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main entry point for the MeSH Semantic Expansion API (FastAPI) and routes registration."
'''

from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict

from fastapi import FastAPI

from src.core.config import get_settings
from src.service.routes_expand import router as expand_router
from src.service.routes_mesh import router as mesh_router
from src.utils.logging_utils import get_logger

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
    """Run startup and shutdown tasks for the FastAPI application.

    Yields:
        None: Control back to FastAPI while the application is running.
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
## ROUTE: HEALTH CHECK
## ============================================================

@app.get("/healthcheck")
def healthcheck() -> Dict[str, str]:
    """Return service status information.

    Returns:
        Dict[str, str]: A dictionary containing the service status and version.
    """

    logger.debug("Healthcheck endpoint called.")
    return {
        "status": "ok",
        "service": "mesh_semantic_expansion",
        "version": settings.app_version,
    }
