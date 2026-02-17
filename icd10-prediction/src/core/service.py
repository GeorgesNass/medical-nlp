'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "FastAPI service layer for ICD10 prediction: model loading, inference endpoints and health checks."
'''

from __future__ import annotations

## Standard library imports
from pathlib import Path
from typing import Any, Dict, List, Optional

## Third-party imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib
import json
import numpy as np
from scipy import sparse

## Internal imports
from src.utils.logging_utils import get_logger
from src.core.config import AppConfig, build_config
from src.core.schema import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionItem,
    SinglePredictionRequest,
    SinglePredictionResponse,
)
from src.model.predict import predict_with_probabilities
from src.nlp.postprocess import select_top_k

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("service", log_file="service.log")

## ============================================================
## GLOBAL STATE
## ============================================================
## In-memory objects loaded once at startup for fast inference
MODEL: Optional[Any] = None
VECTORIZER: Optional[Any] = None
LABELS: Optional[List[str]] = None
CONFIG: Optional[AppConfig] = None

## ============================================================
## MODEL LOADING
## ============================================================
def _resolve_default_paths(config: AppConfig) -> Dict[str, Path]:
    """
        Resolve default artifact paths from configuration

        Args:
            config: Application configuration

        Returns:
            Dictionary containing default artifact paths
    """

    base = config.paths.artifacts_dir

    return {
        "model_path": base / "models" / "model.joblib",
        "vectorizer_path": base / "metadata" / "vectorizer.joblib",
        "labels_path": base / "metadata" / "labels.json",
    }

def _load_labels(labels_path: Path) -> List[str]:
    """
        Load labels list from JSON

        Args:
            labels_path: Path to labels.json

        Returns:
            List of ICD10 labels
    """

    payload = labels_path.read_text(encoding="utf-8")

    return list(json.loads(payload))

def load_artifacts(
    model_path: str | Path,
    vectorizer_path: str | Path,
    labels_path: str | Path,
) -> None:
    """
        Load model + vectorizer + labels into global state

        Args:
            model_path: Path to saved model
            vectorizer_path: Path to saved vectorizer
            labels_path: Path to labels.json
    """

    global MODEL, VECTORIZER, LABELS

    model_p = Path(model_path)
    vect_p = Path(vectorizer_path)
    labels_p = Path(labels_path)

    if not model_p.exists():
        raise FileNotFoundError(f"Model not found: {model_p}")

    if not vect_p.exists():
        raise FileNotFoundError(f"Vectorizer not found: {vect_p}")

    if not labels_p.exists():
        raise FileNotFoundError(f"Labels not found: {labels_p}")

    ## Load model and vectorizer (joblib)
    MODEL = joblib.load(model_p)
    VECTORIZER = joblib.load(vect_p)

    ## Load labels (json list)
    LABELS = _load_labels(labels_p)

    logger.info(
        "Artifacts loaded | model=%s | vectorizer=%s | labels=%d",
        model_p.name,
        vect_p.name,
        len(LABELS),
    )

## ============================================================
## FEATURE BUILDING
## ============================================================
def _vectorize_texts(texts: List[str]) -> sparse.spmatrix:
    """
        Vectorize texts using loaded vectorizer

        Args:
            texts: List of input texts

        Returns:
            Sparse feature matrix
    """

    if VECTORIZER is None:
        raise RuntimeError("Vectorizer not loaded")

    ## Transform only (vectorizer already trained)
    return VECTORIZER.transform(texts)

## ============================================================
## PREDICTION HELPERS
## ============================================================
def _top_k_from_probabilities(
    probabilities: np.ndarray,
    top_k: int,
) -> List[List[PredictionItem]]:
    """
        Convert probabilities to API-friendly top-k predictions

        Args:
            probabilities: Probability matrix
            top_k: Top-k value

        Returns:
            List of predictions per sample
    """

    if LABELS is None:
        raise RuntimeError("Labels not loaded")

    ## Use shared helper to compute top-k tuples
    topk = select_top_k(probabilities, LABELS, top_k=top_k)

    predictions: List[List[PredictionItem]] = []

    for sample in topk:
        preds = [PredictionItem(icd10_code=code, confidence=score) for code, score in sample]
        predictions.append(preds)

    return predictions

## ============================================================
## FASTAPI APP FACTORY
## ============================================================
def create_app() -> FastAPI:
    """
        Create FastAPI app and load artifacts at startup

        Returns:
            FastAPI application instance
    """

    global CONFIG
    CONFIG = build_config()

    app = FastAPI(title="ICD10 Prediction API", version="1.0.0")

    ## Load artifacts on startup
    @app.on_event("startup")
    def _startup() -> None:
        """
            Startup hook to load artifacts
        """

        paths = _resolve_default_paths(CONFIG)

        try:
            load_artifacts(
                model_path=paths["model_path"],
                vectorizer_path=paths["vectorizer_path"],
                labels_path=paths["labels_path"],
            )
        except Exception as exc:
            logger.error("Startup artifact loading failed: %s", str(exc))

    ## ========================================================
    ## HEALTHCHECK
    ## ========================================================
    @app.get("/icd10/health", response_model=HealthResponse)
    def healthcheck() -> HealthResponse:
        """
            Basic service healthcheck

            Returns:
                HealthResponse
        """

        model_loaded = MODEL is not None and VECTORIZER is not None and LABELS is not None

        return HealthResponse(
            status="ok",
            model_loaded=model_loaded,
            run_id=CONFIG.runtime.run_id if CONFIG else "unknown",
        )

    ## ========================================================
    ## MODEL INFO
    ## ========================================================
    @app.get("/icd10/models", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        """
            Return loaded model metadata

            Returns:
                ModelInfoResponse
        """

        if MODEL is None or LABELS is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return ModelInfoResponse(
            model_name=MODEL.__class__.__name__,
            model_type=MODEL.__class__.__name__,
            trained_on="unknown",
            labels_count=len(LABELS),
        )

    ## ========================================================
    ## SINGLE PREDICTION
    ## ========================================================
    @app.post("/icd10/predict", response_model=SinglePredictionResponse)
    def predict(request: SinglePredictionRequest) -> SinglePredictionResponse:
        """
            Predict ICD10 code for a single text input

            Args:
                request: Prediction request payload

            Returns:
                SinglePredictionResponse
        """

        if MODEL is None or VECTORIZER is None or LABELS is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        ## Vectorize text
        X = _vectorize_texts([request.text])

        ## Predict probabilities
        _, probs = predict_with_probabilities(MODEL, X)

        ## Convert to top-k predictions
        top_k = int(request.top_k or 5)
        preds = _top_k_from_probabilities(probs, top_k=top_k)[0]

        return SinglePredictionResponse(
            admission_id=request.admission_id,
            predictions=preds,
        )

    ## ========================================================
    ## BATCH PREDICTION
    ## ========================================================
    @app.post("/icd10/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
        """
            Predict ICD10 codes for a batch of texts

            Args:
                request: Batch prediction request

            Returns:
                BatchPredictionResponse
        """

        if MODEL is None or VECTORIZER is None or LABELS is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        ## Extract texts
        texts = [item.text for item in request.items]

        ## Vectorize texts
        X = _vectorize_texts(texts)

        ## Predict probabilities
        _, probs = predict_with_probabilities(MODEL, X)

        ## Convert to top-k predictions
        top_k = int(request.top_k or 5)
        all_preds = _top_k_from_probabilities(probs, top_k=top_k)

        results: List[SinglePredictionResponse] = []

        for item, preds in zip(request.items, all_preds, strict=False):
            results.append(
                SinglePredictionResponse(
                    admission_id=item.admission_id,
                    predictions=preds,
                )
            )

        return BatchPredictionResponse(results=results)

    return app

## ============================================================
## APP INSTANCE
## ============================================================
app = create_app()