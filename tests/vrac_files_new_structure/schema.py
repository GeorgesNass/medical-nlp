'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Pydantic schemas for ICD10 prediction API (request/response models)."
'''

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

## ============================================================
## BASE MODELS
## ============================================================
class HealthResponse(BaseModel):
    """
        Healthcheck response model

        Attributes:
            status: Service status string
            model_loaded: Whether a trained model is loaded
            run_id: Current runtime identifier
    """

    status: str
    model_loaded: bool
    run_id: str

## ============================================================
## PREDICTION REQUEST MODELS
## ============================================================
class SinglePredictionRequest(BaseModel):
    """
        Single document prediction request

        Attributes:
            admission_id: Unique hospital stay identifier
            text: Full document text content
            top_k: Optional number of top predictions to return
    """

    admission_id: str = Field(..., description="Hospital admission identifier")
    text: str = Field(..., description="Full clinical text content")
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top ICD10 predictions to return",
    )

class BatchPredictionRequest(BaseModel):
    """
        Batch prediction request

        Attributes:
            items: List of prediction requests
            top_k: Optional number of top predictions per item
    """

    items: List[SinglePredictionRequest]
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top ICD10 predictions per item",
    )

## ============================================================
## PREDICTION RESPONSE MODELS
## ============================================================
class PredictionItem(BaseModel):
    """
        Single prediction item

        Attributes:
            icd10_code: Predicted ICD10 code
            confidence: Prediction confidence score
    """

    icd10_code: str
    confidence: float

class SinglePredictionResponse(BaseModel):
    """
        Single prediction response

        Attributes:
            admission_id: Hospital admission identifier
            predictions: List of top predicted ICD10 codes
    """

    admission_id: str
    predictions: List[PredictionItem]

class BatchPredictionResponse(BaseModel):
    """
        Batch prediction response

        Attributes:
            results: List of single prediction responses
    """

    results: List[SinglePredictionResponse]

## ============================================================
## MODEL INFO
## ============================================================
class ModelInfoResponse(BaseModel):
    """
        Model metadata response

        Attributes:
            model_name: Name of the loaded model
            model_type: Model architecture type
            trained_on: Dataset identifier or timestamp
            labels_count: Number of distinct ICD10 labels
    """

    model_name: str
    model_type: str
    trained_on: str
    labels_count: int
