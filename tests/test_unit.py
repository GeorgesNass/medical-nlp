'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unit tests for lab_clustering: schema formatting, norms status computation, and clustering smoke tests."
'''

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import pytest
import numpy as np

from src.clustering.algorithms import run_clustering_algorithm
from src.clustering.preprocess import preprocess_dataset
from src.core.data_quality import run_data_quality
from src.core.data_consistency import run_data_consistency
from src.core.data_drift import run_data_drift
from src.core.config import AppConfig, build_config
from src.parser.check_norms import compute_status_from_norms
from src.parser.format_output import format_structured_output
from src.utils.io_utils import build_features, push_features, get_features
from src.utils.utils import normalize_clinical_text

## ============================================================
## TEST HELPERS
## ============================================================
class DummyClusteringParams:
    """
        Minimal test helper mimicking clustering parameters object

        Designed for clustering smoke tests without requiring a real Pydantic model

        Args:
            algorithm: Clustering algorithm name
            params: Dictionary of clustering parameters

        Returns:
            DummyClusteringParams instance
    """

    def __init__(self, algorithm: str, params: Dict[str, Any]) -> None:
        ## Store algorithm name
        self.algorithm = algorithm

        ## Store parameters dictionary
        self._params = params or {}

    def model_dump(self) -> Dict[str, Any]:
        """
            Return parameters as dictionary

            Returns:
                Parameters dictionary
        """

        return dict(self._params)

    def __repr__(self) -> str:
        """
            Return debug representation

            Returns:
                Readable string representation
        """

        return (
            f"DummyClusteringParams("
            f"algorithm={self.algorithm}, "
            f"params={self._params}"
            f")"
        )

## ============================================================
## FIXTURES
## ============================================================
@pytest.fixture()
def app_config(monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    """
        Build a minimal AppConfig for tests

        High-level workflow:
            1) Force local paths using env variables
            2) Build config instance

        Args:
            monkeypatch: Pytest monkeypatch fixture

        Returns:
            AppConfig instance
    """

    monkeypatch.setenv("DATA_DIR", "data")
    monkeypatch.setenv("LOGS_DIR", "logs")
    monkeypatch.setenv("ARTIFACTS_DIR", "artifacts")
    monkeypatch.setenv("RESOURCES_DIR", "artifacts/resources")
    monkeypatch.setenv("USE_GPU", "false")
    monkeypatch.setenv("RANDOM_SEED", "42")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "lab_clustering_test")

    return build_config()

@pytest.fixture
def dummy_clustering_params() -> DummyClusteringParams:
    """
        Provide default clustering parameters for smoke tests

        Returns:
            DummyClusteringParams instance
    """

    return DummyClusteringParams(
        algorithm="kmeans",
        params={"n_clusters": 2},
    )

@pytest.fixture
def dummy_clustering_params_factory():
    """
        Provide a factory to build custom clustering parameters

        Returns:
            Factory function creating DummyClusteringParams
    """

    def _factory(
        algorithm: str = "kmeans",
        params: Dict[str, Any] | None = None,
    ) -> DummyClusteringParams:
        """
            Build custom dummy clustering params

            Args:
                algorithm: Clustering algorithm name
                params: Optional clustering parameters

            Returns:
                DummyClusteringParams instance
        """

        return DummyClusteringParams(
            algorithm=algorithm,
            params=params or {"n_clusters": 2},
        )

    return _factory

## ============================================================
## TESTS: NORMS STATUS
## ============================================================
def test_compute_status_from_norms_low() -> None:
    """
        Verify status computation returns low

        High-level workflow:
            1) Compute status for a value below min
            2) Assert expected output

        Returns:
            None
    """

    status = compute_status_from_norms(value=1.0, norms_min=2.0, norms_max=5.0)
    
    assert status == "low"

def test_compute_status_from_norms_high() -> None:
    """
        Verify status computation returns high

        High-level workflow:
            1) Compute status for a value above max
            2) Assert expected output

        Returns:
            None
    """

    status = compute_status_from_norms(value=6.0, norms_min=2.0, norms_max=5.0)
    
    assert status == "high"

def test_compute_status_from_norms_normal() -> None:
    """
        Verify status computation returns normal

        High-level workflow:
            1) Compute status for a value within bounds
            2) Assert expected output

        Returns:
            None
    """

    status = compute_status_from_norms(value=3.0, norms_min=2.0, norms_max=5.0)
    
    assert status == "normal"

def test_compute_status_from_norms_unknown() -> None:
    """
        Verify status computation returns unknown

        High-level workflow:
            1) Compute status for missing value
            2) Assert expected output

        Returns:
            None
    """

    status = compute_status_from_norms(value=None, norms_min=2.0, norms_max=5.0)
    
    assert status == "unknown"

## ============================================================
## TESTS: OUTPUT FORMATTING
## ============================================================
def test_format_structured_output_schema(app_config: AppConfig) -> None:
    """
        Verify formatted structured output contains required columns

        High-level workflow:
            1) Create minimal extracted DataFrame
            2) Format with official schema
            3) Assert expected columns exist

        Args:
            app_config: AppConfig fixture

        Returns:
            None
    """

    df = pd.DataFrame(
        [
            {
                "analyzed_variable": "Sodium",
                "raw_data_entry": "Sodium 140 mmol/L",
                "structured_data_transform_value": 140.0,
                "structured_data_transform_metric": "mmol/l",
                "norms_min": 135.0,
                "norms_max": 145.0,
                "status": "normal",
            }
        ]
    )

    out = format_structured_output(
        df=df,
        source_file="example.txt",
        config=app_config,
    )

    required_cols = [
        "file",
        "gender",
        "sampling_time",
        "dates_dob",
        "dates_edition",
        "analysis_group",
        "analyzed_variable",
        "raw_data_entry",
        "structured_data_origin_value",
        "structured_data_origin_metric",
        "structured_data_transform_value",
        "structured_data_transform_metric",
        "norms_min",
        "norms_max",
        "norms_metric",
        "status",
        "Enfant",
        "Femme",
        "Homme",
        "Metric",
    ]

    for col in required_cols:
        assert col in out.columns

## ============================================================
## TESTS: PREPROCESS + CLUSTERING SMOKE
## ============================================================
def test_preprocess_and_cluster_smoke() -> None:
    """
        Basic smoke test for preprocessing and kmeans clustering

        High-level workflow:
            1) Create a small numeric dataset
            2) Preprocess it
            3) Run KMeans clustering
            4) Assert labels shape and basic fields exist

        Returns:
            None
    """

    df = pd.DataFrame(
        {
            "file": ["a", "b", "c", "d"],
            "feat1": [1.0, 1.2, 5.0, 5.2],
            "feat2": [0.9, 1.1, 4.9, 5.1],
        }
    )

    processed, meta = preprocess_dataset(
        df=df,
        preprocess_params={
            "impute_strategy": "median",
            "apply_pca": False,
        },
    )

    assert processed.shape[0] == 4
    assert meta["scaled"] is True

    params = DummyClusteringParams("kmeans", {"n_clusters": 2})
    result = run_clustering_algorithm(df=processed, clustering_params=params)

    assert "labels" in result
    assert len(result["labels"]) == 4
    assert result["n_clusters"] >= 1
    
## ============================================================
## DATA CONSISTENCY TESTS (CLUSTERING)
## ============================================================
def test_data_consistency_valid_clustering() -> None:
    """
        Validate correct clustering payload

        High-level workflow:
            1) Build valid text + embeddings
            2) Run data consistency
            3) Assert consistency is True

        Returns:
            None
    """

    data = {
        "text": "patient sodium",
        "embeddings": [[0.1, 0.2], [0.1, 0.2]],
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is True

def test_data_consistency_invalid_embeddings_dim() -> None:
    """
        Detect inconsistent embedding dimensions

        High-level workflow:
            1) Build embeddings with different sizes
            2) Run data consistency
            3) Assert consistency is False

        Returns:
            None
    """

    data = {
        "text": "test",
        "embeddings": [[0.1, 0.2], [0.1]],
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is False

def test_data_consistency_nan_embeddings() -> None:
    """
        Detect NaN values in embeddings

        High-level workflow:
            1) Build embeddings containing NaN
            2) Run data consistency
            3) Assert consistency is False

        Returns:
            None
    """

    data = {
        "text": "test",
        "embeddings": [[0.1, float("nan")]],
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is False

def test_data_consistency_empty_text() -> None:
    """
        Detect empty text input

        High-level workflow:
            1) Build payload with empty text
            2) Run data consistency
            3) Assert consistency is False

        Returns:
            None
    """

    data = {
        "text": "",
        "embeddings": [[0.1, 0.2]],
    }

    result = run_data_consistency(data=data)

    assert result["is_consistent"] is False
    
## ============================================================
## DATA QUALITY TESTS (CLUSTERING)
## ============================================================
def test_data_quality_valid_embeddings() -> None:
    """
        Validate correct embeddings
    """

    embeddings = [np.array([0.1, 0.2]), np.array([0.2, 0.3])]

    result = run_data_quality(embeddings=embeddings)

    assert result["is_valid"] is True

def test_data_quality_empty_embedding() -> None:
    """
        Detect empty embedding
    """

    embeddings = [np.array([])]

    result = run_data_quality(embeddings=embeddings)

    assert result["is_valid"] is False

def test_data_quality_nan_embedding() -> None:
    """
        Detect NaN in embedding
    """

    embeddings = [np.array([0.1, float("nan")])]

    result = run_data_quality(embeddings=embeddings)

    assert result["is_valid"] is False

def test_data_quality_anomaly_detection() -> None:
    """
        Detect abnormal embedding norms
    """

    embeddings = [
        np.array([0.1, 0.1]),
        np.array([100.0, 100.0]),
    ]

    result = run_data_quality(embeddings=embeddings, method="zscore")

    assert any(i["rule"] == "embedding_norm_anomaly" for i in result["issues"])

def test_data_quality_scoring() -> None:
    """
        Ensure score exists
    """

    embeddings = [np.array([0.1, 0.2])]

    result = run_data_quality(embeddings=embeddings)

    assert "score" in result

def test_data_quality_strict_mode() -> None:
    """
        Strict mode raises error
    """

    embeddings = [np.array([])]

    with pytest.raises(Exception):
        run_data_quality(embeddings=embeddings, strict=True)
        
## ============================================================
## DATA DRIFT TESTS (CLUSTERING)
## ============================================================
def test_data_drift_no_drift_clustering() -> None:
    """
        Validate no drift scenario

        High-level workflow:
            1) Create identical datasets
            2) Run drift detection
            3) Validate high score

        Returns:
            None
    """

    df_ref = pd.DataFrame({
        "cluster": [0, 0, 1, 1],
        "embedding": [[0.1, 0.2]] * 4,
        "text": ["a", "b", "c", "d"],
    })

    df_cur = df_ref.copy()

    result = run_data_drift(df_ref=df_ref, df_current=df_cur)

    assert result["drift_score"] >= 0.9
    assert result["errors"] == 0

def test_data_drift_detected_clustering() -> None:
    """
        Detect drift in clustering distribution

        High-level workflow:
            1) Create different cluster distributions
            2) Run drift detection
            3) Validate warnings

        Returns:
            None
    """

    df_ref = pd.DataFrame({
        "cluster": [0, 0, 0, 0],
        "embedding": [[0.1, 0.2]] * 4,
    })

    df_cur = pd.DataFrame({
        "cluster": [1, 1, 1, 1],
        "embedding": [[10.0, 20.0]] * 4,
    })

    result = run_data_drift(df_ref=df_ref, df_current=df_cur)

    assert result["drift_score"] < 1.0
    assert result["warnings"] > 0

def test_data_drift_empty_dataset() -> None:
    """
        Validate empty dataset handling

        High-level workflow:
            1) Use empty datasets
            2) Expect failure

        Returns:
            None
    """

    df_ref = pd.DataFrame()
    df_cur = pd.DataFrame()

    with pytest.raises(Exception):
        run_data_drift(df_ref=df_ref, df_current=df_cur)

def test_data_drift_strict_mode() -> None:
    """
        Validate strict mode

        High-level workflow:
            1) Create drift
            2) Enable strict mode
            3) Expect exception

        Returns:
            None
    """

    df_ref = pd.DataFrame({
        "cluster": [0],
    })

    df_cur = pd.DataFrame({
        "cluster": [1],
    })

    with pytest.raises(Exception):
        run_data_drift(df_ref=df_ref, df_current=df_cur, strict=True)
        
## ============================================================
## FEATURE ENGINEERING TESTS
## ============================================================
def test_normalize_clinical_text_basic() -> None:
    """
        Validate basic clinical text normalization

        High-level workflow:
            1) Provide raw text with noise
            2) Normalize text
            3) Assert expected cleaned output

        Returns:
            None
    """

    raw = "  Sodium µmol/L !!!  "
    normalized = normalize_clinical_text(raw)

    assert normalized == "sodium umol/l"
    
def test_text_feature_extraction() -> None:
    """
        Validate text feature extraction (length and tokens)

        High-level workflow:
            1) Normalize text
            2) Compute token and length features
            3) Assert expected values

        Returns:
            None
    """

    text = "Glucose 5.5 mmol/L"
    normalized = normalize_clinical_text(text)

    char_length = len(normalized)
    token_count = len(normalized.split())

    assert char_length > 0
    assert token_count >= 2    

## ============================================================
## FEATURE STORE TESTS
## ============================================================
def test_build_features_basic() -> None:
    """
        Validate feature engineering output structure

        Design:
            - Ensure normalized text feature exists
            - Ensure length feature exists
            - Ensure numeric feature exists

        Returns:
            None
    """

    row = {"text": "Hello", "value": 10}

    ## Build features
    features = build_features(row)

    ## Assertions
    assert "text_normalized" in features
    assert "text_length" in features
    assert "value_scaled" in features
    
def test_feature_store_roundtrip() -> None:
    """
        Validate feature store roundtrip (Redis / Feast)

        Design:
            - Store features
            - Retrieve features
            - Ensure non-empty result

        Returns:
            None
    """

    entity_id = "test_entity_fs"
    features = {"a": 1, "b": 2}

    ## Store features
    push_features(entity_id, features)

    ## Retrieve features
    retrieved = get_features(entity_id)

    ## Assertions
    assert isinstance(retrieved, dict)
    assert len(retrieved) > 0
    
def test_feature_engineering_pipeline_integration() -> None:
    """
        Validate full feature engineering + feature store integration

        Design:
            - Build features
            - Store them
            - Retrieve them
            - Validate pipeline consistency

        Returns:
            None
    """

    row = {"text": "Sample Data", "num": 5}

    ## Build features
    features = build_features(row)

    ## Store features
    push_features("entity_test_fs", features)

    ## Retrieve features
    retrieved = get_features("entity_test_fs")

    ## Assertions
    assert isinstance(retrieved, dict)
    assert len(retrieved) > 0