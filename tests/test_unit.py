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

from src.clustering.algorithms import run_clustering_algorithm
from src.clustering.preprocess import preprocess_dataset
from src.core.config import AppConfig, build_config
from src.parser.check_norms import compute_status_from_norms
from src.parser.format_output import format_structured_output

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