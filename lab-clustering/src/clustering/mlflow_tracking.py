'''
__author__ = "Olivia Tortosa"
__copyright__ = None
__credits__ = ["Olivia Tortosa", "Georges Nassopoulos"]
__version__ = "1.0.0"
__maintainer__ = "Georges Nassopoulos"
__email__ = "olivia.tortosa@gmail.com", "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "MLflow tracking helpers: experiment setup, parameter logging, metrics logging and artifact registration"
'''

from __future__ import annotations

from typing import Any, Dict, Optional

import mlflow

from src.core.config import AppConfig
from src.core.errors import MlflowTrackingError
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

## ============================================================
## INTERNAL HELPERS
## ============================================================
def _setup_experiment(config: AppConfig) -> None:
    """
        Configure MLflow tracking URI and experiment

        High-level workflow:
            1) Set tracking URI
            2) Set or create experiment

        Args:
            config: AppConfig instance
    """

    ## Configure tracking URI
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)

    ## Set experiment name
    mlflow.set_experiment(config.mlflow.experiment_name)

## ============================================================
## PUBLIC API
## ============================================================
def track_clustering_run(
    clustering_params: Any,
    preprocess_params: Dict[str, Any],
    metrics: Dict[str, float],
    config: AppConfig,
) -> Optional[str]:
    """
        Track clustering run in MLflow

        High-level workflow:
            1) Setup MLflow experiment
            2) Start run
            3) Log parameters
            4) Log metrics
            5) Return MLflow run_id

        Args:
            clustering_params: Clustering parameter object
            preprocess_params: Preprocessing parameter dictionary
            metrics: Metrics dictionary
            config: AppConfig instance

        Returns:
            MLflow run_id or None
    """

    try:
        ## Setup experiment context
        _setup_experiment(config)

        ## Start MLflow run
        with mlflow.start_run() as run:

            run_id = run.info.run_id

            ## Log clustering parameters
            mlflow.log_param("algorithm", clustering_params.algorithm)

            for key, value in clustering_params.model_dump().items():
                mlflow.log_param(f"clustering_{key}", value)

            ## Log preprocessing parameters
            for key, value in preprocess_params.items():
                mlflow.log_param(f"preprocess_{key}", value)

            ## Log metrics
            for metric_name, metric_value in metrics.items():
                try:
                    mlflow.log_metric(metric_name, float(metric_value))
                except Exception:
                    continue

            logger.info(
                "MLflow run tracked | run_id=%s | experiment=%s",
                run_id,
                config.mlflow.experiment_name,
            )

            return run_id

    except Exception as exc:
        logger.error("MLflow tracking failed | error=%s", str(exc))
        logger.debug("Traceback:", exc_info=True)

        raise MlflowTrackingError(
            message="MLflow tracking failed",
            details={"error": str(exc)},
        )