'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Feature Store + Feature Engineering utilities (Redis / Feast)."
'''

from __future__ import annotations

from typing import Dict, Any
import os
import pandas as pd

import redis

try:
    from feast import FeatureStore
except Exception:
    FeatureStore = None

from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("io_utils")

## ============================================================
## FEATURE STORE CONFIG
## ============================================================
FEATURE_STORE_MODE = os.getenv("FEATURE_STORE_MODE", "redis")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "./feature_repo")

## Redis client (local mode)
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True
)

## ============================================================
## FEATURE ENGINEERING
## ============================================================
def build_features(row: Dict[str, Any]) -> Dict[str, Any]:
    """
        Build structured features from input row

        Design:
            - Lightweight feature engineering
            - Compatible with all pipelines
            - No external ML dependency

        Args:
            row: Input dictionary

        Returns:
            Feature dictionary
    """

    features: Dict[str, Any] = {}

    for key, value in row.items():

        ## Convert safely to string
        value_str = str(value)

        ## TEXT FEATURES
        if isinstance(value, str):

            ## normalize text
            normalized = value_str.lower()

            ## add features
            features[f"{key}_normalized"] = normalized
            features[f"{key}_length"] = len(normalized)

        ## NUMERIC FEATURES
        if isinstance(value, (int, float)):

            ## basic scaling (identity here)
            features[f"{key}_scaled"] = value

    logger.debug("Features built | keys=%s", list(features.keys()))

    return features

## ============================================================
## FEATURE STORE (REDIS + FEAST)
## ============================================================
def push_features(entity_id: str, features: Dict[str, Any]) -> None:
    """
        Store features in feature store

        Design:
            - Redis for local mode
            - Feast for production mode
            - Controlled via FEATURE_STORE_MODE

        Args:
            entity_id: Unique identifier
            features: Feature dictionary

        Returns:
            None
    """

    if FEATURE_STORE_MODE == "redis":

        ## Redis storage
        redis_client.hset(entity_id, mapping=features)

        logger.info("Features stored in Redis | entity_id=%s", entity_id)

    else:

        ## Feast storage
        if FeatureStore is None:
            raise ImportError("Feast is not installed")

        store = FeatureStore(repo_path=FEAST_REPO_PATH)

        df = pd.DataFrame([{**features, "entity_id": entity_id}])

        store.write_to_online_store(df)

        logger.info("Features stored in Feast | entity_id=%s", entity_id)

def get_features(entity_id: str) -> Dict[str, Any]:
    """
        Retrieve features from feature store

        Design:
            - Unified interface (Redis / Feast)
            - Used in training and inference

        Args:
            entity_id: Unique identifier

        Returns:
            Feature dictionary
    """

    if FEATURE_STORE_MODE == "redis":

        ## Redis fetch
        result = redis_client.hgetall(entity_id)

        logger.info("Features retrieved from Redis | entity_id=%s", entity_id)

        return result

    else:

        if FeatureStore is None:
            raise ImportError("Feast is not installed")

        store = FeatureStore(repo_path=FEAST_REPO_PATH)

        result = store.get_online_features(
            features=["features:*"],
            entity_rows=[{"entity_id": entity_id}]
        ).to_dict()

        logger.info("Features retrieved from Feast | entity_id=%s", entity_id)

        return result