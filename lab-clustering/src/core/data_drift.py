'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data drift detection for clustering: clusters, embeddings, distances and Evidently reporting."
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger
from src.utils.drift_utils import (
    compute_ks_test,
    compute_chi2_test,
    compute_text_stats,
    compute_embedding_stats,
    generate_evidently_report,
)

try:
    from src.core.errors import ValidationError, DataError
except Exception:
    ValidationError = ValueError
    DataError = RuntimeError

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("data_drift")

## ============================================================
## ISSUE HANDLING
## ============================================================
def _create_issue(
    rule: str,
    level: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
        Create standardized issue object

        High-level workflow:
            1) Build issue dictionary
            2) Attach optional metadata
            3) Return structured issue

        Args:
            rule: Rule name
            level: Severity level
            message: Description of the issue
            details: Optional metadata

        Returns:
            Issue dictionary
    """

    return {
        "rule": rule,
        "level": level,
        "message": message,
        "details": details or {},
    }

def _add_issue(
    issues: List[Dict[str, Any]],
    rule: str,
    level: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
        Append issue and log it

        High-level workflow:
            1) Create issue
            2) Append to list
            3) Log message

        Args:
            issues: Issue container
            rule: Rule name
            level: Severity level
            message: Description
            details: Optional metadata

        Returns:
            None
    """

    issue = _create_issue(rule, level, message, details)
    issues.append(issue)

    if level == "error":
        logger.error(message)
    else:
        logger.warning(message)

## ============================================================
## DISTANCE FEATURES
## ============================================================
def _compute_distance_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
        Compute simple distance-based features

        High-level workflow:
            1) Extract embedding vectors
            2) Compute pairwise norms (approx)
            3) Build distance proxy metrics

        Args:
            df: Input dataset

        Returns:
            DataFrame with distance statistics
    """

    data: Dict[str, pd.Series] = {}

    if "embedding" in df.columns:
        emb = df["embedding"].apply(
            lambda x: np.array(x) if isinstance(x, (list, tuple)) else np.array([])
        )

        norms = emb.apply(lambda x: np.linalg.norm(x) if x.size else 0.0)

        data["distance_proxy"] = norms

    return pd.DataFrame(data)

## ============================================================
## DRIFT DETECTION
## ============================================================
def _detect_numeric_drift(
    ref: pd.Series,
    cur: pd.Series,
    column: str,
    threshold: float,
    issues: List[Dict[str, Any]],
) -> float:
    """
        Detect numeric drift using KS test

        High-level workflow:
            1) Compute KS test
            2) Compare p-value to threshold
            3) Log issue if drift

        Args:
            ref: Reference series
            cur: Current series
            column: Column name
            threshold: p-value threshold
            issues: Issue container

        Returns:
            p_value
    """

    stat, p_value = compute_ks_test(ref, cur)

    if p_value < threshold:
        _add_issue(
            issues,
            "drift_numeric",
            "warning",
            f"Drift detected in {column}",
            {"p_value": float(p_value)},
        )

    return float(p_value)

def _detect_categorical_drift(
    ref: pd.Series,
    cur: pd.Series,
    column: str,
    threshold: float,
    issues: List[Dict[str, Any]],
) -> float:
    """
        Detect categorical drift using Chi-square test

        High-level workflow:
            1) Compute Chi2 test
            2) Compare p-value
            3) Log issue if drift

        Args:
            ref: Reference series
            cur: Current series
            column: Column name
            threshold: p-value threshold
            issues: Issue container

        Returns:
            p_value
    """

    stat, p_value = compute_chi2_test(ref, cur)

    if p_value < threshold:
        _add_issue(
            issues,
            "drift_categorical",
            "warning",
            f"Drift detected in {column}",
            {"p_value": float(p_value)},
        )

    return float(p_value)

## ============================================================
## MAIN ENTRYPOINT
## ============================================================
def run_data_drift(
    df_ref: pd.DataFrame,
    df_current: pd.DataFrame,
    p_value_threshold: float = 0.05,
    strict: bool = False,
) -> Dict[str, Any]:
    """
        Run data drift detection for clustering outputs

        High-level workflow:
            1) Validate datasets
            2) Detect drift on clusters
            3) Detect drift on embeddings
            4) Detect drift on distances
            5) Detect drift on text features
            6) Compute global drift score
            7) Generate Evidently report

        Args:
            df_ref: Reference dataset
            df_current: Current dataset
            p_value_threshold: Statistical threshold
            strict: Raise error if drift detected

        Returns:
            Drift result dictionary
    """

    issues: List[Dict[str, Any]] = []

    try:
        if df_ref.empty or df_current.empty:
            raise ValidationError("Empty datasets")

        drift_flags: List[bool] = []

        ## CLUSTER DISTRIBUTION
        if "cluster" in df_ref.columns and "cluster" in df_current.columns:
            p_value = _detect_categorical_drift(
                df_ref["cluster"],
                df_current["cluster"],
                "cluster",
                p_value_threshold,
                issues,
            )
            drift_flags.append(p_value < p_value_threshold)

        ## EMBEDDING DRIFT
        ref_emb = compute_embedding_stats(df_ref)
        cur_emb = compute_embedding_stats(df_current)

        for col in ref_emb.columns:
            if col in cur_emb.columns:
                p_value = _detect_numeric_drift(
                    ref_emb[col],
                    cur_emb[col],
                    col,
                    p_value_threshold,
                    issues,
                )
                drift_flags.append(p_value < p_value_threshold)

        ## DISTANCE DRIFT
        ref_dist = _compute_distance_stats(df_ref)
        cur_dist = _compute_distance_stats(df_current)

        for col in ref_dist.columns:
            if col in cur_dist.columns:
                p_value = _detect_numeric_drift(
                    ref_dist[col],
                    cur_dist[col],
                    col,
                    p_value_threshold,
                    issues,
                )
                drift_flags.append(p_value < p_value_threshold)

        ## TEXT DRIFT
        ref_text = compute_text_stats(df_ref)
        cur_text = compute_text_stats(df_current)

        for col in ref_text.columns:
            if col in cur_text.columns:
                p_value = _detect_numeric_drift(
                    ref_text[col],
                    cur_text[col],
                    col,
                    p_value_threshold,
                    issues,
                )
                drift_flags.append(p_value < p_value_threshold)

        ## GLOBAL SCORE
        drift_score = 1.0 - (sum(drift_flags) / len(drift_flags)) if drift_flags else 1.0

        result = {
            "is_drift_ok": True,
            "errors": 0,
            "warnings": len(issues),
            "drift_score": drift_score,
            "issues": issues,
        }

        ## EVIDENTLY
        try:
            report = generate_evidently_report(df_ref, df_current)
            result["evidently_report"] = report
        except Exception as e:
            logger.warning(f"Evidently failed: {e}")

        if strict and drift_score < 1.0:
            raise ValidationError("Drift detected")

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(exc)
        raise DataError("Drift pipeline failed")