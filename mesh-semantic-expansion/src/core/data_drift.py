'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data drift detection for mesh-semantic-expansion: MeSH concepts, expansions, embeddings and Evidently reporting."
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.config import get_config
from src.nlp.preprocess import normalize_medical_text
from src.utils.logging_utils import get_logger
from src.utils.drift_utils import (
    compute_ks_test,
    compute_chi2_test,
    compute_text_stats,
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
            1) Create standardized issue object
            2) Append issue to issues list
            3) Log message depending on severity level

        Args:
            issues: Issue container
            rule: Rule name
            level: Severity level (warning | error)
            message: Description of the issue
            details: Optional metadata

        Returns:
            None
    """

    issue = _create_issue(rule, level, message, details)
    issues.append(issue)

    if level == "error":
        logger.error(f"{rule} - {message}")
    else:
        logger.warning(f"{rule} - {message}")

## ============================================================
## EMBEDDING STATS
## ============================================================
def _compute_embedding_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
        Compute embedding statistics

        High-level workflow:
            1) Extract embedding vectors
            2) Convert to numpy arrays
            3) Compute norm and mean value

        Args:
            df: Input dataset

        Returns:
            DataFrame with embedding statistics
    """

    data: Dict[str, pd.Series] = {}

    if "embedding" in df.columns:
        config = get_config()

        emb = df["embedding"].apply(
            lambda x: np.array(x) if isinstance(x, (list, tuple)) else np.array([])
        )

        if config.feature_engineering.enabled:
            emb = emb.apply(lambda x: x / np.linalg.norm(x) if x.size and np.linalg.norm(x) > 0 else x)
            logger.debug("Feature engineering drift preprocessing applied")
            
        data["embedding_norm"] = emb.apply(
            lambda x: np.linalg.norm(x) if x.size else 0.0
        )

        data["embedding_mean"] = emb.apply(
            lambda x: float(np.mean(x)) if x.size else 0.0
        )

    return pd.DataFrame(data)

## ============================================================
## EXPANSION STATS
## ============================================================
def _compute_expansion_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
        Compute semantic expansion statistics

        High-level workflow:
            1) Extract expansions column
            2) Compute number of generated terms
            3) Compute average length of expansions

        Args:
            df: Input dataset

        Returns:
            DataFrame with expansion statistics
    """

    data: Dict[str, pd.Series] = {}

    if "expansions" in df.columns:
        expansions = df["expansions"]

        data["expansion_count"] = expansions.apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        data["expansion_avg_length"] = expansions.apply(
            lambda x: np.mean([len(str(t)) for t in x]) if isinstance(x, list) and x else 0
        )

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
            2) Compare p-value with threshold
            3) Add warning if drift detected

        Args:
            ref: Reference series
            cur: Current series
            column: Column name
            threshold: Statistical p-value threshold
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
            1) Compute Chi-square test
            2) Compare p-value with threshold
            3) Add warning if drift detected

        Args:
            ref: Reference series
            cur: Current series
            column: Column name
            threshold: Statistical p-value threshold
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
        Run data drift detection for MeSH semantic expansion

        High-level workflow:
            1) Validate datasets
            2) Detect drift on MeSH concepts
            3) Detect drift on semantic expansions
            4) Detect drift on embeddings
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
            raise ValidationError("Empty datasets provided")

        ## Feature engineering preprocessing
        config = get_config()

        if config.feature_engineering.enabled:

            if "mesh_term" in df_ref.columns:
                df_ref["mesh_term"] = df_ref["mesh_term"].apply(lambda x: normalize_medical_text(str(x)))

            if "mesh_term" in df_current.columns:
                df_current["mesh_term"] = df_current["mesh_term"].apply(lambda x: normalize_medical_text(str(x)))
                
        drift_flags: List[bool] = []

        ## MESH CONCEPT DRIFT
        if "mesh_term" in df_ref.columns and "mesh_term" in df_current.columns:
            p_value = _detect_categorical_drift(
                df_ref["mesh_term"],
                df_current["mesh_term"],
                "mesh_term",
                p_value_threshold,
                issues,
            )
            drift_flags.append(p_value < p_value_threshold)

        ## EXPANSION DRIFT
        ref_exp = _compute_expansion_stats(df_ref)
        cur_exp = _compute_expansion_stats(df_current)

        for col in ref_exp.columns:
            if col not in cur_exp.columns:
                continue

            p_value = _detect_numeric_drift(
                ref_exp[col],
                cur_exp[col],
                col,
                p_value_threshold,
                issues,
            )
            drift_flags.append(p_value < p_value_threshold)

        ## EMBEDDING DRIFT
        ref_emb = _compute_embedding_stats(df_ref)
        cur_emb = _compute_embedding_stats(df_current)

        for col in ref_emb.columns:
            if col not in cur_emb.columns:
                continue

            p_value = _detect_numeric_drift(
                ref_emb[col],
                cur_emb[col],
                col,
                p_value_threshold,
                issues,
            )
            drift_flags.append(p_value < p_value_threshold)

        ## TEXT DRIFT
        ref_text = compute_text_stats(df_ref)
        cur_text = compute_text_stats(df_current)

        for col in ref_text.columns:
            if col not in cur_text.columns:
                continue

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

        errors = [i for i in issues if i["level"] == "error"]

        result = {
            "is_drift_ok": len(errors) == 0,
            "errors": len(errors),
            "warnings": len(issues) - len(errors),
            "drift_score": drift_score,
            "issues": issues,
        }

        ## EVIDENTLY REPORT
        try:
            report_paths = generate_evidently_report(df_ref, df_current)
            result["evidently_report"] = report_paths
        except Exception as e:
            logger.warning(f"Evidently failed: {e}")

        logger.info(f"Drift score: {drift_score}")

        if strict and drift_score < 1.0:
            raise ValidationError("Data drift detected")

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        raise DataError("Data drift pipeline failed") from exc