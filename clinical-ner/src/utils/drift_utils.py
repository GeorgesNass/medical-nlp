'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Drift utilities for clinical-ner: statistical tests, NER stats, text features and Evidently reporting."
'''

from __future__ import annotations

from typing import Dict, Tuple, Any

import json
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import ks_2samp, chi2_contingency
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
    
from src.core.config import get_config
from src.nlp.normalization import normalize_clinical_text
from src.utils.logging_utils import get_logger

try:
    from src.core.errors import ValidationError, DataError
except Exception:
    ValidationError = ValueError
    DataError = RuntimeError

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("drift_utils")

def compute_ks_test(ref: pd.Series, cur: pd.Series) -> Tuple[float, float]:
    """
        Compute Kolmogorov-Smirnov test

        Args:
            ref: Reference series
            cur: Current series

        Returns:
            statistic, p_value
    """

    ## REMOVE NaN
    ref_clean = ref.dropna()
    cur_clean = cur.dropna()

    ## HANDLE EMPTY
    if ref_clean.empty or cur_clean.empty:
        return 0.0, 1.0

    ## COMPUTE TEST
    stat, p_value = ks_2samp(ref_clean, cur_clean)

    return float(stat), float(p_value)

def compute_chi2_test(ref: pd.Series, cur: pd.Series) -> Tuple[float, float]:
    """
        Compute Chi-square test

        Args:
            ref: Reference series
            cur: Current series

        Returns:
            statistic, p_value
    """

    ## VALUE COUNTS
    ref_counts = ref.value_counts()
    cur_counts = cur.value_counts()

    ## ALIGN INDEX
    all_index = ref_counts.index.union(cur_counts.index)
    ref_aligned = ref_counts.reindex(all_index, fill_value=0)
    cur_aligned = cur_counts.reindex(all_index, fill_value=0)

    ## BUILD TABLE
    table = np.array([ref_aligned.values, cur_aligned.values])

    ## HANDLE EMPTY
    if table.sum() == 0:
        return 0.0, 1.0

    ## COMPUTE TEST
    stat, p_value, _, _ = chi2_contingency(table)

    return float(stat), float(p_value)

def compute_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
        Compute text-based features

        Args:
            df: Input dataset

        Returns:
            DataFrame with text stats
    """

    data: Dict[str, pd.Series] = {}

    if "text" in df.columns:
        config = get_config()

        text_series = df["text"].fillna("").astype(str)

        if config.feature_engineering.enabled:
            text_series = text_series.apply(normalize_clinical_text)

        ## LENGTH
        data["text_length"] = text_series.str.len()

        ## WORD COUNT
        data["text_word_count"] = text_series.str.split().apply(len)

        if config.feature_engineering.enabled:
            data["avg_token_length"] = text_series.apply(
                lambda t: sum(len(w) for w in t.split()) / max(1, len(t.split()))
            )
            
    return pd.DataFrame(data)

def compute_ner_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
        Compute NER-based features

        Args:
            df: Input dataset

        Returns:
            DataFrame with NER stats
    """

    data: Dict[str, pd.Series] = {}

    if "entities" in df.columns:
        entities = df["entities"]

        ## ENTITY COUNT
        data["entity_count"] = entities.apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        config = get_config()

        if config.feature_engineering.enabled and "text" in df.columns:
            text_len = df["text"].fillna("").astype(str).apply(len)

            data["entity_density"] = entities.apply(
                lambda x: len(x) if isinstance(x, list) else 0
            ) / text_len.replace(0, 1)
            
    return pd.DataFrame(data)

def generate_evidently_report(
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
    output_dir: str = "reports",
) -> Dict[str, str]:
    """
        Generate Evidently data drift report

        High-level workflow:
            1) Initialize report
            2) Run drift comparison
            3) Save HTML

        Args:
            df_ref: Reference dataset
            df_cur: Current dataset
            output_dir: Output directory

        Returns:
            Dict with report path
    """

    ## BUILD REPORT
    report = Report(metrics=[DataDriftPreset()])

    ## RUN
    report.run(
        reference_data=df_ref,
        current_data=df_cur,
    )

    ## SAVE
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    html_path = path / "evidently_report.html"
    report.save_html(str(html_path))

    return {
        "evidently_html": str(html_path),
    }

def generate_drift_report(
    metrics: Dict[str, Any],
    output_dir: str = "reports",
) -> Dict[str, str]:
    """
        Generate custom drift report

        Args:
            metrics: Drift metrics
            output_dir: Output directory

        Returns:
            Paths dict
    """

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    ## JSON
    json_path = path / "drift_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    ## HTML
    html_path = path / "drift_report.html"
    html_content = "<html><body><h1>Clinical NER Drift Report</h1><pre>"
    html_content += json.dumps(metrics, indent=2)
    html_content += "</pre></body></html>"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return {
        "report_json": str(json_path),
        "report_html": str(html_path),
    # }