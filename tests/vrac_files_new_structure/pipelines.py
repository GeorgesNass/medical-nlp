'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main pipeline orchestration: RSS parsing, CSV building, vectorization, training, evaluation and export."
'''

from __future__ import annotations

## Standard library
from pathlib import Path
from typing import List

## Third-party
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

## Internal
from src.utils.logging_utils import get_logger
from src.utils.io_utils import write_parquet
from src.core.errors import (
    DataError,
    ModelError,
    log_and_raise_data_error,
    log_and_raise_missing_folder,
)
from src.icd10.parse_rss import parse_rss_folder
from src.icd10.build_clinical_csv import (
    build_clinical_records_csv,
    build_icd10_consolidated_csv,
)
from src.nlp.preprocess import preprocess_text
from src.nlp.vectorizers import (
    build_tfidf_vectorizer,
    fit_transform_tfidf,
)
from src.model.train import (
    TrainingConfig,
    train_model,
    save_model,
)
from src.model.evaluate import (
    compute_basic_metrics,
    export_metrics,
)

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("pipelines", log_file="pipelines.log")

## ============================================================
## RSS PIPELINE
## ============================================================
def run_rss_parsing(
    rss_folder: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """
        Parse RSS folder and export consolidated CSV

        Args:
            rss_folder: Folder containing .rss files
            output_csv: Output CSV path

        Returns:
            Consolidated RSS DataFrame
    """

    logger.info("Running RSS parsing pipeline")

    rss_folder = Path(rss_folder)

    if not rss_folder.exists():
        from src.core.errors import log_and_raise_missing_folder
        log_and_raise_missing_folder(
            rss_folder,
            reason="RSS folder does not exist. Expected: data/raw/icd10/",
        )

    rss_files = sorted(rss_folder.glob("*.rss"))
    if len(rss_files) == 0:
        from src.core.errors import log_and_raise_data_error
        log_and_raise_data_error(
            reason=(
                f"No .rss files found in: {rss_folder} | "
                "Add RSS files to data/raw/icd10/ then rerun: python main.py --parse-rss"
            )
        )

    rss_df = parse_rss_folder(rss_folder)

    required_cols = {"admission_id", "primary_diagnosis_code"}
    missing = sorted(list(required_cols - set(rss_df.columns)))

    if rss_df.empty or missing:
        from src.core.errors import log_and_raise_data_error
        log_and_raise_data_error(
            reason=(
                f"RSS parsing produced no usable records | "
                f"rows={len(rss_df)} missing_cols={missing} | "
                f"folder={rss_folder} | "
                "Verify RSS input format and parser rules."
            )
        )

    build_icd10_consolidated_csv(
        rss_df=rss_df,
        output_path=output_csv,
    )

    return rss_df

def run_clinical_csv_build(
    clinical_records_dir: str | Path,
    rss_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
        Build per-admission CSV files

        Args:
            clinical_records_dir: Folder with admission subfolders
            rss_df: Parsed RSS DataFrame
            output_dir: Output directory
    """

    logger.info("Building clinical record CSV files")

    build_clinical_records_csv(
        clinical_records_dir=clinical_records_dir,
        rss_df=rss_df,
        output_dir=output_dir,
    )

## ============================================================
## DATASET PREPARATION
## ============================================================
def build_training_dataset(
    clinical_csv_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """
        Build feature matrix and encoded labels

        Args:
            clinical_csv_dir: Folder with per-admission CSV files

        Returns:
            Tuple (X, y_encoded, class_names)
    """

    logger.info("Building training dataset")

    ## --------------------------------------------------------
    ## Validate input directory + list CSV files
    ## --------------------------------------------------------
    csv_dir = Path(clinical_csv_dir)

    if not csv_dir.exists() or not csv_dir.is_dir():
        log_and_raise_missing_folder(
            csv_dir,
            reason="Run: python main.py --build-clinical-csv (this folder is required for training).",
        )

    csv_files = sorted(csv_dir.glob("*.csv"))

    if not csv_files:
        log_and_raise_data_error(
            reason=(
                f"No per-admission CSV files found in: {csv_dir} | "
                "Run: python main.py --build-clinical-csv and verify outputs."
            )
        )

    ## --------------------------------------------------------
    ## Load + validate samples
    ## --------------------------------------------------------
    texts: List[str] = []
    labels: List[str] = []

    skipped_missing_cols = 0
    skipped_empty_text = 0
    skipped_read_errors = 0

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            skipped_read_errors += 1
            logger.warning("Skipping unreadable CSV: %s | %s", file_path, str(exc))
            continue

        ## Required columns
        if "primary_diagnosis_code" not in df.columns or "text_content" not in df.columns:
            skipped_missing_cols += 1
            continue

        ## Use first diagnosis as label
        label = df["primary_diagnosis_code"].iloc[0]

        if pd.isna(label) or str(label).strip() == "":
            skipped_missing_cols += 1
            continue

        ## Concatenate all text_content rows
        full_text = " ".join(df["text_content"].astype(str).tolist())

        ## Preprocess text minimally
        full_text = preprocess_text(full_text)

        ## Drop empty samples after preprocessing
        if not full_text or not full_text.strip():
            skipped_empty_text += 1
            continue

        texts.append(full_text)
        labels.append(str(label).strip())

    ## --------------------------------------------------------
    ## Final validation before vectorization
    ## --------------------------------------------------------
    if not texts or not labels:
        log_and_raise_data_error(
            reason=(
                f"No valid training samples found in: {csv_dir} | "
                f"csv_files={len(csv_files)} | skipped_missing_cols={skipped_missing_cols} | "
                f"skipped_empty_text={skipped_empty_text} | skipped_read_errors={skipped_read_errors} | "
                "Check that CSVs contain 'primary_diagnosis_code' and 'text_content' with non-empty content."
            )
        )

    ## Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    ## Vectorize texts
    vectorizer = build_tfidf_vectorizer()

    try:
        X = fit_transform_tfidf(vectorizer, texts)
    except ValueError as exc:
        ## Common sklearn case: empty vocabulary
        log_and_raise_data_error(
            reason=(
                f"TF-IDF vectorization failed: {str(exc)} | "
                "Most common cause: all documents become empty after preprocessing (or contain only stopwords). "
                "Inspect a few generated CSVs and ensure 'text_content' has meaningful content."
            )
        )

    logger.info(
        "Dataset ready | samples=%d | skipped_missing_cols=%d | skipped_empty_text=%d | skipped_read_errors=%d",
        X.shape[0],
        skipped_missing_cols,
        skipped_empty_text,
        skipped_read_errors,
    )

    return X, y_encoded, list(label_encoder.classes_)

## ============================================================
## TRAINING PIPELINE
## ============================================================
def run_training_pipeline(
    clinical_csv_dir: str | Path,
    model_output_path: str | Path,
    metrics_output_path: str | Path,
) -> None:
    """
        Full training + evaluation pipeline

        Args:
            clinical_csv_dir: Folder with per-admission CSV files
            model_output_path: Where to save trained model
            metrics_output_path: Where to save metrics JSON
    """

    logger.info("Starting training pipeline")

    ## Build dataset (raises DataError with user-friendly messages)
    X, y, class_names = build_training_dataset(clinical_csv_dir)

    ## Train model
    try:
        config = TrainingConfig(model_type="logreg")
        model = train_model(X, y, config)
    except DataError:
        raise
    except Exception as exc:
        raise ModelError(f"Model training failed: {str(exc)}") from exc

    ## Save model
    save_model(model, model_output_path)

    ## Evaluate on training set (baseline)
    y_pred = model.predict(X)

    metrics = compute_basic_metrics(y, y_pred)

    export_metrics(metrics, metrics_output_path)

    logger.info("Training pipeline completed | classes=%d", len(class_names))

## ============================================================
## DATA EXPORT
## ============================================================
def export_dataset_parquet(
    clinical_csv_dir: str | Path,
    output_parquet: str | Path,
) -> None:
    """
        Export full training dataset as parquet

        Args:
            clinical_csv_dir: Folder with CSV files
            output_parquet: Destination parquet file
    """

    logger.info("Exporting dataset to parquet")

    X, y, class_names = build_training_dataset(clinical_csv_dir)

    df = pd.DataFrame(X.toarray())
    df["label"] = y

    write_parquet(df, output_parquet)

    logger.info("Dataset exported to %s | classes=%d", output_parquet, len(class_names))