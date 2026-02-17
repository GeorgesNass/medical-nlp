'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Model training utilities for ICD10 prediction: Logistic Regression, Random Forest, LightGBM, FastText and BiLSTM."
'''

from __future__ import annotations

## Standard library
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional

## Third-party
import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except Exception:
    FASTTEXT_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    BILSTM_AVAILABLE = True
except Exception:
    BILSTM_AVAILABLE = False

## Internal
from src.core.errors import ModelError
from src.utils.io_utils import ensure_parent_dir
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("train", log_file="train.log")

## ============================================================
## TYPES
## ============================================================
ModelType = Literal["logreg", "random_forest", "lightgbm", "fasttext", "bilstm"]

@dataclass(frozen=True)
class TrainingConfig:
    """
        Training configuration

        Args:
            model_type: Model architecture type
            random_state: Random seed
            n_jobs: Parallel jobs
            epochs: Epochs for FastText and BiLSTM
            lr: Learning rate for FastText and BiLSTM
            batch_size: Batch size for BiLSTM
            word_ngrams: Word n-grams for FastText
    """

    model_type: ModelType
    random_state: int = 42
    n_jobs: int = -1
    epochs: int = 5
    lr: float = 0.001
    batch_size: int = 32
    word_ngrams: int = 2

## ============================================================
## BILSTM MODEL
## ============================================================
class BiLSTMClassifier(nn.Module):
    """
        BiLSTM classifier for sequence-like inputs

        Note:
            This implementation treats each sample as a sequence of length 1
            with feature dimension = input_dim (works for quick baseline).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(hidden_cat)

## ============================================================
## MODEL FACTORY
## ============================================================
def build_model(config: TrainingConfig) -> Any:
    """
        Build model instance according to configuration

        Args:
            config: TrainingConfig

        Returns:
            Initialized model
    """

    logger.info("Building model | type=%s", config.model_type)

    if config.model_type == "logreg":
        return LogisticRegression(
            max_iter=1000,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
            verbose=0,
        )

    if config.model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )

    if config.model_type == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise ModelError("LightGBM not installed")

        return LGBMClassifier(
            n_estimators=500,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )

    if config.model_type == "fasttext":
        if not FASTTEXT_AVAILABLE:
            raise ModelError("fasttext not installed")

        ## FastText is trained via fasttext.train_supervised on a text file
        return "fasttext"

    if config.model_type == "bilstm":
        if not BILSTM_AVAILABLE:
            raise ModelError("PyTorch not installed")

        ## BiLSTM is built in train_model because it requires output_dim
        return "bilstm"

    raise ModelError(f"Unsupported model_type: {config.model_type}")

## ============================================================
## FASTTEXT HELPERS
## ============================================================
def _write_fasttext_train_file(
    texts: List[str],
    y: np.ndarray,
    output_path: str | Path,
) -> Path:
    """
        Write FastText supervised train file

        Args:
            texts: List of raw texts
            y: Class indices aligned with texts
            output_path: Destination path

        Returns:
            Train file path
    """

    path = ensure_parent_dir(output_path)

    with path.open("w", encoding="utf-8") as f:
        for text, label in zip(texts, y, strict=False):
            safe_text = " ".join(str(text).splitlines()).strip()
            f.write(f"__label__{int(label)} {safe_text}\n")

    return path

def _train_fasttext_supervised(
    texts: List[str],
    y: np.ndarray,
    config: TrainingConfig,
) -> Any:
    """
        Train FastText supervised classifier

        Args:
            texts: List of raw texts
            y: Class indices aligned with texts
            config: TrainingConfig

        Returns:
            Trained fasttext model
    """

    if not FASTTEXT_AVAILABLE:
        raise ModelError("fasttext not installed")

    logger.info("Training FastText | samples=%d", len(texts))

    with tempfile.TemporaryDirectory() as tmpdir:
        train_file = Path(tmpdir) / "fasttext_train.txt"

        _write_fasttext_train_file(
            texts=texts,
            y=y,
            output_path=train_file,
        )

        model = fasttext.train_supervised(
            input=str(train_file),
            epoch=int(config.epochs),
            lr=float(config.lr),
            wordNgrams=int(config.word_ngrams),
        )

    logger.info("FastText training completed")

    return model

## ============================================================
## BILSTM HELPERS
## ============================================================
def _train_bilstm_classifier(
    X: sparse.spmatrix | np.ndarray,
    y: np.ndarray,
    config: TrainingConfig,
) -> nn.Module:
    """
        Train BiLSTM classifier

        Args:
            X: Feature matrix
            y: Class indices
            config: TrainingConfig

        Returns:
            Trained torch model
    """

    if not BILSTM_AVAILABLE:
        raise ModelError("PyTorch not installed")

    logger.info("Training BiLSTM | samples=%d", X.shape[0])

    ## Convert X to dense tensor
    X_dense = X.toarray() if sparse.issparse(X) else X
    X_tensor = torch.tensor(X_dense, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    ## Shape for BiLSTM: (batch, seq_len=1, input_dim)
    dataset = TensorDataset(X_tensor.unsqueeze(1), y_tensor)
    loader = DataLoader(dataset, batch_size=int(config.batch_size), shuffle=True)

    input_dim = int(X_tensor.shape[1])
    output_dim = int(len(np.unique(y)))
    hidden_dim = 128

    model = BiLSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(int(config.epochs)):
        epoch_loss = 0.0

        for xb, yb in loader:
            optimizer.zero_grad()

            outputs = model(xb)
            loss = criterion(outputs, yb)

            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        logger.info("BiLSTM epoch=%d | loss=%.6f", epoch + 1, epoch_loss)

    logger.info("BiLSTM training completed")

    return model

## ============================================================
## TRAINING
## ============================================================
def train_model(
    X: sparse.spmatrix | np.ndarray,
    y: np.ndarray,
    config: TrainingConfig,
    texts: Optional[List[str]] = None,
) -> Any:
    """
        Train model on provided dataset

        Notes:
            - sklearn/lightgbm use X
            - fasttext uses raw texts (texts argument)
            - bilstm uses X as dense tensor

        Args:
            X: Feature matrix
            y: Target labels (encoded integers)
            config: Training configuration
            texts: Raw texts required for fasttext

        Returns:
            Trained model
    """

    logger.info("Starting training | type=%s", config.model_type)

    model_type = config.model_type

    ## --------------------------------------------------------
    ## SKLEARN / LIGHTGBM
    ## --------------------------------------------------------
    if model_type in ["logreg", "random_forest", "lightgbm"]:
        logger.info("Training sklearn/lightgbm | samples=%d", X.shape[0])

        model = build_model(config)
        model.fit(X, y)

        logger.info("Training completed")

        return model

    ## --------------------------------------------------------
    ## FASTTEXT
    ## --------------------------------------------------------
    if model_type == "fasttext":
        if texts is None:
            raise ModelError("FastText training requires raw texts (texts=...)")

        return _train_fasttext_supervised(texts=texts, y=y, config=config)

    ## --------------------------------------------------------
    ## BILSTM
    ## --------------------------------------------------------
    if model_type == "bilstm":
        return _train_bilstm_classifier(X=X, y=y, config=config)

    raise ModelError("Unsupported model_type")

## ============================================================
## PERSISTENCE
## ============================================================
def save_model(model: Any, output_path: str | Path) -> Path:
    """
        Save trained model to disk

        Notes:
            - Sklearn/LightGBM are saved via joblib
            - FastText uses model.save_model
            - BiLSTM uses torch.save (state_dict only)

        Args:
            model: Trained model instance
            output_path: Destination path

        Returns:
            Saved path
    """

    path = ensure_parent_dir(output_path)

    ## FastText persistence
    if FASTTEXT_AVAILABLE and model.__class__.__name__.lower().startswith("fasttext"):
        model.save_model(str(path))
        logger.info("FastText model saved to %s", path)
        return path

    ## BiLSTM persistence
    if BILSTM_AVAILABLE and isinstance(model, nn.Module):
        torch.save(model.state_dict(), str(path))
        logger.info("BiLSTM model saved to %s", path)
        return path

    ## Default persistence
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)

    return path

def load_model(model_path: str | Path) -> Any:
    """
        Load model from disk

        Notes:
            - FastText loads .bin via fasttext.load_model
            - BiLSTM returns a loaded state_dict (model reconstruction done elsewhere)
            - Others load via joblib

        Args:
            model_path: Path to serialized model

        Returns:
            Loaded model
    """

    path = Path(model_path).expanduser().resolve()
    logger.info("Loading model from %s", path)

    ## FastText models are typically .bin
    if path.suffix.lower() == ".bin":
        if not FASTTEXT_AVAILABLE:
            raise ModelError("fasttext not installed")
        return fasttext.load_model(str(path))

    ## Torch models are typically .pt or .pth
    if path.suffix.lower() in [".pt", ".pth"]:
        if not BILSTM_AVAILABLE:
            raise ModelError("PyTorch not installed")
        return torch.load(str(path), map_location="cpu")

    return joblib.load(path)