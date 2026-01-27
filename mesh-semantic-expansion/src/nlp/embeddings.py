'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Embedding backends (Sentence-Transformers, CamemBERT pooling, FastText) and utilities to build/query MeSH embeddings."
'''

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.core.config import get_settings
from src.utils.logging_utils import get_logger

logger = get_logger("embeddings")

EmbeddingBackend = Literal["sentence_transformers", "camembert_pooling", "fasttext"]


## ============================================================
## CONFIG (FROM .ENV)
## ============================================================

@dataclass
class EmbeddingConfig:
    """
    Embedding configuration loaded from environment variables.

    Attributes:
        backend (EmbeddingBackend): Embedding backend to use.
        st_model_name (str): Sentence-Transformers model name.
        camembert_model_name (str): Transformers model name for CamemBERT pooling.
        fasttext_model_path (Optional[Path]): Path to FastText model (.bin).
        batch_size (int): Batch size for embedding computation.
        max_length (int): Max token length for transformer models.
        device (str): Device hint ('cpu' or 'cuda'). Backend-dependent.
    """

    backend: EmbeddingBackend
    st_model_name: str
    camembert_model_name: str
    fasttext_model_path: Optional[Path]
    batch_size: int
    max_length: int
    device: str


def get_embedding_config() -> EmbeddingConfig:
    """
    Load embedding configuration from .env / environment variables.

    Expected .env keys:
        EMBEDDING_BACKEND=sentence_transformers|camembert_pooling|fasttext
        ST_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        CAMEMBERT_MODEL_NAME=camembert-base
        FASTTEXT_MODEL_PATH=/path/to/cc.fr.300.bin
        EMB_BATCH_SIZE=64
        EMB_MAX_LENGTH=256
        EMB_DEVICE=cpu|cuda

    Returns:
        EmbeddingConfig: Loaded embedding configuration.
    """

    import os

    backend = os.getenv("EMBEDDING_BACKEND", "sentence_transformers").strip()
    st_model_name = os.getenv(
        "ST_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ).strip()
    camembert_model_name = os.getenv("CAMEMBERT_MODEL_NAME", "camembert-base").strip()
    fasttext_model_path_str = os.getenv("FASTTEXT_MODEL_PATH", "").strip()
    batch_size = int(os.getenv("EMB_BATCH_SIZE", "64").strip())
    max_length = int(os.getenv("EMB_MAX_LENGTH", "256").strip())
    device = os.getenv("EMB_DEVICE", "cpu").strip()

    fasttext_model_path = Path(fasttext_model_path_str) if fasttext_model_path_str else None

    if backend not in {"sentence_transformers", "camembert_pooling", "fasttext"}:
        raise ValueError(
            f"Unsupported EMBEDDING_BACKEND='{backend}'. "
            "Allowed: sentence_transformers, camembert_pooling, fasttext"
        )

    return EmbeddingConfig(
        backend=backend,  # type: ignore
        st_model_name=st_model_name,
        camembert_model_name=camembert_model_name,
        fasttext_model_path=fasttext_model_path,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )


## ============================================================
## TEXT NORMALIZATION
## ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize input text before embedding.

    Args:
        text (str): Raw text.

    Returns:
        str: Normalized text.
    """

    return " ".join((text or "").strip().split())


## ============================================================
## BACKEND: SENTENCE-TRANSFORMERS
## ============================================================

def _embed_sentence_transformers(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> "Any":
    """
    Compute embeddings using Sentence-Transformers.

    Args:
        texts (List[str]): Input texts.
        model_name (str): Sentence-Transformers model name.
        batch_size (int): Batch size.
        device (str): Device (cpu/cuda).

    Returns:
        Any: Numpy array of shape (n, dim), dtype float32.
    """

    try:
        import numpy as np  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise ImportError(
            "Missing dependency for sentence-transformers backend. Install: sentence-transformers"
        ) from e

    model = SentenceTransformer(model_name, device=device)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(embs, dtype="float32")


## ============================================================
## BACKEND: CAMEMBERT POOLING
## ============================================================

def _mean_pooling(last_hidden_state: "Any", attention_mask: "Any") -> "Any":
    """
    Apply mean pooling over token embeddings.

    Args:
        last_hidden_state (Any): Tensor [batch, seq, hidden].
        attention_mask (Any): Tensor [batch, seq].

    Returns:
        Any: Tensor [batch, hidden] pooled.
    """

    import torch  # type: ignore

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _embed_camembert_pooling(
    texts: List[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> "Any":
    """
    Compute embeddings using CamemBERT pooling via Hugging Face Transformers.

    Notes:
        - This uses mean pooling on the last hidden state.
        - For cosine similarity, you can normalize later before FAISS.

    Args:
        texts (List[str]): Input texts.
        model_name (str): Transformers model name (e.g., camembert-base).
        batch_size (int): Batch size.
        max_length (int): Max token length.
        device (str): Device (cpu/cuda).

    Returns:
        Any: Numpy array of shape (n, dim), dtype float32.
    """

    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as e:
        raise ImportError(
            "Missing dependency for camembert_pooling backend. Install: transformers torch"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    torch_device = torch.device(device if device else "cpu")
    model.to(torch_device)

    all_vecs: List["Any"] = []

    ## Process in batches to control memory
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded = {k: v.to(torch_device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            pooled = _mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])

        all_vecs.append(pooled.cpu().numpy())

    embs = np.vstack(all_vecs).astype("float32")
    return embs


## ============================================================
## BACKEND: FASTTEXT
## ============================================================

def _load_fasttext_model(model_path: Path) -> "Any":
    """
    Load a FastText binary model.

    Notes:
        - On Windows, we recommend installing `fasttext-wheel` (prebuilt).
        - The import name is still `fasttext`.

    Args:
        model_path (Path): Path to a FastText .bin model.

    Returns:
        Any: FastText model instance.

    Raises:
        FileNotFoundError: If model file does not exist.
    """

    if not model_path.exists():
        raise FileNotFoundError(f"FastText model not found: {model_path}")

    try:
        import fasttext  # provided by `fasttext-wheel` or `fasttext`
    except Exception as e:
        raise ImportError(
            "FastText backend requires a FastText Python package. "
            "On Windows, install: pip install fasttext-wheel "
            "(import name remains `fasttext`)."
        ) from e

    return fasttext.load_model(str(model_path))

def _embed_fasttext(
    texts: List[str],
    model_path: Path,
) -> "Any":
    """
    Compute embeddings using FastText.

    Strategy:
        - Vectorize each text as the average of word vectors.
        - FastText handles OOV via subword n-grams.

    Args:
        texts (List[str]): Input texts.
        model_path (Path): Path to FastText .bin model.

    Returns:
        Any: Numpy array of shape (n, dim), dtype float32.
    """

    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise ImportError("Missing dependency: numpy is required.") from e

    model = _load_fasttext_model(model_path)
    dim = int(model.get_dimension())

    vectors = np.zeros((len(texts), dim), dtype="float32")
    for i, txt in enumerate(texts):
        words = normalize_text(txt).split()
        if not words:
            continue
        word_vecs = [model.get_word_vector(w) for w in words]
        vectors[i] = np.mean(np.vstack(word_vecs), axis=0).astype("float32")

    return vectors

## ============================================================
## PUBLIC EMBEDDING API
## ============================================================

def embed_texts(
    texts: List[str],
    backend: Optional[EmbeddingBackend] = None,
    config: Optional[EmbeddingConfig] = None,
) -> "Any":
    """
    Compute embeddings for a list of texts using the selected backend.

    Args:
        texts (List[str]): Input texts.
        backend (Optional[EmbeddingBackend]): Override backend.
        config (Optional[EmbeddingConfig]): Optional preloaded config.

    Returns:
        Any: Numpy array (n, dim), dtype float32.
    """

    cfg = config if config else get_embedding_config()
    selected = backend if backend else cfg.backend

    ## Normalize inputs
    cleaned = [normalize_text(t) for t in texts]

    logger.info(f"Embedding backend: {selected}. Texts: {len(cleaned)}")

    if selected == "sentence_transformers":
        return _embed_sentence_transformers(
            texts=cleaned,
            model_name=cfg.st_model_name,
            batch_size=cfg.batch_size,
            device=cfg.device,
        )

    if selected == "camembert_pooling":
        return _embed_camembert_pooling(
            texts=cleaned,
            model_name=cfg.camembert_model_name,
            batch_size=cfg.batch_size,
            max_length=cfg.max_length,
            device=cfg.device,
        )

    if selected == "fasttext":
        if cfg.fasttext_model_path is None:
            raise ValueError("FASTTEXT_MODEL_PATH must be set for fasttext backend.")
        return _embed_fasttext(
            texts=cleaned,
            model_path=cfg.fasttext_model_path,
        )

    raise ValueError(f"Unsupported backend: {selected}")


def embed_query_text(
    text: str,
    backend: Optional[EmbeddingBackend] = None,
    config: Optional[EmbeddingConfig] = None,
) -> "Any":
    """
    Convenience method to embed a single query text.

    Args:
        text (str): Query text.
        backend (Optional[EmbeddingBackend]): Override backend.
        config (Optional[EmbeddingConfig]): Optional preloaded config.

    Returns:
        Any: Numpy array shape (dim,), float32.
    """

    vectors = embed_texts([text], backend=backend, config=config)
    return vectors[0]


## ============================================================
## BUILD MESH EMBEDDINGS (mesh_parsed.jsonl -> parquet)
## ============================================================

def _load_mesh_texts_for_embedding(mesh_jsonl_path: Path) -> Tuple[List[str], List[str]]:
    """
    Load MeSH UIs and embedding texts from a JSONL file.

    Strategy:
        - One embedding per UI
        - Text = preferred term + synonyms (joined)

    Args:
        mesh_jsonl_path (Path): Path to mesh_parsed.jsonl.

    Returns:
        Tuple[List[str], List[str]]: (ui_list, text_list)
    """

    ui_list: List[str] = []
    text_list: List[str] = []

    with open(mesh_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ui = (row.get("ui") or "").strip()
            if not ui:
                continue

            preferred = row.get("preferred_terms", []) or []
            synonyms = row.get("synonyms", []) or []

            ## Build a single embedding text per UI
            parts = []
            if preferred:
                parts.append(preferred[0])
            if synonyms:
                ## Limit synonyms to keep text size stable
                parts.extend(synonyms[:15])

            text = " | ".join([normalize_text(p) for p in parts if normalize_text(p)])
            if not text:
                continue

            ui_list.append(ui)
            text_list.append(text)

    return ui_list, text_list


def save_embeddings_parquet(
    ui_list: List[str],
    vectors: "Any",
    output_path: Path,
) -> Path:
    """
    Save embeddings to parquet as emb_0..emb_n columns.

    Args:
        ui_list (List[str]): List of MeSH UI identifiers.
        vectors (Any): Numpy array (n, dim).
        output_path (Path): Parquet output path.

    Returns:
        Path: Output parquet path.
    """

    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as e:
        raise ImportError("Missing dependency: numpy/pandas required to save parquet.") from e

    if len(ui_list) != int(vectors.shape[0]):
        raise ValueError("ui_list length must match number of vectors.")

    vectors = np.asarray(vectors, dtype="float32")
    dim = int(vectors.shape[1])

    data: Dict[str, Any] = {"ui": ui_list}
    for i in range(dim):
        data[f"emb_{i}"] = vectors[:, i]

    df = pd.DataFrame(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.info(f"Embeddings parquet saved: {output_path} (rows={len(df)}, dim={dim})")
    return output_path


def build_mesh_embeddings(
    mesh_jsonl_path: Optional[Path] = None,
    output_parquet_path: Optional[Path] = None,
    backend: Optional[EmbeddingBackend] = None,
    overwrite: bool = False,
) -> Path:
    """
    Build MeSH embeddings parquet from mesh_parsed.jsonl.

    Args:
        mesh_jsonl_path (Optional[Path]): Input MeSH JSONL path.
        output_parquet_path (Optional[Path]): Output parquet path.
        backend (Optional[EmbeddingBackend]): Optional backend override.
        overwrite (bool): If True, overwrite existing parquet.

    Returns:
        Path: Path to generated parquet file.
    """

    settings = get_settings()
    cfg = get_embedding_config()

    src_path = mesh_jsonl_path if mesh_jsonl_path else settings.mesh_parsed_file
    out_path = output_parquet_path if output_parquet_path else settings.mesh_embeddings_file
    selected = backend if backend else cfg.backend

    if not src_path.exists():
        raise FileNotFoundError(f"MeSH parsed JSONL not found: {src_path}")

    if out_path.exists() and not overwrite:
        logger.info(f"Embeddings parquet already exists: {out_path}")
        return out_path

    if out_path.exists() and overwrite:
        out_path.unlink()

    logger.info(f"Building MeSH embeddings: backend={selected}")
    ui_list, texts = _load_mesh_texts_for_embedding(src_path)

    logger.info(f"Loaded MeSH records for embedding: {len(ui_list)}")
    vectors = embed_texts(texts, backend=selected, config=cfg)

    return save_embeddings_parquet(ui_list=ui_list, vectors=vectors, output_path=out_path)
