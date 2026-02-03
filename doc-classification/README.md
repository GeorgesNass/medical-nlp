# 🏥 Medical Document Classification & Similarity-Based Labeling

## 1. Project Overview

This project implements a **medical document classification pipeline** based on **semantic similarity** and **segment-level analysis**.

The goal is to automatically assign **multiple binary labels (multi-label classification)** to medical documents such as:

- Hospital discharge summaries (CRH)
- Operative reports (CRO)
- Anesthesia reports (CRA)
- Prescriptions
- Laboratory results
- Administrative admission forms

A single document may belong to **multiple categories simultaneously**, which motivates a **similarity-based approach** rather than a single end-to-end classifier.

---

## 2. Problem Statement

Medical documents often:

- Combine several document types in a single file
- Share overlapping vocabulary and structure
- Contain heterogeneous clinical and administrative sections

This project addresses these constraints by:

- Working at **segment level**
- Reusing **labeled documents as semantic anchors**
- Applying **binary decisions per label** with explainable evidence

---

## 3. Classification Strategy

### Multi-label, Binary Decisions

Each label is treated independently as a binary decision:

- Is there evidence of a hospital discharge summary (CRH)?
- Is there operative or anesthesia content (CRO / CRA)?
- Are prescriptions, lab results, or admission forms present?

This allows overlapping labels, fine-grained threshold tuning, and explainability.

### Similarity-Based Labeling

The pipeline:

1. Segments labeled documents into overlapping text blocks
2. Encodes segments into dense embeddings
3. Builds a similarity index
4. Segments and encodes unlabeled documents
5. Retrieves nearest labeled segments
6. Aggregates similarity scores per label
7. Applies per-label thresholds

---

## 4. Pipeline Architecture

```text
Document (.txt)
   ↓
Normalization
   ↓
Segmentation (sliding window)
   ↓
Embeddings (CPU / GPU auto)
   ↓
Similarity Index
   ↓
Label Aggregation
   ↓
Multi-label Predictions
   ↓
CSV / Reports
```

---

## 5. Exploratory Data Analysis (EDA)

An EDA module is included to analyze labeled and unlabeled corpora:

- Number of documents
- Average document length
- Number of segments
- Label distribution
- Multi-label frequency
- Weak keyword diagnostics

Outputs are exported as JSON reports in:

```text
artifacts/reports/
```

---

## 6. Project Structure

```text
doc-classification/
├── main.py                     # CLI entry point (full pipeline, EDA, index, predict)
├── pipeline.py                 # High-level orchestration logic
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── .env                        # Environment configuration
├── menu_pipeline.sh            # Interactive CLI menu
├── docker/
│   ├── Dockerfile              # Docker image definition
│   └── docker-compose.yml      # Docker Compose configuration
├── data/
│   ├── labeled/                # Labeled medical documents (.txt)
│   ├── unlabeled/              # Unlabeled medical documents (.txt)
│   └── processed/              # Preprocessed / intermediate files
├── artifacts/
│   ├── indexes/                # Similarity indexes
│   ├── models/                 # Optional trained models
│   ├── reports/                # EDA and diagnostics outputs
│   └── exports/                # CSV prediction outputs
├── tests/
│   └── test_unit.py            # Unit tests
└── src/
    ├── core/
    │   ├── config.py            # Global configuration and environment loading
    │   ├── errors.py            # Centralized custom exceptions
    │   └── eda.py               # Exploratory Data Analysis logic
    ├── domain/
    │   └── schema.py            # Core domain dataclasses (Document, Segment, Prediction)
    ├── nlp/
    │   ├── segmenter.py         # Text segmentation (sliding windows)
    │   ├── embeddings.py        # Embedding backend (CPU/GPU)
    │   └── similarity_index.py  # Vector similarity search
    ├── labeling/
    │   ├── label_definitions.py # Label configuration and thresholds
    │   ├── similarity_labeler.py# Similarity-based classifier
    │   └── hybrid_labeler.py    # Extension point for hybrid strategies
    └── utils/
        ├── io_utils.py          # Text loading and normalization
        ├── data_utils.py        # CSV/JSON export and helpers
        └── logging_utils.py     # Centralized logging
```

---

## 7. Prerequisites

### General

- Python **3.10+**
- Docker and Docker Compose
- Optional GPU with CUDA support

### Windows & WSL2 Prerequisites

```bash
# PowerShell
wsl --status
wsl --install
wsl --list --online
wsl --install -d Ubuntu
wsl -d Ubuntu

docker --version
docker compose version
```

### Ubuntu

```bash
sudo apt update
sudo apt install -y git
git --version
```

### Python

```bash
python3 --version
sudo apt install -y python3-pip python3-venv
```

---

## 8. Setup

### Manual installation

```bash
python -m venv .dc_env
source .dc_env/bin/activate ## .dc_env\Scripts\activate.bat for windows
pip install --upgrade pip
pip install -r requirements.txt
```

### Docker Usage

Build and start the pipeline:

```bash
docker compose build
docker compose up
```

---

## 9. CLI Usage

```bash
# Run EDA only
python main.py eda

# Build similarity index
python main.py index

# Predict labels for unlabeled documents
python main.py predict

# Run full pipeline
python main.py full
```

---

## 10. Tests

```bash
pytest
```

---

## ✅ Full System Verification (End-to-End)

Run the following commands in order:

```bash
# Check labeled and unlabeled data
ls data/labeled
ls data/unlabeled

# Inspect a sample document
head -n 40 data/labeled/sample.txt

# Check embedding backend
python -c "from src.nlp.embeddings import EmbeddingBackend; b=EmbeddingBackend(); print(b.use_gpu)"

# Test embeddings
python -c "from src.nlp.embeddings import EmbeddingBackend; b=EmbeddingBackend(); print(b.encode(['test medical text']).shape)"

# Build index
python main.py index

# Run full pipeline
python main.py full

# Inspect outputs
ls artifacts/exports
head -n 5 artifacts/exports/predictions.csv

# Run tests
pytest -q
```

---

## Author

**Georges Nassopoulos**  
Email: georges.nassopoulos@gmail.com

**Status:** Research / Professional NLP project

---

### Next possible versions

If you want next:

- ultra-compact version
- scientific paper / methodology report
- jury / defense version

Tell me which one you want.
