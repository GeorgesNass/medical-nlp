# 📄 Medical Document Classification & Similarity-Based Labeling

The project implements a **medical document classification pipeline based on semantic similarity and segment-level analysis**, enabling **multi-label classification of heterogeneous clinical documents**.

---

## 🎯 Project Overview

Main capabilities:

* Multi-label classification for medical documents
* Segment-level semantic similarity analysis
* Similarity-based labeling using reference documents
* Explainable label decisions based on nearest segments
* CLI-driven pipeline execution
* Export of predictions and diagnostics reports

The system assigns **multiple document-type labels simultaneously** to clinical documents by comparing their segments to **semantically similar labeled segments**.

---

## ⚙️ Tech Stack

Core technologies used in the project:

* Python
* Sentence-transformer embeddings
* Vector similarity search
* Docker & Docker Compose
* Sliding-window text segmentation
* CPU / GPU embedding inference

---

## 📂 Project Structure

```text
doc-classification/
├── main.py                           ## CLI entry point (full pipeline, EDA, index, predict)
├── pipeline.py                       ## High-level orchestration logic
├── README.md                         ## Project documentation
├── requirements.txt                  ## Python dependencies
├── .env                              ## Environment configuration
├── menu_pipeline.sh                  ## Interactive CLI menu
│
├── docker/
│   ├── Dockerfile                    ## Docker image definition
│   └── docker-compose.yml            ## Docker Compose configuration
│
├── data/
│   ├── labeled/                      ## Labeled medical documents (.txt)
│   ├── unlabeled/                    ## Unlabeled medical documents (.txt)
│   └── processed/                    ## Preprocessed / intermediate files
│
├── artifacts/
│   ├── indexes/                      ## Similarity indexes
│   ├── models/                       ## Optional trained models
│   ├── reports/                      ## EDA and diagnostics outputs
│   └── exports/                      ## CSV prediction outputs
│
├── tests/
│   └── test_unit.py                  ## Unit tests
│
└── src/
    ├── core/
    │   ├── config.py                 ## Global configuration and environment loading
    │   ├── errors.py                 ## Centralized custom exceptions
    │   └── eda.py                    ## Exploratory Data Analysis logic
	│
    ├── domain/
    │   └── schema.py                 ## Core domain dataclasses (Document, Segment, Prediction)
	│	
    ├── nlp/
    │   ├── segmenter.py              ## Text segmentation (sliding windows)
    │   ├── embeddings.py             ## Embedding backend (CPU/GPU)
    │   └── similarity_index.py       ## Vector similarity search
	│	
    ├── labeling/
    │   ├── label_definitions.py      ## Label configuration and thresholds
    │   ├── similarity_labeler.py     ## Similarity-based classifier
    │   └── hybrid_labeler.py         ## Extension point for hybrid strategies
	│
    └── utils/
        ├── io_utils.py               ## Text loading and normalization
        ├── data_utils.py             ## CSV/JSON export and helpers
        └── logging_utils.py          ## Centralized logging
```

---

## ❓ Problem Statement

Medical documents frequently present challenges such as:

* multiple document types within a single file
* overlapping clinical and administrative sections
* heterogeneous formatting
* shared vocabulary across document categories

This project addresses these constraints by:

* performing **segment-level analysis**
* reusing **labeled documents as semantic anchors**
* applying **independent binary decisions per label**
* providing **explainable similarity evidence**

---

## 🧠 Approach / Methodology / Strategy

A document may belong to multiple categories.

### Multi-label Binary Decisions

Each label is evaluated independently as a binary decision:

* discharge summary (CRH)
* operative report (CRO)
* anesthesia report (CRA)
* prescription
* laboratory results
* admission forms

- Is there evidence of a hospital discharge summary (CRH)?
- Is there operative or anesthesia content (CRO / CRA)?
- Are prescriptions, lab results, or admission forms present?

This allows:

* overlapping labels
* independent threshold tuning
* interpretable classification logic

---

### Similarity-Based Labeling

The pipeline performs the following steps:

1. Segment labeled documents into overlapping windows
2. Encode segments into dense embeddings
3. Build a similarity index
4. Segment unlabeled documents
5. Encode segments
6. Retrieve nearest labeled segments
7. Aggregate similarity scores per label
8. Apply label-specific thresholds

---

## 🏗 Pipeline Architecture

```text
Document (.txt)
        ↓
Text Normalization
        ↓
Segmentation (sliding window)
        ↓
Embeddings (CPU/GPU)
        ↓
Similarity Index
        ↓
Label Aggregation
        ↓
Multi-label Predictions
        ↓
CSV Export + Reports
```

---

## 📊 Exploratory Data Analysis

The project includes an EDA module for corpus diagnostics:

* document statistics
* segment statistics
* label distribution
* multi-label frequency
* keyword diagnostics

Outputs are stored in:

```
artifacts/reports/
```

---


## 🔧 Setup & Installation

In this section we explain the minimum OS verification, python usage and docker setup.

### 1. Requirements

- Python **3.10+**
- Docker and Docker Compose
- Optional GPU with CUDA support

---

### 2. OS prerequisites

Verify that required packages are installed.

#### Windows / WSL2 (recommended)

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

#### Ubuntu

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential curl git
python --version
```

---

### 3. Python environment

```bash
python -m venv .dc_env
source .dc_env/bin/activate		     ## for windows .dc_env\Scripts\activate.bat
pip install --upgrade pip            ## for windows : .dc_env\Scripts\python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Docker setup

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
```

---


## ▶️ Usage & End-to-End Testing

```bash
## Check labeled and unlabeled data
ls data/labeled
ls data/unlabeled

## Inspect a sample document
head -n 40 data/labeled/sample.txt

## Check embedding backend
python -c "from src.nlp.embeddings import EmbeddingBackend; b=EmbeddingBackend(); print(b.use_gpu)"

## Test embeddings
python -c "from src.nlp.embeddings import EmbeddingBackend; b=EmbeddingBackend(); print("Embedding shape:", b.encode(['test medical text']).shape)"

## Run EDA only
python main.py eda

## Build index
python main.py index

## Predict labels for unlabeled documents
python main.py predict

## Run full pipeline
python main.py full

## Inspect outputs
ls artifacts/exports
head -n 5 artifacts/exports/predictions.csv

## Run tests
pytest -q
```

---

## 📛 Common Errors & Troubleshooting

| Error                     | Cause                     | Solution                             |
| ------------------------- | ------------------------- | ------------------------------------ |
| Embedding backend failure | Missing embedding model   | Install required embedding libraries |
| Index creation failure    | Missing labeled data      | Verify `data/labeled/` contents      |
| Prediction failure        | Missing similarity index  | Run `python main.py index` first     |
| Docker container failure  | Misconfigured environment | Rebuild containers                   |

---

## 👤 Author

**Georges Nassopoulos**
[georges.nassopoulos@gmail.com](mailto:georges.nassopoulos@gmail.com)

**Status:** Medical NLP / Document Classification Project
