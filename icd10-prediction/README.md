# 🏥 ICD10 Prediction from Clinical Records

The pipeline transforms heterogeneous hospital data into **structured datasets and predictive models for ICD10 diagnosis classification**.

---

## 🎯 Project Overview

Main capabilities:

* Parse hospital **RSS administrative files**
* Process **clinical text documents per admission**
* Build structured **ML-ready datasets**
* Train **ICD10 classification models**
* Evaluate prediction quality with medical classification metrics
* Serve predictions through a **FastAPI API**

The system converts raw clinical and administrative hospital data into **automated ICD10 diagnosis predictions**.

---

## ⚙️ Tech Stack

Core technologies used in the project:

* Python
* FastAPI
* Docker & Docker Compose
* Scikit-learn
* LightGBM
* FastText
* PyTorch (BiLSTM)
* Pandas / NumPy
* TF-IDF and embeddings
* Clinical NLP preprocessing

---

## 📂 Project Structure

```
icd10_prediction/
├── main.py                           ## FastAPI entry point (minimal API: config, logging, routes, healthcheck)
├── menu_pipeline.sh                  ## Interactive CLI menu (parse RSS, build CSV, train, eval, predict, export, run API)
├── requirements.txt
├── README.md
├── .env                              ## Environment configuration (paths, GPU, thresholds, etc.)
│
├── docker/                           ## Container definition & service orchestration (API, volumes)
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── logs/                             ## Centralized application logs (auto-created via logging_utils)
│
├── data/
│   ├── raw/
│   │   ├── clinical_records/         ## One folder per admission_id (hospital stay, match RSS)
│   │   └── icd10/                    ## Raw .rss files (structured medical coding information)
│   │
│   ├── interim/
│   │   ├── clinical_records_csv/     ## One CSV per admission_id (RSS fields, document types, file name, text content)
│   │   ├── icd10_csv/                ## Single consolidated CSV parsed from all .rss files (ordered by year)
│   │   ├── datasets/                 ## ML-ready datasets (train/val/test in parquet format)
│   │   └── embeddings/               ## Optional cached embeddings (if transformer models used)
│   │
│   └── processed/
│       ├── features/                 ## Final vectorized features (TF-IDF, FastText, embeddings)
│       ├── labels/                   ## Encoded diagnosis labels (primary_diagnosis_code)
│         └── error_analysis/         ## False positives/negatives and misclassification dumps
│
├── artifacts/
│   ├── models/                       ## Trained models (LR, RF, LightGBM, FastText, BiLSTM, etc.)
│   ├── metadata/                     ## Label encoders, vectorizers, config snapshots, mappings
│   ├── predictions/                  ## Raw prediction outputs (jsonl/parquet)
│   ├── exports/
│   │   ├── review.csv                ## Human validation file (top-k ICD10 codes + confidence)
│   │   ├── validated.md              ## Manual validation notes and adjustments
│   │   └── eda/                      ## EDA plots and dataset diagnostics
│   │
│   └── reports/                      ## Evaluation metrics, evaluation report, most frequent ICD10 confusions
│       ├── metrics.json
│       ├── metrics.md
│       └── confusion_top_codes.csv
│
├── tests/                            ## Unit tests (RSS parsing, hashtag extraction, metrics, taxonomy, end-to-end smoke test)
│   └── test_unit.py
│
└── src/
    ├── pipelines.py                  ## End-to-end orchestration logic (parse → merge → train → eval → export)
    ├── __init__.py
    │
    ├── utils/
    │   ├── __init__.py
    │   ├── logging_utils.py          ## Centralized logging (no print statements)
    │   └──io_utils.py                ## Safe CSV / JSONL / Parquet read-write helpers
    │
    ├── core/
    │   ├── __init__.py
    │   ├── service.py                ## FastAPI routes (/predict, /topk, /batch, /health, /models)
    │   ├── schema.py                 ## Pydantic request/response models
    │   ├── config.py                 ## Environment configuration + path resolution + run_id
    │   ├── eda.py                    ## Exploratory Data Analysis logic
    │   └── errors.py                 ## Centralized custom exceptions
    │
    ├── nlp/
    │   ├── __init__.py
    │   ├── preprocess.py             ## Text normalization and minimal cleaning (no content loss)
    │   ├── vectorizers.py            ## TF-IDF / hashing-based vectorization
    │   ├── embeddings.py             ## Sentence-transformers / clinical embedding models (optional)
    │   └── postprocess.py            ## Thresholding, top-k selection, calibration logic
    │
    ├── model/
    │   ├── __init__.py
    │   ├── train.py                  ## Model training (LogReg, RF, LightGBM, FastText, BiLSTM)
    │   ├── evaluate.py               ## Evaluation metrics (micro/macro F1, Precision@k, Recall@k)
    │   ├── predict.py                ## Inference wrapper (single & batch)
    │   ├── calibrate.py              ## Optional probability calibration
    │   └── explain.py                ## Feature importance / attention visualization
    │
    └── icd10/
        ├── __init__.py
        ├── build_clinical_csv.py     ## Structured RSS info with text content records per admission_id
        ├── parse_rss.py              ## Clean fixed-width RSS parser → structured records
        ├── index_icd10.py            ## Optional SQLite / FTS index for ICD10 code lookup
        └── taxonomy.py               ## ICD10 hierarchy utilities (parent/child relations)
```

---

## ❓ Problem Statement

Hospital medical data is distributed across multiple heterogeneous sources:

* RSS administrative export files
* Clinical text documents per hospital admission
* Multiple document types per admission

Key challenges include:

* heterogeneous data formats (`.rss`, `.txt`)
* multiple documents per admission
* sensitive clinical content
* strong class imbalance across ICD10 codes

This project addresses these challenges through:

* structured RSS parsing
* per-admission clinical dataset construction
* text vectorization and embeddings
* supervised classification models
* reproducible ML pipeline orchestration

---

## 🧠 Approach / Methodology / Strategy

The platform predicts **primary ICD10 diagnosis codes** using clinical text and structured hospital metadata.

Core principles:

* **multi-source data consolidation**
* **clinical text preprocessing**
* **vectorized feature extraction**
* **supervised classification models**
* **evaluation with medical classification metrics**

### Classification Ecosystem

| Component               | Role                                           |
| ----------------------- | ---------------------------------------------- |
| RSS Parser              | Extract structured hospital diagnosis metadata |
| Clinical Record Builder | Consolidate per-admission text datasets        |
| Text Vectorization      | TF-IDF and hashing vectorizers                 |
| Embeddings              | Sentence-transformer clinical embeddings       |
| Model Training          | Supervised classifiers for ICD10 prediction    |
| Evaluation              | Medical classification metrics                 |

### Supported Models

| Model               | Type                         |
| ------------------- | ---------------------------- |
| Logistic Regression | Linear baseline              |
| Random Forest       | Ensemble tree model          |
| LightGBM            | Gradient boosting model      |
| FastText            | Efficient text classifier    |
| BiLSTM              | Deep learning sequence model |

---

## 🏗 Pipeline Architecture

```
RSS (.rss)
   ↓
Structured Parsing
   ↓
Consolidated ICD10 CSV
   ↓
Merge with Clinical Records (.txt)
   ↓
One CSV per admission_id
   ↓
Vectorization / Embeddings
   ↓
Model Training
   ↓
Evaluation & Metrics
   ↓
Exports & API
```

---

## 📊 Exploratory Data Analysis

The EDA module provides:

* ICD10 label distribution
* top-k most frequent codes
* clinical text length statistics
* dataset diagnostics

Outputs are exported to:

```
artifacts/exports/eda/
artifacts/reports/
```

---

## 🔧 Setup & Installation

In this section we explain the minimum OS verification, python usage and docker setup.

### 1. Requirements

* Python 3.10+
* Docker & Docker Compose
* Optional GPU (for BiLSTM)

### 2. OS prerequists

Verify that you have the necessairy packages installed.

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

### 3. Python environment

```bash
python -m venv .icd10_env
source .icd10_env/bin/activate   							## for windows : .icd10_env\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel		## for windows : .icd10_env\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Docker setup

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
```

---

## ▶️ Usage & End-to-End Testing

```bash
## Check raw inputs
ls data/raw/icd10
ls data/raw/clinical_records

## Inspect a sample RSS file
head -n 60 data/raw/icd10/sample.rss

## Verify FastText import (Windows uses fasttext-wheel)
python -c "import fasttext; print(fasttext.__file__)"

## Verify PyTorch (GPU optional)
python -c "import torch; print('cuda_available=', torch.cuda.is_available())"

## Parse RSS -> consolidated CSV
python main.py --parse-rss

## Inspect RSS structured output
ls data/interim/icd10_csv
head -n 5 data/interim/icd10_csv/icd10_structured.csv

## Build per-admission CSVs (merge RSS + clinical_records)
python main.py --build-clinical-csv

## Inspect per-admission CSV outputs
ls data/interim/clinical_records_csv
head -n 5 data/interim/clinical_records_csv/*.csv

## Train model + metrics
python main.py --train

## Inspect model + metrics
ls artifacts/models
ls artifacts/reports
cat artifacts/reports/metrics.json

## Run EDA
python main.py --eda

## Inspect EDA outputs
ls artifacts/exports/eda

## Run API
python main.py --run-api

## Run Full Pipeline
python main.py --run-all

## Run tests
pytest -q
```

---

## 📛 Common Errors & Troubleshooting

| Error                    | Cause                         | Solution                    |
| ------------------------ | ----------------------------- | --------------------------- |
| RSS parsing failure      | Incorrect RSS format          | Validate RSS schema         |
| Dataset merge failure    | Missing clinical records      | Verify admission_id folders |
| Model training error     | Insufficient training samples | Check dataset balance       |
| Docker container failure | Environment misconfiguration  | Rebuild containers          |

---

## 👤 Author

**Georges Nassopoulos**
[georges.nassopoulos@gmail.com](mailto:georges.nassopoulos@gmail.com)

**Status:** Clinical NLP / Medical AI Project
