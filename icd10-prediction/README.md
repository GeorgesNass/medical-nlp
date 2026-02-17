# 🏥 ICD10 Prediction from Clinical Records

## 1. Project Overview

This project implements a complete **ICD10 prediction pipeline** from raw hospital data.

The objective is to automatically predict the **primary ICD10 diagnosis code** associated with a hospital admission using:

- Structured RSS hospital metadata
- Raw clinical text documents
- Machine learning models (classical + deep learning)

The pipeline transforms heterogeneous hospital data into a structured dataset and trains predictive models for ICD10 classification.

---

## 2. Problem Statement

Hospital data is distributed across:

- RSS export files containing administrative and diagnosis information
- Clinical document folders per admission ID
- Multiple document types per admission

Challenges:

- Heterogeneous data formats (.rss, .txt)
- Multiple document types per admission
- Sensitive medical content
- Class imbalance in ICD10 codes

This project addresses these constraints through:

- Structured RSS parsing
- Per-admission CSV consolidation
- Text vectorization and embedding
- Supervised classification models
- Clean pipeline orchestration

---

## 3. Classification Strategy

### Primary Diagnosis Prediction

Each hospital admission is associated with:

- A unique admission_id
- A primary_diagnosis_code (ICD10)

The objective is:

> Predict the primary ICD10 code using all associated clinical text.

### Supported Models

The training module supports:

- Logistic Regression
- Random Forest
- LightGBM
- FastText (supervised text classifier)
- BiLSTM (PyTorch)

---

## 4. Pipeline Architecture

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

## 5. Exploratory Data Analysis (EDA)

The EDA module provides:

- ICD10 label distribution
- Top-k most frequent codes
- Text length statistics
- Dataset diagnostics

Outputs are exported in:

```
artifacts/exports/eda/
artifacts/reports/
```

---

## 6. Project Structure

```
icd10_prediction/
├── main.py                        		## FastAPI entry point (minimal API: config, logging, routes, healthcheck)
├── menu_pipeline.sh               		## Interactive CLI menu (parse RSS, build CSV, train, eval, predict, export, run API)
├── requirements.txt               
├── README.md                      
├── .env                           		## Environment configuration (paths, GPU, thresholds, etc.)
│
├── docker/                        		## Container definition & service orchestration (API, volumes)
│   ├── Dockerfile                 
│   └── docker-compose.yml         
│
├── logs/                          		## Centralized application logs (auto-created via logging_utils)
│
├── data/
│   ├── raw/                       
│   │   ├── clinical_records/      		## One folder per admission_id (hospital stay, match RSS)
│   │   └── icd10/                  	##  Raw .rss files (structured medical coding information)
│   │
│   ├── interim/
│   │   ├── clinical_records_csv/  		## One CSV per admission_id (RSS fields, document types, file name, text content)
│   │   ├── icd10_csv/                  ## Single consolidated CSV parsed from all .rss files (Ordered by year)
│   │   ├── datasets/                   ## ML-ready datasets (train/val/test in parquet format)
│   │   └── embeddings/                 ## Optional cached embeddings (if transformer models used)
│   │
│   ├── processed/                   
│   │   ├── features/                   ## Final vectorized features (TF-IDF, FastText, embeddings)
│   │   ├── labels/                     ## Encoded diagnosis labels (primary_diagnosis_code)
│   │   └── error_analysis/             ## False positives/negatives and misclassification dumps
│
├── artifacts/
│   ├── models/                         ## Trained models (LR, RF, LightGBM, FastText, BiLSTM, etc.)
│   ├── metadata/                       ## Label encoders, vectorizers, config snapshots, mappings
│   ├── predictions/                    ## Raw prediction outputs (jsonl/parquet)
│   ├── exports/
│   │   ├── review.csv                  ## Human validation file (top-k ICD10 codes + confidence)
│   │   ├── validated.md                ## Manual validation notes and adjustments
│   │   └── eda/                        ## EDA plots and dataset diagnostics
│   │
│   └── reports/                        ## evaluation metrics, evaluation report, Most frequent ICD10 confusions
│       ├── metrics.json            
│       ├── metrics.md              
│       └── confusion_top_codes.csv  
│
├── tests/                              ## Unit tests (RSS parsing, hashtag extraction, metrics, taxonomy, End-to-end smoke test)
│   └── test_unit.py                
│
└── src/
    ├── pipelines.py                    ## End-to-end orchestration logic (parse → merge → train → eval → export)
    ├── __init__.py
    │
    ├── utils/
    │   ├── __init__.py
    │   ├── logging_utils.py            ## Centralized logging (no print statements)
    │   ├── io_utils.py                 ## Safe CSV / JSONL / Parquet read-write helpers
    │
    ├── core/
    │   ├── __init__.py
    │   ├── service.py                  ## FastAPI routes (/predict, /topk, /batch, /health, /models)
    │   ├── schema.py                   ## Pydantic request/response models
    │   ├── config.py                   ## Environment configuration + path resolution + run_id
    │   ├── eda.py                      ## Exploratory Data Analysis logic
    │   └── errors.py                   ## Centralized custom exceptions
    │
    ├── nlp/
    │   ├── __init__.py
    │   ├── preprocess.py               ## Text normalization and minimal cleaning (no content loss)
    │   ├── vectorizers.py              ## TF-IDF / hashing-based vectorization
    │   ├── embeddings.py               ## Sentence-transformers / clinical embedding models (optional)
    │   └── postprocess.py              ## Thresholding, top-k selection, calibration logic
    │
    ├── model/
    │   ├── __init__.py
    │   ├── train.py                    ## Model training (LogReg, RF, LightGBM, FastText, BiLSTM)
    │   ├── evaluate.py                 ## Evaluation metrics (micro/macro F1, Precision@k, Recall@k)
    │   ├── predict.py                  ## Inference wrapper (single & batch)
    │   ├── calibrate.py                ## Optional probability calibration
    │   └── explain.py                  ## Feature importance / attention visualization
    │
    └── icd10/
        ├── __init__.py
        ├── build_clinical_csv.py       ## structured rss info with text content records per admission_id	
        ├── parse_rss.py                ## Clean fixed-width RSS parser ==> structured records
        ├── index_icd10.py              ## Optional SQLite / FTS index for ICD10 code lookup
        └── taxonomy.py                 ## ICD10 hierarchy utilities (parent/child relations)
```

---

## 7. Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Optional GPU (for BiLSTM)

### Ubuntu Example

```bash
sudo apt update
sudo apt install python python3-pip
python --version
```

---

## 8. Setup

### Python

```bash
python -m venv .icd10_env
source .icd10_env/bin/activate   							## for windows : .icd10_env\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel		## for windows : .icd10_env\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Docker

```
docker compose build
docker compose up
```

---

## ✅ Full System Verification (End-to-End)

Run the following commands in order:

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

## Author

**Georges Nassopoulos**  
Email: georges.nassopoulos@gmail.com
