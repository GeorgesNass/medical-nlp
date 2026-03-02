# 🧪 Laboratory Data Clustering

## 1. Project Overview

This project implements a complete **unsupervised clustering pipeline** for laboratory reports.

The objective is to:

- Parse heterogeneous laboratory `.txt` reports
- Extract structured analyte-level data
- Normalize units and reference intervals
- Build ML-ready datasets (wide / long format)
- Run unsupervised clustering
- Track experiments with MLflow
- Export structured artifacts and diagnostics

The pipeline transforms raw laboratory reports into structured datasets and cluster-based analytical insights.

---

## 2. Problem Statement

Laboratory reports are:

- Semi-structured `.txt` files
- Heterogeneous in format
- Containing variable units and reference ranges
- Containing implicit clinical normalization rules

Challenges:

- Complex regex extraction
- Unit normalization
- Reference interval interpretation
- Missing values
- High-dimensional data

This project addresses these constraints through:

- Regex-based analyte extraction
- Unit harmonization
- Reference interval comparison (low / normal / high)
- Dataset standardization (wide / long)
- Unsupervised clustering algorithms
- MLflow experiment tracking

---

## 3. Clustering Strategy

### Objective

Group laboratory profiles into homogeneous clusters based on:

- Numeric analyte values
- Normalized units
- Optional derived features

### Supported Algorithms

- KMeans
- Agglomerative Clustering
- DBSCAN
- Birch
- KModes

### Preprocessing

- Missing value imputation
- Standard scaling
- Optional PCA dimensionality reduction

---

## 4. Pipeline Architecture

```
Raw TXT (.txt)
    ↓
Metadata Extraction (demographics, dates, analysis group)
    ↓
Regex Extraction
    ↓
Structured CSV (one per file)
    ↓
Dataset Builder (wide / long)
    ↓
Preprocessing (impute + scale + PCA)
    ↓
Clustering
    ↓
Evaluation Metrics
    ↓
MLflow Tracking
    ↓
Exports (assignments, profiles, EDA, metadata)
```

---

## 5. Exploratory Data Analysis (EDA)

The EDA module provides:

- Analyte distribution analysis
- Missing values diagnostics
- Dataset dimensionality summary
- Cluster size analysis
- Statistical summaries

Outputs are exported in:

```
artifacts/exports/eda/
artifacts/exports/clustering/
```

---

## 6. Project Structure

```
lab-clustering/
├── main.py                              ## FastAPI entry point (minimal API: config, logging, routes, healthcheck)
├── menu_pipeline.sh                     ## Interactive CLI menu (parse TXT, build dataset, cluster, eval, export, run API)
├── requirements.txt
├── README.md
├── .env                                 ## Environment configuration (paths, MLflow tracking URI, clustering defaults)
│
├── docker/                              ## Container definition & service orchestration
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── logs/                                ## Centralized application logs
│
├── data/
│   ├── raw/                             ## Raw laboratory reports (.txt) (all files directly here)
│   │
│   ├── interim/
│   │   ├── lab_structured_csv/          ## One CSV per source_file (structured analytes extracted from TXT)
│   │   └── datasets/                    ## ML-ready datasets (wide/long formats in parquet/csv)
│   │
│   └── processed/
│       ├── features/                    ## Final clustering-ready features (scaled, imputed, reduced)
│       └── error_analysis/              ## Parsing failures, unmatched segments, regex debug dumps
│
├── artifacts/
│   ├── mlflow/                          ## MLflow experiment runs (run_id, params, metrics, artifacts)
│   ├── models/                          ## Saved clustering models (sklearn pipelines)
│   ├── resources/                       ## JSON/CSV resources (regex rules, norms, mappings)
│   │   ├── regex_labo.json
│   │   └── normes_labo.csv
│   │
│   ├── config/
│   │   └── swagger.yaml                 ## OpenAPI specification (local use)
│   │
│   └── exports/
│       ├── clustering/                  ## cluster_assignments.csv, cluster_profiles.csv, summaries
│       ├── eda/                         ## EDA plots and dataset diagnostics
│       ├── metadata/                    ## Config snapshots, schema versions
│       ├── metrics.json
│       ├── metrics.md
│       └── cluster_sizes.csv
│
├── tests/
│   └── test_unit.py                     ## Unit tests (TXT parsing, normalization, metrics, clustering smoke test)
│
└── src/
    ├── pipelines.py                     ## End-to-end orchestration (parse → dataset → cluster → eval → export)
    │
    ├── utils/
    │   ├── logging_utils.py             ## Centralized logging
    │   ├── io_utils.py                  ## Safe CSV / JSON / Parquet read-write helpers
    │   └── utils.py                     ## Generic utilities
    │
    ├── core/
	│   ├── __init__.py                  	
    │   ├── service.py                   ## FastAPI routes (/parse, /dataset, /cluster, /export, /health)
    │   ├── schema.py                    ## Pydantic request/response models
    │   ├── config.py                    ## Environment configuration + path resolution
    │   ├── eda.py                       ## Exploratory Data Analysis logic
    │   └── errors.py                    ## Centralized custom exceptions
    │
    ├── parser/
	│   ├── __init__.py                   
	│   ├── parse_txt.py                 ## Main TXT orchestrator (load → metadata → extraction → format)
	│   ├── extract_analytes.py          ## Segment text and detect candidate analyte lines
	│   ├── interpret_values.py          ## Interpret analyte line (value, unit, norms lookup, status)
	│   ├── unit_conversion.py           ## Normalize and convert measurement units
	│   ├── check_norms.py               ## Compute low/normal/high from numeric interval
	│   ├── regex_store.py               ## Load canonical resources (regex, keywords, norms, conversion)
	│   └── format_output.py             ## Align DataFrame to official CSV schema
	│
	├── metadata/                        
	│   ├── __init__.py                  
	│   ├── metadata_builder.py          ## Build consolidated metadata dict for one TXT file
	│   ├── dim_extractions.py           ## Extract gender, DOB and related demographic info
	│   ├── time_extractions.py          ## Extract sampling time and edition date
	│   └── analysis_group.py            ## Detect analysis group (biochimie, hemato, etc.)
    │
    └── clustering/
	    ├── __init__.py                  
        ├── build_dataset.py             ## Structured CSVs → wide/long dataset builder
        ├── preprocess.py                ## Imputation, scaling, optional PCA
        ├── algorithms.py                ## Unsupervised clustering algorithms
        ├── evaluate.py                  ## Unsupervised metrics + summaries
        ├── mlflow_tracking.py           ## MLflow logging helpers
        └── export.py                    ## Export assignments, profiles, plots
```

---

## 7. Prerequisites

- Python 3.10+
- Docker & Docker Compose (optional)
- No GPU required

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
python -m venv .lab_env
source .lab_env/bin/activate                                ## Windows: .lab_env\Scripts\activate.bat
pip install --upgrade pip                     				## for windows : .lab_env\Scripts\python.exe -m pip install --upgrade pip 
pip install "setuptools<82" "wheel<0.46" "packaging<24"     ## for windows : .lab_env\Scripts\python.exe -m pip install "setuptools<82" "wheel<0.46" "packaging<24" 
pip install -r requirements.txt
python -m pip check
```

### Docker

```bash
docker compose build
docker compose up
```

---

## ✅ Full System Verification (End-to-End)

```bash
## Check raw TXT inputs
ls data/raw

## Parse TXT → structured CSV
python main.py --parse-txt

## Inspect structured output
ls data/interim/lab_structured_csv
head -n 5 data/interim/lab_structured_csv/*.csv

## Build dataset (wide)
python main.py --build-dataset --dataset-format wide

## Inspect dataset
ls data/interim/datasets

## Run clustering
python main.py --cluster --algorithm kmeans --n-clusters 3

## Inspect clustering exports
ls artifacts/exports/clustering

## Run EDA
python main.py --eda

## Run API
python main.py --run-api

## Run full pipeline
python main.py --run-all

## Run tests
pytest -q
```

---

## Authors

**Olivia Tortosa**  
Email: olivia.tortosa@gmail.com

**Georges Nassopoulos**  
Email: georges.nassopoulos@gmail.com