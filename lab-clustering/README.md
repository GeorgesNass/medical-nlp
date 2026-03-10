# 🧪 Laboratory Data Clustering

The pipeline transforms heterogeneous laboratory `.txt` reports into **structured datasets and cluster-based analytical insights**.

---

## 🎯 Project Overview


Main capabilities:

* Parse heterogeneous laboratory `.txt` reports
* Extract structured analyte-level data
* Normalize measurement units and reference intervals
* Build ML-ready datasets (wide / long format)
* Run unsupervised clustering algorithms
* Track experiments with **MLflow**
* Export structured artifacts and diagnostics

---

## ⚙️ Tech Stack

Core technologies used in the project:

* Python
* FastAPI
* Docker & Docker Compose
* Scikit-learn
* MLflow
* Pandas / NumPy
* Regex-based parsing
* PCA for dimensionality reduction

---

## 📂 Project Structure

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

## ❓ Problem Statement

Laboratory reports are typically **semi-structured `.txt` files** containing heterogeneous formats and implicit normalization rules.

Key challenges:

* Complex **regex extraction** of analytes
* Unit normalization across laboratories
* Interpretation of reference intervals
* Missing values and high-dimensional feature space
* Conversion to ML-ready structured datasets

This project addresses these constraints through:

* Regex-based analyte extraction
* Unit harmonization
* Reference interval interpretation (low / normal / high)
* Dataset standardization (wide / long)
* Unsupervised clustering analysis
* Experiment tracking with MLflow

---

## 🧠 Approach / Methodology / Strategy

The pipeline converts raw laboratory reports into clustering-ready datasets through structured preprocessing and unsupervised learning.

Core principles:

* **Regex-based extraction** of analytes from semi-structured reports
* **Unit harmonization** and reference interval interpretation
* **Dataset standardization** in wide and long formats
* **Feature preprocessing** (imputation, scaling, dimensionality reduction)
* **Experiment tracking** using MLflow

### Clustering Ecosystem

| Component             | Role                                            |
| --------------------- | ----------------------------------------------- |
| TXT Parser            | Extract analytes, values, units, and metadata   |
| Unit Normalization    | Harmonize measurement units                     |
| Dataset Builder       | Generate wide and long ML-ready datasets        |
| Preprocessing         | Missing value imputation, scaling, optional PCA |
| Clustering Algorithms | KMeans, Agglomerative, DBSCAN, Birch, KModes    |
| Evaluation            | Unsupervised clustering metrics                 |
| MLflow Tracking       | Experiment logging and artifact storage         |

---

## 🏗 Pipeline Architecture

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

## 📊 Exploratory Data Analysis

The EDA module provides:

* analyte distribution analysis
* missing value diagnostics
* dataset dimensionality analysis
* cluster size distribution
* statistical summaries

Outputs are exported to:

```
artifacts/exports/eda/
artifacts/exports/clustering/
```

---

## 🔧 Setup & Installation
In this section we explain the minimum OS verification, python usage and docker setup.

### 1. Requirements

* Python 3.10+
* Docker & Docker Compose (optional)
* No GPU required

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
python3 --version
```

### 3. Python environment

```bash
python -m venv .lab_env
source .lab_env/bin/activate                                ## Windows: .lab_env\Scripts\activate.bat
pip install --upgrade pip                     				## for windows : .lab_env\Scripts\python.exe -m pip install --upgrade pip 
pip install "setuptools<82" "wheel<0.46" "packaging<24"     ## for windows : .lab_env\Scripts\python.exe -m pip install "setuptools<82" "wheel<0.46" "packaging<24" 
pip install -r requirements.txt
python -m pip check
```

### 4. Docker setup

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
```

---

## ▶️ Usage & End-to-End Testing

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

## 📛 Common Errors & Troubleshooting

| Error                    | Cause                      | Solution                   |
| ------------------------ | -------------------------- | -------------------------- |
| Regex extraction failure | Unexpected TXT format      | Update regex rules         |
| Missing analyte values   | Incomplete report data     | Verify parsing logic       |
| MLflow tracking error    | Wrong tracking URI         | Check `.env` configuration |
| Dataset build failure    | Inconsistent analyte units | Review normalization rules |

---

## 👤 Author

**Olivia Tortosa**
[olivia.tortosa@gmail.com](mailto:olivia.tortosa@gmail.com)

**Georges Nassopoulos**
[georges.nassopoulos@gmail.com](mailto:georges.nassopoulos@gmail.com)

**Status:** Research / Data Science project
