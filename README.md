# 🩺 Clinical Named Entity Recognition (Clinical NER)

The project implements a **clinical NLP pipeline for Named Entity Recognition in medical documents**, supporting both **dictionary-based extraction and optional machine-learning inference**.

---

## 🎯 Project Overview

Main capabilities:

* Dictionary-based entity extraction (MeSH / custom dictionaries)
* Concept normalization (`concept_id`, `concept_name`, dictionary source)
* Clinical context enrichment
  * **Negation detection** (negated / not_negated / unknown)
  * **Temporality inference** past / current / chronic / future
* Optional model inference (spaCy / HuggingFace)
* Support for labeled and unlabeled datasets
* Export of structured datasets for downstream ML tasks

The pipeline converts raw clinical documents into **structured datasets enriched with contextual medical information**.

The project supports both:

* **Labeled mode**: consume a CSV with entities already annotated
* **Unlabeled mode**: read a folder of `.txt` documents, auto-label with dictionaries, enrich with negation + temporality, and export a labeled CSV

---

## ⚙️ Tech Stack

Core technologies used in the project:

* Python
* spaCy
* HuggingFace Transformers
* Docker & Docker Compose
* Dictionary-based NLP (MeSH / custom lexicons)
* Rule-based negation detection
* Rule-based temporality inference

---

## 📂 Project Structure

```text
clinical-ner/
├── README.md                     ## Project documentation
├── requirements.txt              ## Python dependencies
├── .env                          ## Environment configuration
├── menu_pipeline.sh              ## Interactive CLI pipeline launcher
├── main.py                       ## Pipeline entry point
│
├── docker/
│   ├── Dockerfile                ## Container definition
│   └── docker-compose.yml        ## Docker orchestration
│
├── artifacts/
│   ├── models/                   ## Trained models
│   ├── reports/                  ## Analytics reports
│   ├── exports/                  ## Exported datasets
│   └── dictionaries/             ## Dictionary resources
│
├── data/
│   ├── raw/                      ## Raw clinical documents
│   ├── annotated/                ## Labeled training data
│   ├── interim/                  ## Intermediate datasets
│   └── processed/                ## Final structured datasets
│
├── logs/                         ## Runtime logs
│
├── tests/
│   └── test_unit.py              ## Unit tests
│
└── src/
    ├── pipeline.py               ## Pipeline orchestration
    │
    ├── utils/
    │   ├── logging_utils.py      ## Logging utilities
    │   └── utils.py              ## Shared helpers
    │
    ├── core/
    │   ├── entities.py           ## Entity representation
    │   ├── schema.py             ## Data schema validation
    │   ├── config.py             ## Environment configuration
    │   └── errors.py             ## Custom exceptions
    │
    ├── nlp/
    │   ├── normalization.py      ## Text normalization
    │   ├── rules.py              ## Negation and temporality rules
    │   └── tokenizer.py          ## Tokenization utilities
    │
    └── model/
        ├── spacy_train.py        ## spaCy training pipeline
		├── hf_train.py           ## HuggingFace training pipeline
		├── inference.py          ## Model inference
		└── metrics.py            ## Evaluation metrics
```

---

## ❓ Problem Statement

Clinical documents often present several challenges:

* unstructured and heterogeneous text
* ambiguous medical terminology
* negated medical conditions
* mixed temporal context (past / current / chronic)

This project addresses these issues by:

* using **dictionary-based extraction as a robust baseline**
* enforcing a **strict schema for entity records**
* enriching entities with **negation and temporality**
* providing extension points for **ML-based NER models**

---

## 🧠 Approach / Methodology / Strategy

The pipeline supports **two operating modes**, labeled and unlabeled.

### Extraction Strategy

**Labeled mode**

* Consume CSV records containing annotated entities
* Preserve entity metadata and enrich with contextual information

**Unlabeled mode**

* Read `.txt` documents
* Apply dictionary-based auto-labeling
* Apply negation and temporality inference
* Export labeled dataset

---

### Entity Output Format

Entities are stored as JSON objects inside the `entities` column.

| Field | Type | Description |
|------|------|-------------|
| id | string | Unique identifier of the extracted entity |
| text | string | Exact text span corresponding to the entity |
| start | integer | Character start position of the entity in the document |
| end | integer | Character end position of the entity in the document |
| label | string | Entity category (e.g. DISEASE, MEDICATION) |
| concept_id | string | Normalized identifier of the concept in the dictionary |
| concept_name | string | Canonical name of the normalized medical concept |
| dictionary | string | Source dictionary used for normalization (e.g. MeSH) |
| source | string | Extraction source (dictionary, rule, or model inference) |
| confidence | float | Confidence score of the extraction or prediction |
| negation | string | Negation status (negated / not_negated / unknown) |
| temporality | string | Temporal context (past / current / chronic / future) |
| meta | object | Optional metadata associated with the entity |

---

## 🏗 Pipeline Architecture

```text
Document (.txt) OR Labeled CSV
   ↓
Load + Schema Validation
   ↓
Dictionary Auto-Labeling (MeSH / custom)
   ↓
Negation Detection (rules)
   ↓
Temporality Inference (rules)
   ↓
Optional Model Inference (spaCy / HF)
   ↓
CSV Export (entities as JSON) + Reports (optional)
```

---

## 📊 Exploratory Data Analysis

EDA can be applied to analyze clinical corpora:

* document statistics
* entity frequency distribution
* negation statistics
* temporality distribution

Outputs are typically exported to:

```
artifacts/reports/
```

---

## 🔧 Setup & Installation

In this section we explain the minimum OS verification, python usage and docker setup.

### 1. Requirements

* Python 3.10+
* Docker & Docker Compose
* Optional GPU (CUDA)

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
python -m venv .clinical_ner_env
source .clinical_ner_env/bin/activate 		## for windows : .clinical_ner_env\Scripts\activate.bat
pip install --upgrade pip                   ## for windows : .clinical_ner_env\Scripts\python.exe -m pip install --upgrade pip
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
## Run pipeline with labeled dataset
python main.py --labeled-csv data/annotated/labeled.csv --output-csv artifacts/exports/out.csv

## Run pipeline with unlabeled text documents
python main.py --unlabeled-texts data/raw --output-csv artifacts/exports/out.csv

## Interactive pipeline menu
bash menu_pipeline.sh

## Verify outputs
ls artifacts/exports
head -n 5 artifacts/exports/out.csv

## Run tests
pytest -q
```

---

## 📛 Common Errors & Troubleshooting

| Error                    | Cause                     | Solution                                  |
| ------------------------ | ------------------------- | ----------------------------------------- |
| Dictionary loading error | Missing dictionary files  | Verify `artifacts/dictionaries/` contents |
| CSV schema mismatch      | Missing required columns  | Validate dataset schema                   |
| Model inference failure  | Missing ML dependencies   | Install spaCy / transformers packages     |
| Docker container failure | Misconfigured environment | Rebuild containers                        |

---

## 👤 Author

**Georges Nassopoulos**
[georges.nassopoulos@gmail.com](mailto:georges.nassopoulos@gmail.com)

**Status:** Clinical NLP / Medical AI Project
