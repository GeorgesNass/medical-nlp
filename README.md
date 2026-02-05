# 🏥 Clinical Named Entity Recognition (Clinical NER)

## 1. Project Overview

This project implements a **Clinical Named Entity Recognition (Clinical NER)** pipeline for medical texts with:

- **Dictionary-based entity extraction** (MeSH / custom dictionaries)
- **Concept normalization** (concept_id, concept_name, dictionary source)
- **Clinical context enrichment**
  - **Negation** (negated / not_negated / unknown)
  - **Temporality**
    - **MEDICATION**: past / current / chronic / future
    - **DISEASE (pathology)**: past / current / chronic (no future)
- **Optional model inference** (spaCy / Hugging Face)

The pipeline supports both:

- **Labeled mode**: consume a CSV with entities already annotated
- **Unlabeled mode**: read a folder of `.txt` documents, auto-label with dictionaries, enrich with negation + temporality, and export a labeled CSV

---

## 2. Problem Statement

Clinical documents often:

- Are unstructured and inconsistent
- Include ambiguous medical terminology requiring normalization
- Contain mentions that are **negated** (e.g., “denies asthma”)
- Mix **past**, **current**, and **chronic** conditions/medications

This project addresses these constraints by:

- Using **dictionary-based extraction** as a robust baseline for entity detection
- Enforcing a **strict data schema** for records and entities
- Making **negation** and **temporality** first-class fields
- Providing extension points for **ML inference** and future training pipelines

---

## 3. Extraction Strategy

### 3.1 Labeled + Unlabeled Compatibility

The code accepts two input formats.

A) Labeled input (CSV)

- One row per document record
- Required columns: `text`, `name_document`, `type_document`, `patient_id`, `record_id`, `date_document`
- Entities stored as JSON in a column named `entities`

B) Unlabeled input (folder of `.txt`)

- Dictionary auto-labeling using files from `artifacts/dictionaries/`
- Negation and temporality inference applied after extraction
- Export to a CSV that follows the same canonical schema as labeled mode

---

### 3.2 Entity Output Format (JSON in `entities` column)

Each entity stored in the `entities` JSON list includes:

- `id`
- `text`
- `start`
- `end`
- `label`
- `concept_id`
- `concept_name`
- `dictionary`
- `source`
- `confidence`
- `negation`
- `temporality`
- `meta`

---

## 4. Pipeline Architecture

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

## 5. Exploratory Data Analysis (EDA)

An EDA module can be added later to analyze corpora:

- Number of documents
- Average length
- Entity distribution by label
- Negation distribution
- Temporality distribution

Outputs are typically stored as JSON / CSV in:

```text
artifacts/reports/
```

---

## 6. Project Structure

```text
clinical-ner/
├── README.md
├── requirements.txt
├── .env
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── menu_pipeline.sh
├── main.py
├── artifacts/
│   ├── models/
│   ├── reports/
│   ├── exports/
│   └── dictionaries/
├── data/
│   ├── raw/
│   ├── annotated/
│   ├── interim/
│   └── processed/
├── logs/
├── tests/
│   └── test_unit.py
└── src/
    ├── __init__.py
    ├── utils/
    │   ├── __init__.py
    │   ├── logging_utils.py
    │   └── utils.py
    ├── core/
    │   ├── __init__.py
    │   ├── entities.py
    │   ├── schema.py
    │   ├── config.py
    │   └── errors.py
    ├── nlp/
    │   ├── __init__.py
    │   ├── normalization.py
    │   ├── rules.py
    │   └── tokenizer.py
    ├── model/
    │   ├── __init__.py
    │   ├── spacy_train.py
    │   ├── hf_train.py
    │   ├── inference.py
    │   └── metrics.py
    └── pipeline.py
```

---

## 7. Prerequisites

### General

- Python **3.10+** (recommended: 3.11)
- Docker and Docker Compose (optional)
- Optional GPU (CUDA) for HF inference

### Windows / WSL2 (optional)

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
sudo apt install -y git python3-pip python3-venv
git --version
python3 --version
```

---

## 8. Setup

### Manual installation

```bash
python -m venv .clinical_ner_env
source .clinical_ner_env/bin/activate ## .venv\Scripts\activate.bat for Windows
pip install --upgrade pip
pip install -r requirements.txt
```

### Docker Usage

Build and run:

```bash
docker compose build
docker compose up
```

---

## 9. CLI Usage

### Run pipeline (labeled CSV)

```bash
python main.py --labeled-csv data/annotated/labeled.csv --output-csv artifacts/exports/out.csv
```

### Run pipeline (unlabeled .txt folder)

```bash
python main.py --unlabeled-texts data/raw --output-csv artifacts/exports/out.csv
```

### Interactive menu

```bash
bash menu_pipeline.sh
```

---

## 10. Tests

```bash
pytest -q
```

---

## ✅ Full System Verification (End-to-End)

Run the following commands in order:

```bash
# Check inputs
ls data/raw
ls artifacts/dictionaries

# Run pipeline
python main.py --unlabeled-texts data/raw --output-csv artifacts/exports/clinical_ner_records.csv

# Inspect outputs
ls artifacts/exports
head -n 5 artifacts/exports/clinical_ner_records.csv

# Run tests
pytest -q
```

---

## Author

**Georges Nassopoulos**  
Email: georges.nassopoulos@gmail.com

**Status:** Research / Professional NLP project

---