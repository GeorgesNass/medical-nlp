# 🧬 MeSH Semantic Expansion – NLP & RAG Pipeline

FastAPI-based project for **semantic expansion of the MeSH medical ontology** from unstructured medical documents.

---

## 🎯 Project Overview

Main capabilities:

* Explore and query the **MeSH ontology**
* Detect medical entities in documents
* Extract synonyms and abbreviations
* Generate candidate terms using semantic similarity
* Human-in-the-loop validation workflow
* Build extended MeSH aligned with existing concepts

The system identifies **synonyms, abbreviations and candidate medical terms** and assists human validation to build an **extended MeSH vocabulary**.

---

## ⚙️ Tech Stack

Core technologies used in the project:

* Python
* FastAPI
* SQLite FTS
* FAISS vector indexing
* FastText embeddings
* NLP pipelines (NER + similarity)
* Docker & Docker Compose

---

## 📂 Project Structure

```text
mesh_semantic_expansion/
├── main.py                        				## FastAPI application entrypoint
├── menu_pipeline.sh               				## Interactive CLI pipeline
├── requirements.txt               				## Python dependencies
├── README.md                     	 			## Project documentation
├── .env                           				## Environment configuration
│
├── docker/
│   ├── Dockerfile                 				## Docker image definition
│   └── docker-compose.yml         				## Docker orchestration
│
├── logs/                          				## Application logs
│
├── data/
│   ├── raw/
│   │   ├── mesh/                  				## MeSH XML files
│   │   └── medical_docs/          				## Source medical documents
│   │
│   ├── interim/
│   │   ├── mesh_parsed.jsonl      				## Parsed MeSH ontology
│   │   ├── doc_embeddings.parquet 				## Document embeddings
│   │   └── mesh_embeddings.parquet				## MeSH embeddings
│   │
│   ├── processed/
│   │   ├── entities_detected.jsonl				## Extracted medical entities
│   │   └── candidates.jsonl       				## Candidate MeSH expansions
│   │
│   └── outputs/
│       ├── export_candidates.csv
│       ├── export_candidates_validated.csv
│       ├── mesh_extended.json
│       └── report_diff.md
│
├── tests/            							## End-to-end unit tests
│   └── test_unit.py               			    
│
└── src/
	├── pipelines.py              				## Pipeline orchestration
	│
	├── service/
	│   ├── routes_mesh.py         				## MeSH API endpoints
	│   └── routes_expand.py       				## Expansion endpoints
	│
	├── core/
	│   ├── config.py              				## Application configuration
	│   └── models.py              				## Data models
	│
	├── mesh/
	│   ├── download_mesh.py       				## MeSH download utility
	│   ├── parse_mesh.py          				## XML → JSONL parser
	│   ├── index_mesh.py          				## SQLite / FAISS indexing
	│   └── query_mesh.py          				## MeSH query utilities
	│
	├── nlp/
	│   ├── ner_mesh.py            				## Entity recognition
    ├── embeddings.py          				## Embedding generation
	│   ├── expand_terms.py        				## Candidate generation
	│   └── judge_quality.py       				## Candidate validation logic
	│
    └── utils/
	    ├── utils_cli.py           				## CLI helpers
        └── logging_utils.py       				## Logging utilities

```

---

## ❓ Problem Statement

Medical vocabularies such as **MeSH** are essential for clinical information systems.

However:

* medical language evolves rapidly
* synonyms and abbreviations are common
* new terminology appears frequently in clinical documents

This project addresses these challenges by:

* extracting candidate terms from documents
* measuring semantic similarity with existing MeSH concepts
* enabling **human validation**
* generating **extended MeSH vocabularies**

---

## 🧠 Approach / Methodology / Strategy

The system implements a **human-in-the-loop terminology expansion pipeline**.

### Ontology Preparation

* Download MeSH ontology XML
* Parse into structured JSONL
* Build **SQLite FTS and optional FAISS index**

---

### Entity Detection

* Detect medical entities in documents
* Extract candidate terms
* Identify synonyms and abbreviations

---

### Candidate Expansion

* Generate candidate terms using embeddings
* Rank candidates by semantic similarity
* Export validation datasets

---

### Human Validation

* Candidate terms are reviewed manually
* Accepted terms are integrated into **extended MeSH**

---

## 🏗 Pipeline Architecture

```text id="q8dx3k"
MeSH XML
      ↓
XML Parsing
      ↓
Ontology Indexing (FTS + FAISS)
      ↓
Medical Document Processing
      ↓
Entity Detection
      ↓
Candidate Extraction
      ↓
Semantic Similarity Ranking
      ↓
Human Validation
      ↓
Extended MeSH
```

---

## 📊 Exploratory Data Analysis

Diagnostics include:

* number of MeSH concepts
* entity detection statistics
* candidate term frequency
* similarity score distribution

Outputs are exported to:

```
data/outputs/
logs/
```

---

## 🔧 Setup & Installation

In this section we explain the minimum OS verification, python usage and docker setup.

### 1. Requirements

* Python ≥ 3.10
* Docker & Docker Compose (optional)

---

### 2. OS prerequisites

Verify that required packages are installed.

#### Windows / WSL2 (recommended)

```powershell
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

---

### 3. Python environment

```bash
python -m venv .mesh_env
source .mesh_env/bin/activate							## for windows : .mesh_env\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel    ## for windows : .mesh_env\Scripts\python.exe -m pip install --upgrade pip setuptools wheel 
pip install -r requirements.txt
pip install fasttext-wheel==0.9.2
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

## ------------------------------------------------------------
## MeSH DATA PIPELINE CHECK
## ------------------------------------------------------------

## List MeSH raw directory (should contain desc2025.xml)
ls data/raw/mesh

## Download official MeSH 2025 XML (NLM-only, raw XML)
curl -L "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/NLM_only/desc2025.xml" -o data/raw/mesh/desc2025.xml

## Verify downloaded MeSH file is valid XML
python -c "from pathlib import Path; p=Path('data/raw/mesh/desc2025.xml'); print(p.stat().st_size); print(p.read_text(encoding='utf-8', errors='replace')[:120])"

## Parse MeSH XML into JSONL
python -c "from src.utils.utils_cli import cmd_parse_mesh; print(cmd_parse_mesh(overwrite=True))"

## Build SQLite FTS index from MeSH JSONL
python -c "from src.utils.utils_cli import cmd_index_sqlite; print(cmd_index_sqlite(overwrite=True))"

## Test MeSH full-text search (FTS)
python -c "from src.mesh.query_mesh import search_mesh; r=search_mesh('Hypertension', limit=5); print(len(r), r[0]['ui'], r[0]['preferred_terms'])"

## Inspect a full MeSH record by UI
python -c "from src.mesh.query_mesh import lookup_ui; import json; print(json.dumps(lookup_ui('D065627'), ensure_ascii=False, indent=2)[:1200])"

## ------------------------------------------------------------
## NLP / EMBEDDINGS CHECK
## ------------------------------------------------------------

## Check FastText installation
python -c "import fasttext; print(fasttext.__file__)"

## Check embedding backend configuration
python -c "from src.nlp.embeddings import get_embedding_config; print(get_embedding_config().backend)"

## Test single-text embedding
python -c "from src.nlp.embeddings import embed_query_text; print(embed_query_text('myocardial infarction').shape)"

## Test batch embeddings
python -c "from src.nlp.embeddings import embed_texts; print(embed_texts(['heart attack','diabetes']).shape)"

## ------------------------------------------------------------
## TESTS & API CHECK
## ------------------------------------------------------------

## Run unit tests (all tests must pass)
pytest -q

## Start FastAPI application
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

## Check API health endpoint (run in another terminal)
curl -s http://localhost:8000/healthcheck
```

---

## 📛 Common Errors & Troubleshooting

| Error                         | Cause                               | Solution                         |
| ----------------------------- | ----------------------------------- | -------------------------------- |
| MeSH parsing failure          | Invalid XML file                    | Verify downloaded MeSH XML       |
| FastText installation failure | Windows build issues                | Use precompiled wheel            |
| FAISS indexing error          | Missing FAISS dependency            | Install FAISS compatible version |
| API startup failure           | Incorrect environment configuration | Verify `.env` variables          |

---

## 👤 Author

**Georges Nassopoulos**
[georges.nassopoulos@gmail.com](mailto:georges.nassopoulos@gmail.com)

**Status:** Medical NLP / Ontology Expansion Project
