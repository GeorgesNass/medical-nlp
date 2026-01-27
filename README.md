# MeSH Semantic Expansion – NLP & RAG Pipeline

## 📘 Overview
FastAPI-based project for **semantic expansion of medical vocabularies (MeSH)** from unstructured medical documents.

The project is designed to:
- Explore and query the **MeSH ontology**
- Detect medical entities in documents
- Identify **synonyms, abbreviations and candidate terms**
- Assist **human validation**
- Build an **extended MeSH** aligned with existing concepts or enriched with new ones

It demonstrates:
- NLP pipelines (NER, embeddings, semantic similarity)
- SQLite FTS + FAISS indexing
- Human-in-the-loop workflows
- FastAPI services
- Dockerized deployment

---

## 📂 Project Structure
```text
mesh_semantic_expansion/
├── main.py
├── menu_pipeline.sh
├── requirements.txt
├── README.md
├── .env
├── docker/
│ ├── Dockerfile
│ └── docker-compose.yml
├── logs/
├── data/
│ ├── raw/
│ │ ├── mesh/
│ │ └── medical_docs/
│ ├── interim/
│ │ ├── mesh_parsed.jsonl
│ │ ├── doc_embeddings.parquet
│ │ └── mesh_embeddings.parquet
│ ├── processed/
│ │ ├── entities_detected.jsonl
│ │ └── candidates.jsonl
│ └── outputs/
│ ├── export_candidates.csv
│ ├── export_candidates_validated.csv
│ ├── mesh_extended.json
│ └── report_diff.md
├── src/
│ ├── pipelines.py
│ ├── service/
│ │ ├── routes_mesh.py
│ │ └── routes_expand.py
│ ├── core/
│ │ ├── config.py
│ │ └── models.py
│ ├── mesh/
│ │ ├── download_mesh.py
│ │ ├── parse_mesh.py
│ │ ├── index_mesh.py
│ │ └── query_mesh.py
│ ├── nlp/
│ │ ├── ner_mesh.py
│ │ ├── embeddings.py
│ │ ├── expand_terms.py
│ │ └── judge_quality.py
│ └── utils/
│ ├── utils_cli.py
│ └── logging_utils.py
└── tests/
├── test_mesh.py
└── test_unit.py
```
---

## 🖥️ Pipeline Overview

1. Download & parse MeSH (XML → JSONL)
2. Index MeSH (SQLite FTS + optional FAISS)
3. Detect entities in medical documents
4. Extract synonyms, abbreviations and candidate terms
5. Human validation via CSV
6. Build extended MeSH + diff report

---

## Prerequisites

### General
- Python ≥ 3.10
- Docker & Docker Compose (optional)

---

## Windows & WSL2 Prerequisites

Windows users should use **WSL2 (Ubuntu)**.

```bash
wsl --install
```
Inside WSL:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential curl git
```
Running embeddings and FAISS directly on Windows without WSL is not recommended.

---

## ⚙️ Setup

Local installation:
```bash
python -m venv .venv
source .venv/bin/activate ## .venv\Scripts\activate.bat for Windows
(.venv) python -m pip install --upgrade pip setuptools wheel
(.venv) pip install -r requirements.txt
(.venv) pip install fasttext-wheel==0.9.2
(.venv) python -c "import fasttext; print(fasttext.__file__)"
```
### Windows FastText Setup (Recommended)

FastText source compilation is unstable on Windows due to C++ toolchain and
`ssize_t` compatibility issues with MSVC.

To ensure a reliable installation, this project uses a **precompiled wheel**
instead of building FastText from source.

### Docker

```bash
docker compose up --build
```
---


## ✅ Full System Verification (End-to-End)

This section describes how to verify that **the entire MeSH Semantic Expansion system**
is correctly installed and fully functional: data, indexing, NLP, tests, and API.

Run the commands **in the order below**.

```bash
# ------------------------------------------------------------
# MeSH DATA PIPELINE CHECK
# ------------------------------------------------------------

## List MeSH raw directory (should contain desc2025.xml)
dir data\raw\mesh

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


# ------------------------------------------------------------
# NLP / EMBEDDINGS CHECK
# ------------------------------------------------------------

## Check FastText installation
python -c "import fasttext; print(fasttext.__file__)"

## Check embedding backend configuration
python -c "from src.nlp.embeddings import get_embedding_config; print(get_embedding_config().backend)"

## Test single-text embedding
python -c "from src.nlp.embeddings import embed_query_text; print(embed_query_text('myocardial infarction').shape)"

## Test batch embeddings
python -c "from src.nlp.embeddings import embed_texts; print(embed_texts(['heart attack','diabetes']).shape)"

# ------------------------------------------------------------
# TESTS & API CHECK
# ------------------------------------------------------------

## Run unit tests (all tests must pass)
pytest -q

## Start FastAPI application
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

## Check API health endpoint (run in another terminal)
curl -s http://localhost:8000/healthcheck
```
---



## 🧪 CLI Pipeline (Interactive)
```bash
chmod +x menu_pipeline.sh
./menu_pipeline.sh
```
Available actions:
- Download MeSH
- Parse MeSH
- Build SQLite / FAISS indexes
- Extract candidates
- Build extended MeSH
- Run API

---


## 🏁 API Overview

Endpoint | Method | Description | Purpose
/healthcheck | GET | Service status | Check API availability
/mesh/search | POST | Full-text MeSH search | Query MeSH concepts
/mesh/browse | POST | Browse by tree prefix | Navigate MeSH hierarchy
/mesh/lookup/{ui} | GET | Lookup MeSH concept | Retrieve concept details
/expand/extract_candidates | POST | Run expansion pipeline | Extract candidate terms

---

## 🧪 Testing

Run all tests:
```bash
pytest tests/
```
tests/test_mesh.py – End-to-End Tests
- MeSH parsing: JSONL generation from XML
- SQLite FTS: Index creation and querying
- Search: FTS search correctness
- Lookup: Retrieval by MeSH UI
- Browse: Tree-based navigation

This file validates a full MeSH lifecycle:
JSONL → SQLite FTS → Queries

tests/test_unit.py – Unit Tests
- judge_baseline: Label acceptance / rejection logic
- Abbreviations: (LONG FORM (ABBR)) detection
- Entity detection: Dictionary-based matching
- Embeddings: Output shape & robustness
- FastText: Safe handling of empty inputs

---

## 🌟 Notes
- Human validation is intentional (medical domain constraints)
- Embedding backends are interchangeable via .env
- Designed for RAG, decision-support and terminology enrichment use cases

---

## Author
Georges Nassopoulos
Email: georges.nassopoulos@gmail.com
Status: Research / Professional NLP project
