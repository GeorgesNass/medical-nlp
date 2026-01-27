#!/usr/bin/env bash

###############################################################################
# MeSH Semantic Expansion - Pipeline Menu
# Author: Georges Nassopoulos
# Version: 1.0.0
# Description:
#   CLI menu to run the main project pipelines:
#   - download MeSH
#   - parse MeSH
#   - index MeSH (SQLite FTS / FAISS)
#   - extract candidates
#   - build extended MeSH
#   - run API
###############################################################################

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=============================================="
echo " MeSH Semantic Expansion - Pipeline Menu"
echo "=============================================="
echo "Project root: ${PROJECT_ROOT}"
echo ""

## ---------------------------------------------------------------------------
## Helpers
## ---------------------------------------------------------------------------

pause() {
  read -rp "Press ENTER to continue..."
}

run_python() {
  echo ""
  echo ">>> $*"
  $PYTHON_BIN "$@"
}

## ---------------------------------------------------------------------------
## Menu
## ---------------------------------------------------------------------------

while true; do
  echo ""
  echo "Select an action:"
  echo " 1) Download MeSH"
  echo " 2) Parse MeSH (XML -> JSONL)"
  echo " 3) Build SQLite FTS index"
  echo " 4) Build MeSH embeddings"
  echo " 5) Build FAISS index"
  echo " 6) Extract candidates from medical documents"
  echo " 7) Build extended MeSH (from validated CSV)"
  echo " 8) Run API (uvicorn)"
  echo " 0) Exit"
  echo ""

  read -rp "Your choice: " choice

  case "$choice" in
    1)
      read -rp "MeSH download URL: " MESH_URL
      run_python -m src.utils.utils_cli cmd_download_mesh "$MESH_URL"
      pause
      ;;
    2)
      run_python -m src.utils.utils_cli cmd_parse_mesh
      pause
      ;;
    3)
      run_python -m src.utils.utils_cli cmd_index_sqlite
      pause
      ;;
    4)
      run_python -m src.utils.utils_cli cmd_build_embeddings
      pause
      ;;
    5)
      run_python -m src.utils.utils_cli cmd_index_faiss
      pause
      ;;
    6)
      read -rp "Path to medical docs folder: " DOCS_DIR
      run_python -m src.utils.utils_cli cmd_extract_candidates "$DOCS_DIR"
      pause
      ;;
    7)
      run_python -m src.utils.utils_cli cmd_build_extended_mesh
      pause
      ;;
    8)
      echo ""
      echo "Starting API with uvicorn..."
      echo "CTRL+C to stop"
      echo ""
      uvicorn main:app --host 0.0.0.0 --port 8000 --reload
      ;;
    0)
      echo "Bye"
      exit 0
      ;;
    *)
      echo "Invalid choice."
      pause
      ;;
  esac
done