#!/usr/bin/env bash

###############################################################################
# MeSH Semantic Expansion - Pipeline Menu
# Author: Georges Nassopoulos
# Version: 1.1.0
# Description:
#   CLI menu to run the main project pipelines:
#   - download MeSH (with data consistency)
#   - parse MeSH (XML -> JSONL) (with data consistency)
#   - index MeSH (SQLite FTS / FAISS) (with data consistency)
#   - extract candidates (with data consistency)
#   - build extended MeSH (with data consistency)
#   - run API (with data consistency)
#   - run data drift (Evidently)
###############################################################################

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

## ============================================================
## FEATURE STORE NEW
## ============================================================
: "${FEATURE_STORE_MODE:=redis}"

echo "=============================================="
echo " MeSH Semantic Expansion - Pipeline Menu"
echo "=============================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Feature Store Mode: ${FEATURE_STORE_MODE}"
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
  echo " 1) Download MeSH (with data consistency)"
  echo " 2) Parse MeSH (XML -> JSONL) (with data consistency)"
  echo " 3) Build SQLite FTS index (with data consistency)"
  echo " 4) Build MeSH embeddings (with data consistency)"
  echo " 5) Build FAISS index (with data consistency)"
  echo " 6) Extract candidates from medical documents (with data consistency)"
  echo " 7) Build extended MeSH (from validated CSV) (with data consistency)"
  echo " 8) Run API (uvicorn) (with data consistency)"
  echo " 9) Run data drift"
  echo " 0) Exit"
  echo ""

  read -rp "Your choice: " choice

  ## ============================================================
  ## FEATURE STORE PROMPT
  ## ============================================================
  read -rp "Feature store mode (redis/feast) [default: ${FEATURE_STORE_MODE}]: " FSM
  FSM="${FSM:-$FEATURE_STORE_MODE}"
  export FEATURE_STORE_MODE="$FSM"

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
      read -rp "Enable feature engineering? (y/n): " FE

      if [[ "$FE" == "y" ]]; then
        run_python -m src.utils.utils_cli cmd_extract_candidates "$DOCS_DIR" --features --feature-store-mode "$FSM"
      else
        run_python -m src.utils.utils_cli cmd_extract_candidates "$DOCS_DIR" --feature-store-mode "$FSM"
      fi

      pause
      ;;
    7)
      run_python -m src.utils.utils_cli cmd_build_extended_mesh
      pause
      ;;
    8)
      read -rp "Enable feature engineering? (y/n): " FE

      echo ""
      echo "Starting API with uvicorn..."
      echo "CTRL+C to stop"
      echo ""

      if [[ "$FE" == "y" ]]; then
        uvicorn main:app --host 0.0.0.0 --port 8000 --reload --features --feature-store-mode "$FSM"
      else
        uvicorn main:app --host 0.0.0.0 --port 8000 --reload --feature-store-mode "$FSM"
      fi
      ;;
    9)
      ## DATA DRIFT (MESH + EVIDENTLY)
      read -rp "Reference dataset CSV: " REF
      read -rp "Current dataset CSV: " CUR

      run_python main.py \
        --mode drift \
        --ref "$REF" \
        --current "$CUR" \
        --feature-store-mode "$FSM"

      pause
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