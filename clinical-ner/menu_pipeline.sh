#!/usr/bin/env bash

###############################################################################
# Clinical NER - Pipeline Menu
# Author: Georges Nassopoulos
# Version: 1.0.0
# Description:
#   CLI menu to run the main Clinical NER pipelines:
#   - labeled mode (CSV with entities)
#   - unlabeled mode (folder of .txt + dictionaries)
#   - run unit tests
#   - run quick smoke checks
###############################################################################

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

## Ensure project root is on PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "=============================================="
echo " Clinical NER - Pipeline Menu"
echo "=============================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Python bin  : ${PYTHON_BIN}"
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
  "${PYTHON_BIN}" "$@"
}

ensure_file_exists() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: File not found: $path"
    return 1
  fi
}

ensure_dir_exists() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: Directory not found: $path"
    return 1
  fi
}

## ---------------------------------------------------------------------------
## Menu
## ---------------------------------------------------------------------------

while true; do
  echo ""
  echo "Select an action:"
  echo " 1) Run pipeline (LABELED CSV -> export CSV)"
  echo " 2) Run pipeline (UNLABELED TXT -> export CSV)"
  echo " 3) Run unit tests (pytest)"
  echo " 4) Quick smoke: create sample TXT and run unlabeled pipeline"
  echo " 0) Exit"
  echo ""

  read -rp "Your choice: " choice

  case "${choice}" in
    1)
      read -rp "Path to labeled CSV: " LABELED_CSV
      ensure_file_exists "${LABELED_CSV}" || { pause; continue; }

      read -rp "Output CSV path (default: artifacts/exports/clinical_ner_records.csv): " OUT_CSV
      OUT_CSV="${OUT_CSV:-${PROJECT_ROOT}/artifacts/exports/clinical_ner_records.csv}"

      run_python "${PROJECT_ROOT}/main.py" \
        --labeled-csv "${LABELED_CSV}" \
        --output-csv "${OUT_CSV}" \
        --project-root "${PROJECT_ROOT}"

      pause
      ;;
    2)
      read -rp "Path to folder containing .txt docs: " DOCS_DIR
      ensure_dir_exists "${DOCS_DIR}" || { pause; continue; }

      read -rp "Output CSV path (default: artifacts/exports/clinical_ner_records.csv): " OUT_CSV
      OUT_CSV="${OUT_CSV:-${PROJECT_ROOT}/artifacts/exports/clinical_ner_records.csv}"

      run_python "${PROJECT_ROOT}/main.py" \
        --unlabeled-texts "${DOCS_DIR}" \
        --output-csv "${OUT_CSV}" \
        --project-root "${PROJECT_ROOT}"

      pause
      ;;
    3)
      echo ""
      echo "Running pytest..."
      echo ""
      (cd "${PROJECT_ROOT}" && "${PYTHON_BIN}" -m pytest -q)
      pause
      ;;
    4)
      echo ""
      echo "Creating a tiny smoke dataset in data/raw/smoke_docs..."
      echo ""
      SMOKE_DIR="${PROJECT_ROOT}/data/raw/smoke_docs"
      mkdir -p "${SMOKE_DIR}"

      echo "Patient denies asthma. Chronic diabetes. Aspirin current." > "${SMOKE_DIR}/doc1.txt"

      OUT_CSV="${PROJECT_ROOT}/artifacts/exports/smoke_out.csv"

      run_python "${PROJECT_ROOT}/main.py" \
        --unlabeled-texts "${SMOKE_DIR}" \
        --output-csv "${OUT_CSV}" \
        --project-root "${PROJECT_ROOT}"

      echo ""
      echo "Smoke output: ${OUT_CSV}"
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
