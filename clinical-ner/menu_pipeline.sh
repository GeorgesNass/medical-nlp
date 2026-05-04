#!/usr/bin/env bash

###############################################################################
# Clinical NER - Pipeline Menu
# Author: Georges Nassopoulos
# Version: 1.2.0
# Description:
#   CLI menu to run the main Clinical NER pipelines:
#   - labeled mode (CSV with entities)
#   - unlabeled mode (folder of .txt + dictionaries)
#   - run unit tests
#   - run quick smoke checks
#   - run data drift (Evidently)
#   - all modes include data consistency + data quality checks
#   - feature engineering toggle available
###############################################################################

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

## Ensure project root is on PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "=============================================="
echo " Clinical NER - Pipeline Menu (data consistency + data quality)"
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
  echo " 4) Quick smoke test"
  echo " 5) Run data quality check only"
  echo " 6) Run data drift"
  echo " 0) Exit"
  echo ""

  read -rp "Your choice: " choice

  case "${choice}" in

    1)
      read -rp "Path to labeled CSV: " LABELED_CSV
      ensure_file_exists "${LABELED_CSV}" || { pause; continue; }

      read -rp "Enable feature engineering? (y/n): " FE

      read -rp "Output CSV path (default: artifacts/exports/clinical_ner_records.csv): " OUT_CSV
      OUT_CSV="${OUT_CSV:-${PROJECT_ROOT}/artifacts/exports/clinical_ner_records.csv}"

      if [[ "$FE" == "y" ]]; then
        run_python "${PROJECT_ROOT}/main.py" \
          --labeled-csv "${LABELED_CSV}" \
          --output-csv "${OUT_CSV}" \
          --project-root "${PROJECT_ROOT}" \
          --features
      else
        run_python "${PROJECT_ROOT}/main.py" \
          --labeled-csv "${LABELED_CSV}" \
          --output-csv "${OUT_CSV}" \
          --project-root "${PROJECT_ROOT}"
      fi

      pause
      ;;

    2)
      read -rp "Path to folder containing .txt docs: " DOCS_DIR
      ensure_dir_exists "${DOCS_DIR}" || { pause; continue; }

      read -rp "Enable feature engineering? (y/n): " FE

      read -rp "Output CSV path (default: artifacts/exports/clinical_ner_records.csv): " OUT_CSV
      OUT_CSV="${OUT_CSV:-${PROJECT_ROOT}/artifacts/exports/clinical_ner_records.csv}"

      if [[ "$FE" == "y" ]]; then
        run_python "${PROJECT_ROOT}/main.py" \
          --unlabeled-texts "${DOCS_DIR}" \
          --output-csv "${OUT_CSV}" \
          --project-root "${PROJECT_ROOT}" \
          --features
      else
        run_python "${PROJECT_ROOT}/main.py" \
          --unlabeled-texts "${DOCS_DIR}" \
          --output-csv "${OUT_CSV}" \
          --project-root "${PROJECT_ROOT}"
      fi

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
      echo "Creating smoke dataset..."
      SMOKE_DIR="${PROJECT_ROOT}/data/raw/smoke_docs"
      mkdir -p "${SMOKE_DIR}"

      echo "Patient denies asthma. Chronic diabetes. Aspirin current." > "${SMOKE_DIR}/doc1.txt"

      read -rp "Enable feature engineering? (y/n): " FE

      OUT_CSV="${PROJECT_ROOT}/artifacts/exports/smoke_out.csv"

      if [[ "$FE" == "y" ]]; then
        run_python "${PROJECT_ROOT}/main.py" \
          --unlabeled-texts "${SMOKE_DIR}" \
          --output-csv "${OUT_CSV}" \
          --project-root "${PROJECT_ROOT}" \
          --features
      else
        run_python "${PROJECT_ROOT}/main.py" \
          --unlabeled-texts "${SMOKE_DIR}" \
          --output-csv "${OUT_CSV}" \
          --project-root "${PROJECT_ROOT}"
      fi

      echo "Smoke output: ${OUT_CSV}"
      pause
      ;;

    5)
      echo ""
      echo "Running data quality check..."
      run_python "${PROJECT_ROOT}/main.py" --validate-config
      pause
      ;;

    6)
      read -rp "Reference dataset CSV: " REF
      ensure_file_exists "${REF}" || { pause; continue; }

      read -rp "Current dataset CSV: " CUR
      ensure_file_exists "${CUR}" || { pause; continue; }

      run_python "${PROJECT_ROOT}/main.py" \
        --mode drift \
        --ref "${REF}" \
        --current "${CUR}" \
        --project-root "${PROJECT_ROOT}"

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