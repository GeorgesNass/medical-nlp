#!/usr/bin/env bash

###############################################################################
# Doc-Classification - Pipeline Menu
# Author: Georges Nassopoulos
# Version: 1.1.0
# Description:
#   CLI menu to run the main doc-classification pipelines (with data consistency + data quality):
#   - build similarity index from labeled docs + manifest
#   - predict labels for unlabeled docs
#   - export predictions to CSV
#   - run EDA on a folder
#   - run full pipeline (build-index + predict + export)
#   - run data drift (Evidently)
###############################################################################

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=============================================="
echo " Doc-Classification - Pipeline Menu (data consistency + data quality)"
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
  echo " 1) Build similarity index (labeled + manifest) (data consistency + data quality)"
  echo " 2) Predict labels (unlabeled) (data consistency + data quality)"
  echo " 3) Export predictions to CSV (requires predict in same run) (data consistency + data quality)"
  echo " 4) Run EDA (choose folder) (data consistency + data quality)"
  echo " 5) Run full pipeline (build-index + predict + export) (data consistency + data quality)"
  echo " 6) Run data quality check only"
  echo " 7) Run data drift"
  echo " 0) Exit"
  echo ""

  read -rp "Your choice: " choice

  case "$choice" in
    1)
      read -rp "Labeled folder [default: ./data/labeled]: " LABELED_DIR
      read -rp "Manifest JSON [default: ./data/labeled_manifest.json]: " MANIFEST

      LABELED_DIR="${LABELED_DIR:-./data/labeled}"
      MANIFEST="${MANIFEST:-./data/labeled_manifest.json}"

      run_python main.py --build-index --labeled-dir "$LABELED_DIR" --manifest "$MANIFEST"
      pause
      ;;
    2)
      read -rp "Labeled folder [default: ./data/labeled]: " LABELED_DIR
      read -rp "Manifest JSON [default: ./data/labeled_manifest.json]: " MANIFEST
      read -rp "Unlabeled folder [default: ./data/unlabeled]: " UNLABELED_DIR

      LABELED_DIR="${LABELED_DIR:-./data/labeled}"
      MANIFEST="${MANIFEST:-./data/labeled_manifest.json}"
      UNLABELED_DIR="${UNLABELED_DIR:-./data/unlabeled}"

      run_python main.py --predict --labeled-dir "$LABELED_DIR" --manifest "$MANIFEST" --unlabeled-dir "$UNLABELED_DIR"
      pause
      ;;
    3)
      read -rp "Output CSV name [default: predictions.csv]: " OUTCSV
      read -rp "Include scores? (y/n) [default: y]: " INCSCORES
      read -rp "Include evidence? (y/n) [default: y]: " INCEVID
      read -rp "Labeled folder [default: ./data/labeled]: " LABELED_DIR
      read -rp "Manifest JSON [default: ./data/labeled_manifest.json]: " MANIFEST
      read -rp "Unlabeled folder [default: ./data/unlabeled]: " UNLABELED_DIR

      OUTCSV="${OUTCSV:-predictions.csv}"
      INCSCORES="${INCSCORES:-y}"
      INCEVID="${INCEVID:-y}"
      LABELED_DIR="${LABELED_DIR:-./data/labeled}"
      MANIFEST="${MANIFEST:-./data/labeled_manifest.json}"
      UNLABELED_DIR="${UNLABELED_DIR:-./data/unlabeled}"

      CMD_ARGS=(main.py --predict --export --output-csv "$OUTCSV" --labeled-dir "$LABELED_DIR" --manifest "$MANIFEST" --unlabeled-dir "$UNLABELED_DIR")

      if [[ "$INCSCORES" == "y" || "$INCSCORES" == "Y" ]]; then
        CMD_ARGS+=("--include-scores")
      fi
      if [[ "$INCEVID" == "y" || "$INCEVID" == "Y" ]]; then
        CMD_ARGS+=("--include-evidence")
      fi

      run_python "${CMD_ARGS[@]}"
      pause
      ;;
    4)
      read -rp "EDA folder [default: ./data/labeled]: " EDA_DIR
      read -rp "EDA output JSON [default: eda_summary.json]: " EDA_OUT

      EDA_DIR="${EDA_DIR:-./data/labeled}"
      EDA_OUT="${EDA_OUT:-eda_summary.json}"

      run_python main.py --eda --eda-folder "$EDA_DIR" --eda-output "$EDA_OUT"
      pause
      ;;
    5)
      read -rp "Labeled folder [default: ./data/labeled]: " LABELED_DIR
      read -rp "Manifest JSON [default: ./data/labeled_manifest.json]: " MANIFEST
      read -rp "Unlabeled folder [default: ./data/unlabeled]: " UNLABELED_DIR
      read -rp "Output CSV name [default: predictions.csv]: " OUTCSV
      read -rp "Include scores? (y/n) [default: y]: " INCSCORES
      read -rp "Include evidence? (y/n) [default: y]: " INCEVID

      LABELED_DIR="${LABELED_DIR:-./data/labeled}"
      MANIFEST="${MANIFEST:-./data/labeled_manifest.json}"
      UNLABELED_DIR="${UNLABELED_DIR:-./data/unlabeled}"
      OUTCSV="${OUTCSV:-predictions.csv}"
      INCSCORES="${INCSCORES:-y}"
      INCEVID="${INCEVID:-y}"

      CMD_ARGS=(main.py --run-all --labeled-dir "$LABELED_DIR" --manifest "$MANIFEST" --unlabeled-dir "$UNLABELED_DIR" --output-csv "$OUTCSV")

      if [[ "$INCSCORES" == "y" || "$INCSCORES" == "Y" ]]; then
        CMD_ARGS+=("--include-scores")
      fi
      if [[ "$INCEVID" == "y" || "$INCEVID" == "Y" ]]; then
        CMD_ARGS+=("--include-evidence")
      fi

      run_python "${CMD_ARGS[@]}"
      pause
      ;;
    6)
      echo "Running data quality check..."

      run_python main.py --validate-config

      pause
      ;;
    7)
      ## DATA DRIFT (DOC CLASSIFICATION + EVIDENTLY)
      read -rp "Reference dataset [default: ./data/processed/reference.csv]: " REF
      read -rp "Current dataset [default: ./data/processed/current.csv]: " CUR

      REF="${REF:-./data/processed/reference.csv}"
      CUR="${CUR:-./data/processed/current.csv}"

      run_python main.py \
        --mode drift \
        --ref "$REF" \
        --current "$CUR"

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