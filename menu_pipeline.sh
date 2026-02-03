#!/usr/bin/env bash

###############################################################################
# Doc-Classification - Pipeline Menu
# Author: Georges Nassopoulos
# Version: 1.0.0
# Description:
#   CLI menu to run the main doc-classification pipelines:
#   - build similarity index from labeled docs + manifest
#   - predict labels for unlabeled docs
#   - export predictions to CSV
#   - run EDA on a folder
#   - run full pipeline (build-index + predict + export)
###############################################################################

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=============================================="
echo " Doc-Classification - Pipeline Menu"
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
  echo " 1) Build similarity index (labeled + manifest)"
  echo " 2) Predict labels (unlabeled)"
  echo " 3) Export predictions to CSV (requires predict in same run)"
  echo " 4) Run EDA (choose folder)"
  echo " 5) Run full pipeline (build-index + predict + export)"
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

      ## Predict needs an index; main.py will rebuild index if not provided in this run
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

      ## Export requires predictions; we run predict + export in one command
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
