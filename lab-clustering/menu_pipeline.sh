#!/usr/bin/env bash

###############################################################################
# Lab-Clustering - Pipeline Menu
# Author: Georges Nassopoulos
# Version: 1.0.0
# Description:
#   CLI menu to run the main lab_clustering pipelines:
#   - parse TXT laboratory reports into structured CSV files
#   - build wide/long dataset from structured CSV files
#   - run unsupervised clustering + MLflow tracking + exports
#   - run EDA on structured data and datasets
#   - run full pipeline (parse-txt + build-dataset + cluster + eda)
#   - run FastAPI service
###############################################################################

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=============================================="
echo " Lab-Clustering - Pipeline Menu"
echo "=============================================="
echo "Project root: ${PROJECT_ROOT}"
echo ""

pause() {
  read -rp "Press ENTER to continue..."
}

run_python() {
  echo ""
  echo ">>> $*"
  $PYTHON_BIN "$@"
}

while true; do
  echo ""
  echo "Select an action:"
  echo " 1) Parse TXT -> structured CSV"
  echo " 2) Build dataset (wide/long)"
  echo " 3) Run clustering"
  echo " 4) Run EDA"
  echo " 5) Run full pipeline (parse-txt + build-dataset + cluster + eda)"
  echo " 6) Run API (uvicorn)"
  echo " 0) Exit"
  echo ""

  read -rp "Your choice: " choice

  case "$choice" in
    1)
      read -rp "TXT files (comma separated) [default: all in ./data/raw]: " TXT_FILES
      read -rp "Overwrite existing outputs? (y/n) [default: n]: " OVERWRITE

      OVERWRITE="${OVERWRITE:-n}"

      if [[ "$OVERWRITE" == "y" || "$OVERWRITE" == "Y" ]]; then
        run_python main.py --parse-txt --txt-files "$TXT_FILES" --overwrite
      else
        run_python main.py --parse-txt --txt-files "$TXT_FILES"
      fi

      pause
      ;;
    2)
      read -rp "Dataset format (wide/long) [default: wide]: " FORMAT
      read -rp "Overwrite existing dataset? (y/n) [default: n]: " OVERWRITE

      FORMAT="${FORMAT:-wide}"
      OVERWRITE="${OVERWRITE:-n}"

      if [[ "$OVERWRITE" == "y" || "$OVERWRITE" == "Y" ]]; then
        run_python main.py --build-dataset --dataset-format "$FORMAT" --overwrite
      else
        run_python main.py --build-dataset --dataset-format "$FORMAT"
      fi

      pause
      ;;
    3)
      read -rp "Dataset path [default: ./data/interim/datasets/dataset_wide.parquet]: " DATASET_PATH
      read -rp "Algorithm (kmeans/agglomerative/dbscan/birch) [default: kmeans]: " ALGO
      read -rp "Number of clusters [default: 3]: " NCLUST
      read -rp "Apply PCA? (y/n) [default: n]: " APPLY_PCA
      read -rp "PCA components [default: 2]: " PCA_COMP
      read -rp "Overwrite existing exports? (y/n) [default: n]: " OVERWRITE

      DATASET_PATH="${DATASET_PATH:-./data/interim/datasets/dataset_wide.parquet}"
      ALGO="${ALGO:-kmeans}"
      NCLUST="${NCLUST:-3}"
      PCA_COMP="${PCA_COMP:-2}"
      APPLY_PCA="${APPLY_PCA:-n}"
      OVERWRITE="${OVERWRITE:-n}"

      CMD="main.py --cluster --dataset-path \"$DATASET_PATH\" --algorithm \"$ALGO\" --n-clusters \"$NCLUST\" --pca-n-components \"$PCA_COMP\""

      if [[ "$APPLY_PCA" == "y" || "$APPLY_PCA" == "Y" ]]; then
        CMD="$CMD --apply-pca"
      fi

      if [[ "$OVERWRITE" == "y" || "$OVERWRITE" == "Y" ]]; then
        CMD="$CMD --overwrite"
      fi

      eval run_python $CMD

      pause
      ;;
    4)
      run_python main.py --eda
      pause
      ;;
    5)
      read -rp "Dataset format (wide/long) [default: wide]: " FORMAT
      read -rp "Algorithm (kmeans/agglomerative/dbscan/birch) [default: kmeans]: " ALGO
      read -rp "Number of clusters [default: 3]: " NCLUST

      FORMAT="${FORMAT:-wide}"
      ALGO="${ALGO:-kmeans}"
      NCLUST="${NCLUST:-3}"

      run_python main.py --run-all --dataset-format "$FORMAT" --algorithm "$ALGO" --n-clusters "$NCLUST"

      pause
      ;;
    6)
      read -rp "Host [default: 0.0.0.0]: " HOST
      read -rp "Port [default: 8000]: " PORT
      read -rp "Reload? (y/n) [default: n]: " RELOAD

      HOST="${HOST:-0.0.0.0}"
      PORT="${PORT:-8000}"
      RELOAD="${RELOAD:-n}"

      if [[ "$RELOAD" == "y" || "$RELOAD" == "Y" ]]; then
        run_python main.py --run-api --host "$HOST" --port "$PORT" --reload
      else
        run_python main.py --run-api --host "$HOST" --port "$PORT"
      fi

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