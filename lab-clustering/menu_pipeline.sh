#!/usr/bin/env bash

###############################################################################
# Lab-Clustering - Pipeline Menu
# Author: Georges Nassopoulos
# Version: 1.1.0
# Description:
#   CLI menu to run the main lab_clustering pipelines:
#   - parse TXT laboratory reports into structured CSV files (with data consistency + data quality)
#   - build wide/long dataset from structured CSV files (with data consistency + data quality)
#   - run unsupervised clustering + MLflow tracking + exports (with data consistency + data quality)
#   - run EDA on structured data and datasets (with data consistency + data quality)
#   - run full pipeline (parse-txt + build-dataset + cluster + eda) (with data consistency + data quality)
#   - run FastAPI service (with data consistency + data quality)
#   - run data drift (Evidently)
###############################################################################

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=============================================="
echo " Lab-Clustering - Pipeline Menu (data consistency + data quality)"
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
  echo " 1) Parse TXT -> structured CSV (with data consistency + data quality)"
  echo " 2) Build dataset (wide/long) (with data consistency + data quality)"
  echo " 3) Run clustering (with data consistency + data quality)"
  echo " 4) Run EDA (with data consistency + data quality)"
  echo " 5) Run full pipeline (parse-txt + build-dataset + cluster + eda) (with data consistency + data quality)"
  echo " 6) Run API (uvicorn) (with data consistency + data quality)"
  echo " 7) Run data quality check only"
  echo " 8) Run data drift"
  echo " 0) Exit"
  echo ""

  read -rp "Your choice: " choice

  case "$choice" in
    1)
      read -rp "TXT files (comma separated) [default: all in ./data/raw]: " TXT_FILES
      read -rp "Overwrite existing outputs? (y/n) [default: n]: " OVERWRITE

      ## FE NEW
      read -rp "Enable feature engineering? (y/n) [default: n]: " FE
      FE="${FE:-n}"

      OVERWRITE="${OVERWRITE:-n}"

      if [[ "$OVERWRITE" == "y" || "$OVERWRITE" == "Y" ]]; then
        if [[ "$FE" == "y" || "$FE" == "Y" ]]; then
          run_python main.py --parse-txt --txt-files "$TXT_FILES" --overwrite --features
        else
          run_python main.py --parse-txt --txt-files "$TXT_FILES" --overwrite
        fi
      else
        if [[ "$FE" == "y" || "$FE" == "Y" ]]; then
          run_python main.py --parse-txt --txt-files "$TXT_FILES" --features
        else
          run_python main.py --parse-txt --txt-files "$TXT_FILES"
        fi
      fi

      pause
      ;;
    2)
      read -rp "Dataset format (wide/long) [default: wide]: " FORMAT
      read -rp "Overwrite existing dataset? (y/n) [default: n]: " OVERWRITE

      ## FE NEW
      read -rp "Enable feature engineering? (y/n) [default: n]: " FE
      FE="${FE:-n}"

      FORMAT="${FORMAT:-wide}"
      OVERWRITE="${OVERWRITE:-n}"

      if [[ "$OVERWRITE" == "y" || "$OVERWRITE" == "Y" ]]; then
        if [[ "$FE" == "y" || "$FE" == "Y" ]]; then
          run_python main.py --build-dataset --dataset-format "$FORMAT" --overwrite --features
        else
          run_python main.py --build-dataset --dataset-format "$FORMAT" --overwrite
        fi
      else
        if [[ "$FE" == "y" || "$FE" == "Y" ]]; then
          run_python main.py --build-dataset --dataset-format "$FORMAT" --features
        else
          run_python main.py --build-dataset --dataset-format "$FORMAT"
        fi
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

      ## FE NEW
      read -rp "Enable feature engineering? (y/n) [default: n]: " FE
      FE="${FE:-n}"

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

      if [[ "$FE" == "y" || "$FE" == "Y" ]]; then
        CMD="$CMD --features"
      fi

      eval run_python $CMD

      pause
      ;;
    4)
      read -rp "Enable feature engineering? (y/n) [default: n]: " FE
      FE="${FE:-n}"

      if [[ "$FE" == "y" || "$FE" == "Y" ]]; then
        run_python main.py --eda --features
      else
        run_python main.py --eda
      fi

      pause
      ;;
    5)
      read -rp "Dataset format (wide/long) [default: wide]: " FORMAT
      read -rp "Algorithm (kmeans/agglomerative/dbscan/birch) [default: kmeans]: " ALGO
      read -rp "Number of clusters [default: 3]: " NCLUST

      ## FE NEW
      read -rp "Enable feature engineering? (y/n) [default: n]: " FE
      FE="${FE:-n}"

      FORMAT="${FORMAT:-wide}"
      ALGO="${ALGO:-kmeans}"
      NCLUST="${NCLUST:-3}"

      if [[ "$FE" == "y" || "$FE" == "Y" ]]; then
        run_python main.py --run-all --dataset-format "$FORMAT" --algorithm "$ALGO" --n-clusters "$NCLUST" --features
      else
        run_python main.py --run-all --dataset-format "$FORMAT" --algorithm "$ALGO" --n-clusters "$NCLUST"
      fi

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
    7)
      echo ""
      echo "Running data quality check..."
      echo ""

      run_python main.py --validate-config

      pause
      ;;
    8)
      read -rp "Reference dataset CSV: " REF
      read -rp "Current dataset CSV: " CUR

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