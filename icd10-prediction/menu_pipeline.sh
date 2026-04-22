#!/usr/bin/env bash

###############################################################################
# ICD10-Prediction - Pipeline Menu
# Author: Georges Nassopoulos
# Version: 1.1.0
# Description:
#   CLI menu to run the main icd10_prediction pipelines:
#   - parse RSS files into a consolidated structured CSV (with data consistency + data quality)
#   - build one CSV per admission_id from clinical_records + RSS metadata (with data consistency + data quality)
#   - train baseline model and export metrics (with data consistency + data quality)
#   - run EDA (label distribution + text length stats + plot) (with data consistency + data quality)
#   - run full pipeline (parse-rss + build-clinical-csv + train + eda) (with data consistency + data quality)
#   - run FastAPI service (with data consistency + data quality)
#   - run data drift (Evidently)
###############################################################################

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=============================================="
echo " ICD10-Prediction - Pipeline Menu"
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
  echo " 1) Parse RSS -> consolidated CSV (with data consistency + data quality)"
  echo " 2) Build clinical_records CSVs (1 per admission_id) (with data consistency + data quality)"
  echo " 3) Train baseline model + export metrics (with data consistency + data quality)"
  echo " 4) Run EDA (label distribution + text stats + plot) (with data consistency + data quality)"
  echo " 5) Run full pipeline (parse-rss + build-clinical-csv + train + eda) (with data consistency + data quality)"
  echo " 6) Run API (uvicorn) (with data consistency + data quality)"
  echo " 7) Run data quality check only"
  echo " 8) Run data drift (NEW)"
  echo " 0) Exit"
  echo ""

  read -rp "Your choice: " choice

  case "$choice" in
    1)
      read -rp "RSS folder [default: ./data/raw/icd10]: " RSS_DIR
      read -rp "Output RSS CSV [default: ./data/interim/icd10_csv/icd10_structured.csv]: " RSS_OUT

      RSS_DIR="${RSS_DIR:-./data/raw/icd10}"
      RSS_OUT="${RSS_OUT:-./data/interim/icd10_csv/icd10_structured.csv}"

      run_python main.py --parse-rss --rss-dir "$RSS_DIR" --rss-output-csv "$RSS_OUT"
      pause
      ;;
    2)
      read -rp "Clinical records folder [default: ./data/raw/clinical_records]: " CLINICAL_DIR
      read -rp "RSS consolidated CSV [default: ./data/interim/icd10_csv/icd10_structured.csv]: " RSS_OUT
      read -rp "Output clinical CSV folder [default: ./data/interim/clinical_records_csv]: " CLINICAL_OUT

      CLINICAL_DIR="${CLINICAL_DIR:-./data/raw/clinical_records}"
      RSS_OUT="${RSS_OUT:-./data/interim/icd10_csv/icd10_structured.csv}"
      CLINICAL_OUT="${CLINICAL_OUT:-./data/interim/clinical_records_csv}"

      run_python main.py --build-clinical-csv --clinical-records-dir "$CLINICAL_DIR" --rss-output-csv "$RSS_OUT" --clinical-csv-dir "$CLINICAL_OUT"
      pause
      ;;
    3)
      read -rp "Clinical CSV folder [default: ./data/interim/clinical_records_csv]: " CLINICAL_CSV_DIR
      read -rp "Model output [default: ./artifacts/models/model.joblib]: " MODEL_OUT
      read -rp "Metrics output [default: ./artifacts/reports/metrics.json]: " METRICS_OUT

      CLINICAL_CSV_DIR="${CLINICAL_CSV_DIR:-./data/interim/clinical_records_csv}"
      MODEL_OUT="${MODEL_OUT:-./artifacts/models/model.joblib}"
      METRICS_OUT="${METRICS_OUT:-./artifacts/reports/metrics.json}"

      run_python main.py --train --clinical-csv-dir "$CLINICAL_CSV_DIR" --model-output "$MODEL_OUT" --metrics-output "$METRICS_OUT"
      pause
      ;;
    4)
      read -rp "Clinical CSV folder [default: ./data/interim/clinical_records_csv]: " CLINICAL_CSV_DIR
      read -rp "EDA plot output [default: ./artifacts/exports/eda/label_distribution.png]: " EDA_PLOT_OUT

      CLINICAL_CSV_DIR="${CLINICAL_CSV_DIR:-./data/interim/clinical_records_csv}"
      EDA_PLOT_OUT="${EDA_PLOT_OUT:-./artifacts/exports/eda/label_distribution.png}"

      run_python main.py --eda --clinical-csv-dir "$CLINICAL_CSV_DIR" --eda-plot-output "$EDA_PLOT_OUT"
      pause
      ;;
    5)
      read -rp "RSS folder [default: ./data/raw/icd10]: " RSS_DIR
      read -rp "Clinical records folder [default: ./data/raw/clinical_records]: " CLINICAL_DIR
      read -rp "Output clinical CSV folder [default: ./data/interim/clinical_records_csv]: " CLINICAL_OUT
      read -rp "Output RSS CSV [default: ./data/interim/icd10_csv/icd10_structured.csv]: " RSS_OUT
      read -rp "Model output [default: ./artifacts/models/model.joblib]: " MODEL_OUT
      read -rp "Metrics output [default: ./artifacts/reports/metrics.json]: " METRICS_OUT
      read -rp "EDA plot output [default: ./artifacts/exports/eda/label_distribution.png]: " EDA_PLOT_OUT

      RSS_DIR="${RSS_DIR:-./data/raw/icd10}"
      CLINICAL_DIR="${CLINICAL_DIR:-./data/raw/clinical_records}"
      CLINICAL_OUT="${CLINICAL_OUT:-./data/interim/clinical_records_csv}"
      RSS_OUT="${RSS_OUT:-./data/interim/icd10_csv/icd10_structured.csv}"
      MODEL_OUT="${MODEL_OUT:-./artifacts/models/model.joblib}"
      METRICS_OUT="${METRICS_OUT:-./artifacts/reports/metrics.json}"
      EDA_PLOT_OUT="${EDA_PLOT_OUT:-./artifacts/exports/eda/label_distribution.png}"

      run_python main.py --run-all --rss-dir "$RSS_DIR" --clinical-records-dir "$CLINICAL_DIR" --clinical-csv-dir "$CLINICAL_OUT" --rss-output-csv "$RSS_OUT" --model-output "$MODEL_OUT" --metrics-output "$METRICS_OUT" --eda-plot-output "$EDA_PLOT_OUT"
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
      echo "Running standalone data quality check..."
      run_python main.py --validate-config
      pause
      ;;
    8)
      ## DATA DRIFT (ICD10 + EVIDENTLY)
      read -rp "Reference dataset [default: ./data/processed/train.csv]: " REF
      read -rp "Current dataset [default: ./data/processed/new.csv]: " CUR

      REF="${REF:-./data/processed/train.csv}"
      CUR="${CUR:-./data/processed/new.csv}"

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