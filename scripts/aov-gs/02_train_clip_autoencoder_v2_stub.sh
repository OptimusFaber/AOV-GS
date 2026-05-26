#!/bin/bash
##################################################
# Step 2: LangSplatV2 pipeline note
#
# В V2 отдельный AE-этап больше не нужен:
# codebook + sparse coefficients обучаются внутри step 3.
# Этот скрипт сохранён как совместимый no-op с валидацией входных данных.
##################################################

set -euo pipefail

RESULT_DIR=${1:?"Usage: $0 RESULT_DIR"}

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

if [ ! -d "${RESULT_DIR}/language_features" ]; then
    echo "ERROR: language_features не найдены в ${RESULT_DIR}"
    echo "       Сначала выполните 01_slam_exploration.sh"
    exit 1
fi

echo "=============================================="
echo "  Result dir   : $RESULT_DIR"
echo "  Features dir : ${RESULT_DIR}/language_features"
echo "=============================================="
echo "LangSplatV2: step 2 skipped (AE removed)."
echo "Proceed to step 3 for codebook+quantization training."

echo "=== Step 2 complete (no-op) ==="
