#!/bin/bash
##################################################
# Step 2: LangSplatV2 pipeline note
#
# In V2 a separate AE stage is no longer needed:
# codebook + sparse coefficients are trained inside step 3.
# This script is kept as a compatible no-op with input validation.
##################################################

set -euo pipefail

RESULT_DIR=${1:?"Usage: $0 RESULT_DIR"}

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

if [ ! -d "${RESULT_DIR}/language_features" ]; then
    echo "ERROR: language_features not found in ${RESULT_DIR}"
    echo "       Run 01_slam_exploration.sh first"
    exit 1
fi

echo "=============================================="
echo "  Result dir   : $RESULT_DIR"
echo "  Features dir : ${RESULT_DIR}/language_features"
echo "=============================================="
echo "LangSplatV2: step 2 skipped (AE removed)."
echo "Proceed to step 3 for codebook+quantization training."

echo "=== Step 2 complete (no-op) ==="
