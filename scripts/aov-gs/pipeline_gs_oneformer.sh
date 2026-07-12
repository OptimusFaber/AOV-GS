#!/usr/bin/env bash
##################################################
# Pipeline 2 — Gaussian Splatting with OneFormer
#
# Config: ActiveSem (semsplatam + OneFormer + semantic planner active_gsv2).
# OneFormer / HuggingFace weights required (see configs/Replica/<scene>/ActiveSem.py).
# Result: semantic map; SAM+CLIP language_features are not written here.
#
# Usage:
#   bash scripts/aov-gs/pipeline_gs_oneformer.sh [SCENE] [SEED] [ENABLE_VIS] [DEBUG]
#
# Example:
#   bash scripts/aov-gs/pipeline_gs_oneformer.sh office0 0 0 0
##################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE="${1:-office0}"
SEED="${2:-0}"
ENABLE_VIS="${3:-0}"
DEBUG="${4:-0}"
EXP="ActiveSem"
RESULT_DIR="${PROJ_DIR}/results/Replica/${SCENE}/${EXP}/run_0"

if [ ! -f "${PROJ_DIR}/configs/Replica/${SCENE}/${EXP}.py" ]; then
  echo "ERROR: missing config ${PROJ_DIR}/configs/Replica/${SCENE}/${EXP}.py"
  echo "       Copy/adapt ActiveSem.py for this scene (see other office*/ActiveSem.py)."
  exit 1
fi

echo "=============================================="
echo "  Pipeline 2: GS + OneFormer (ActiveSem)"
echo "  Scene       : ${SCENE}"
echo "  Config      : configs/Replica/${SCENE}/${EXP}.py"
echo "  Result dir  : ${RESULT_DIR}"
echo "=============================================="

bash "${SCRIPT_DIR}/01_slam_exploration.sh" "${SCENE}" "${EXP}" "${SEED}" "${ENABLE_VIS}" "${DEBUG}"

echo ""
echo "=== Pipeline 2 finished ==="
echo "    Gaussians: ${RESULT_DIR}/splatam/final/params0.npz (or params.npz)"
echo "    For open-vocab language field use pipeline 3 or 4 (ActiveOpenSem)."
