#!/bin/bash
##################################################
# Step 1: SLAM exploration + SAM/CLIP extraction (CorrCLIP-style postproc **off**)
#
# Results (auto run_N if run_* already exists):
#   Passive     → results/Replica/{SCENE}/Passive/run_N/
#   ActiveGeom  → results/Replica/{SCENE}/ActiveGeom/run_N/
#   ActiveOpenSem       → results/Replica/{SCENE}/ActiveOpenSem/run_N/
#
# Usage:
#   bash scripts/aov-gs/01_slam_exploration.sh [SCENE] [EXP] [SEED] [ENABLE_VIS] [DEBUG] [RESULT_RUN] [MASK_COLLECTOR]
#
#   RESULT_RUN:      run_0, run_1, … (optional; else auto run_N)
#   MASK_COLLECTOR:  sam (default) | corrclip
#
#   EXP: ActiveOpenSemGeom | ActiveOpenSemPassive | ActiveOpenSem | ActiveOpenSem_base | ...
#   ENABLE_VIS: 0 = headless, 1 = live OpenCV windows
#   DEBUG:      1 = save keyframes/ as JPEG
#
# Examples:
#   bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSemGeom
#   bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem
##################################################

set -e

SCENE=${1:-office0}
EXP=${2:-ActiveOpenSemGeom}
SEED=${3:-0}
ENABLE_VIS=${4:-0}
DEBUG=${5:-0}
RESULT_RUN=${6:-}
MASK_COLLECTOR=${7:-sam}

export CUDA_VISIBLE_DEVICES=${GPU:-0,1}
PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

case "${MASK_COLLECTOR,,}" in
  corrclip) CORRCLIP=1 ;;
  *)        CORRCLIP=0 ;;
esac

RESULT_FLAG=""
RUN_NOTE="auto (run_N under experiment folder)"
if [ -n "$RESULT_RUN" ]; then
  RESULT_DIR="${PROJ_DIR}/results/Replica/${SCENE}/${EXP}/${RESULT_RUN}"
  mkdir -p "$RESULT_DIR"
  RESULT_FLAG="--result_dir ${RESULT_DIR}"
  RUN_NOTE="${RESULT_DIR}"
fi

echo "=============================================="
echo "  Scene      : $SCENE"
echo "  Config     : configs/Replica/${SCENE}/${EXP}.py"
echo "  Seed       : $SEED"
echo "  Visualize  : $ENABLE_VIS"
echo "  Debug      : $DEBUG"
echo "  CorrCLIP   : $([ "$CORRCLIP" = 1 ] && echo ON || echo OFF) (--corrclip ${CORRCLIP})"
echo "  Run dir    : $RUN_NOTE"
echo "  CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
echo "=============================================="

DEBUG_FLAG=""
[ "$DEBUG" = "1" ] && DEBUG_FLAG="--debug"

python src/main/activesgm.py \
    --cfg        "configs/Replica/${SCENE}/${EXP}.py" \
    --seed       "$SEED" \
    --enable_vis "$ENABLE_VIS" \
    --corrclip   "$CORRCLIP" \
    $RESULT_FLAG \
    $DEBUG_FLAG

echo ""
echo "=== SLAM finished ==="
echo "    Check latest run_* under results/Replica/${SCENE}/"
