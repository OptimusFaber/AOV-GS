#!/bin/bash
##################################################
# Step 1: SLAM exploration + SAM/CLIP extraction (CorrCLIP-style postproc **off**)
#
# Результаты (auto run_N если run_* уже есть):
#   Passive     → results/Replica/{SCENE}/Passive/run_N/
#   ActiveGeom  → results/Replica/{SCENE}/ActiveGeom/run_N/
#   ActiveOpenSem       → results/Replica/{SCENE}/ActiveOpenSem/run_N/
#
# Использование:
#   bash scripts/aov-gs/01_slam_exploration.sh [SCENE] [EXP] [SEED] [ENABLE_VIS] [DEBUG]
#
#   EXP: ActiveOpenSemGeom | ActiveOpenSemPassive | ActiveOpenSem | ActiveOpenSem_base | ...
#   ENABLE_VIS: 0 = headless, 1 = live OpenCV windows
#   DEBUG:      1 = save keyframes/ as JPEG
#
# Примеры:
#   bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSemGeom
#   bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem
##################################################

set -e

SCENE=${1:-office0}
EXP=${2:-ActiveOpenSemGeom}
SEED=${3:-0}
ENABLE_VIS=${4:-0}
DEBUG=${5:-0}

export CUDA_VISIBLE_DEVICES=0,1
PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

echo "=============================================="
echo "  Scene      : $SCENE"
echo "  Config     : configs/Replica/${SCENE}/${EXP}.py"
echo "  Seed       : $SEED"
echo "  Visualize  : $ENABLE_VIS"
echo "  Debug      : $DEBUG"
echo "  CorrCLIP   : OFF (--corrclip 0)"
echo "  Run dir    : auto (run_N under experiment folder)"
echo "=============================================="

DEBUG_FLAG=""
[ "$DEBUG" = "1" ] && DEBUG_FLAG="--debug"

python src/main/activesgm.py \
    --cfg        "configs/Replica/${SCENE}/${EXP}.py" \
    --seed       "$SEED" \
    --enable_vis "$ENABLE_VIS" \
    --corrclip   0 \
    $DEBUG_FLAG

echo ""
echo "=== SLAM finished ==="
echo "    Check latest run_* under results/Replica/${SCENE}/"
