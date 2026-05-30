#!/bin/bash
##################################################
# Step 1: SLAM exploration + SAM/CLIP extraction (CorrCLIP-style postproc **on**)
#
# Использует значения [sam_clip] из конфига (например corrclip_mask_merge,
# corrclip_interclass_suppress_alpha в configs/Replica/office0/ActiveOpenSem_base.py).
# Передаётся --corrclip 1 → cfg_loader оставляет [sam_clip] как в конфиге
# (merge + inter-class suppression включены, если так задано в .py).
#
# Для отключения CorrCLIP используйте scripts/aov-gs/01_slam_exploration.sh
#
# Использование:
#   bash scripts/aov-gs/01_slam_exploration_with_corr_clip.sh [SCENE] [EXP] [SEED] [ENABLE_VIS] [DEBUG] [RESULT_RUN]
#
# Примеры:
#   bash scripts/aov-gs/01_slam_exploration_with_corr_clip.sh
#   bash scripts/aov-gs/01_slam_exploration_with_corr_clip.sh office0 ActiveOpenSem 0 0 0 run_corrclip
##################################################

set -e

SCENE=${1:-office0}
EXP=${2:-ActiveOpenSem}
SEED=${3:-0}
ENABLE_VIS=${4:-0}
DEBUG=${5:-0}
RESULT_RUN=${6:-run_0}

export CUDA_VISIBLE_DEVICES=0,1
PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

RESULT_DIR="${PROJ_DIR}/results/Replica/${SCENE}/${EXP}/${RESULT_RUN}"
mkdir -p "$RESULT_DIR"

echo "=============================================="
echo "  Scene      : $SCENE"
echo "  Config     : configs/Replica/${SCENE}/${EXP}.py"
echo "  Result dir : $RESULT_DIR"
echo "  Seed       : $SEED"
echo "  Visualize  : $ENABLE_VIS"
echo "  Debug      : $DEBUG"
echo "  CorrCLIP   : ON (from config [sam_clip])"
echo "=============================================="

DEBUG_FLAG=""
[ "$DEBUG" = "1" ] && DEBUG_FLAG="--debug"

python src/main/activesgm.py \
    --cfg        "configs/Replica/${SCENE}/${EXP}.py" \
    --seed       "$SEED" \
    --result_dir "$RESULT_DIR" \
    --enable_vis "$ENABLE_VIS" \
    --corrclip   1 \
    $DEBUG_FLAG

echo ""
echo "=== SLAM finished ==="
echo "    Gaussians : ${RESULT_DIR}/splatam/final/params0.npz"
echo "    Poses     : ${RESULT_DIR}/keyframe_poses.json"
echo "    CLIP feats: ${RESULT_DIR}/language_features/"
if [ "$DEBUG" = "1" ]; then
  echo "    Keyframes : ${RESULT_DIR}/keyframes/"
fi
