#!/bin/bash
##################################################
# Step 1: SLAM exploration + SAM/CLIP extraction
#
# Запускает ActiveSGM для проезда по сцене.
# По окончании в RESULT_DIR будут:
#   splatam/final/params0.npz   – чекпойнт гауссиан
#   keyframe_poses.json         – позы ключевых кадров
#   language_features/*_f.npy  – CLIP-фичи масок (512d)
#   language_features/*_s.npy  – размеры масок SAM
#   keyframes/frame_*.jpg       – RGB-кейфреймы (только с --debug)
#
# Использование:
#   bash scripts/aov-gs/01_slam_exploration.sh [SCENE] [EXP] [SEED] [ENABLE_VIS] [DEBUG]
#
#   ENABLE_VIS: 0 = без окон OpenCV (RGB-D), 1 = показать live-визуализацию (нужен дисплей / X11)
#   DEBUG:      1 = дополнительно писать keyframes/ и segmentframes/
#
# Примеры:
#   bash scripts/aov-gs/01_slam_exploration.sh
#   bash scripts/aov-gs/01_slam_exploration.sh office0
#   bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem 0 0 1
#   # визуализация RGB-D + debug:
#   bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem 0 1 1
##################################################

set -e

SCENE=${1:-office0}
EXP=${2:-ActiveOpenSem}
SEED=${3:-0}
ENABLE_VIS=${4:-0}   # 1 = --enable_vis 1 → main_cfg.visualizer.vis_rgbd = True (окна OpenCV)
DEBUG=${5:-0}        # 1 = сохранять keyframes/ как JPEG

export CUDA_VISIBLE_DEVICES=0,1
PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

RESULT_DIR="${PROJ_DIR}/results/Replica/${SCENE}/${EXP}/run_0"
mkdir -p "$RESULT_DIR"

echo "=============================================="
echo "  Scene      : $SCENE"
echo "  Config     : configs/Replica/${SCENE}/${EXP}.py"
echo "  Result dir : $RESULT_DIR"
echo "  Seed       : $SEED"
echo "  Visualize  : $ENABLE_VIS"
echo "  Debug      : $DEBUG"
echo "=============================================="

DEBUG_FLAG=""
[ "$DEBUG" = "1" ] && DEBUG_FLAG="--debug"

python src/main/activesgm.py \
    --cfg        "configs/Replica/${SCENE}/${EXP}.py" \
    --seed       "$SEED" \
    --result_dir "$RESULT_DIR" \
    --enable_vis "$ENABLE_VIS" \
    $DEBUG_FLAG

echo ""
echo "=== SLAM finished ==="
echo "    Gaussians : ${RESULT_DIR}/splatam/final/params0.npz"
echo "    Poses     : ${RESULT_DIR}/keyframe_poses.json"
echo "    CLIP feats: ${RESULT_DIR}/language_features/"
[ "$DEBUG" = "1" ] && echo "    Keyframes : ${RESULT_DIR}/keyframes/"
