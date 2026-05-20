#!/bin/bash
##################################################
# Step 1: SLAM exploration + SAM/CLIP extraction (CorrCLIP-style postproc **off**)
#
# Запускает ActiveSGM для проезда по сцене. В отличие от конфига по умолчанию,
# передаётся --corrclip 0 → отключаются merge похожих масок и inter-class suppression
# (см. configs/.../ActiveOpenSem.py → sam_clip, src/semantic/sam_clip_extractor.py).
#
# По окончании в RESULT_DIR будут:
#   splatam/final/params0.npz   – чекпойнт гауссиан
#   keyframe_poses.json         – позы ключевых кадров
#   language_features/*_f.npy  – CLIP-фичи масок (512d)
#   language_features/*_s.npy  – размеры масок SAM
#   keyframes/frame_*.jpg       – RGB-кейфреймы (только с --debug)
#
# Использование:
#   bash scripts/aov-gs/01_slam_exploration.sh [SCENE] [EXP] [SEED] [ENABLE_VIS] [DEBUG] [RESULT_RUN]
#
#   ENABLE_VIS: 0 = без окон OpenCV (RGB-D), 1 = показать live-визуализацию (нужен дисплей / X11)
#   DEBUG:      1 = дополнительно писать keyframes/ и segmentframes/
#   RESULT_RUN: подпапка результатов (по умолч. run_0)
#   MASK_COLLECTOR: sam | corrclip — обычный SAM или CorrCLIP-постобработка (7-й аргумент)
#
# Примеры:
#   bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem 0 0 0 run_0 sam
#   bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem 0 0 0 run_corrclip corrclip
##################################################

set -e

SCENE=${1:-office0}
EXP=${2:-ActiveOpenSem}
SEED=${3:-0}
ENABLE_VIS=${4:-0}   # 1 = --enable_vis 1 → main_cfg.visualizer.vis_rgbd = True (окна OpenCV)
DEBUG=${5:-0}        # 1 = сохранять keyframes/ как JPEG
RESULT_RUN=${6:-run_0}
MASK_COLLECTOR=${7:-sam}

export CUDA_VISIBLE_DEVICES=0,1
PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

CORRCLIP_FLAG=0
case "${MASK_COLLECTOR,,}" in
  sam|plain|default|0) CORRCLIP_FLAG=0 ;;
  corrclip|corr|1)     CORRCLIP_FLAG=1 ;;
  *)
    echo "ERROR: MASK_COLLECTOR must be 'sam' or 'corrclip', got: ${MASK_COLLECTOR}"
    exit 1
    ;;
esac

RESULT_DIR="${PROJ_DIR}/results/Replica/${SCENE}/${EXP}/${RESULT_RUN}"
mkdir -p "$RESULT_DIR"

echo "=============================================="
echo "  Scene      : $SCENE"
echo "  Config     : configs/Replica/${SCENE}/${EXP}.py"
echo "  Result dir : $RESULT_DIR"
echo "  Seed       : $SEED"
echo "  Visualize  : $ENABLE_VIS"
echo "  Debug      : $DEBUG"
echo "  Mask mode  : ${MASK_COLLECTOR} (--corrclip ${CORRCLIP_FLAG})"
echo "=============================================="

DEBUG_FLAG=""
[ "$DEBUG" = "1" ] && DEBUG_FLAG="--debug"

python src/main/activesgm.py \
    --cfg        "configs/Replica/${SCENE}/${EXP}.py" \
    --seed       "$SEED" \
    --result_dir "$RESULT_DIR" \
    --enable_vis "$ENABLE_VIS" \
    --corrclip   "$CORRCLIP_FLAG" \
    $DEBUG_FLAG

echo ""
echo "=== SLAM finished ==="
echo "    Gaussians : ${RESULT_DIR}/splatam/final/params0.npz"
echo "    Poses     : ${RESULT_DIR}/keyframe_poses.json"
echo "    CLIP feats: ${RESULT_DIR}/language_features/"
[ "$DEBUG" = "1" ] && echo "    Keyframes : ${RESULT_DIR}/keyframes/"
