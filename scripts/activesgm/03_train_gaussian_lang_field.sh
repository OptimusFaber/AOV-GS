#!/bin/bash
##################################################
# Step 3: Train language field → добавить языковые
#         параметры к замороженным гауссианам
#
# Читает params0.npz + keyframe_poses.json + language_features_dimN/,
# обучает языковое поле и сохраняет lang_field.pt.
#
# Использование:
#   bash scripts/activesgm/03_train_gaussian_lang_field.sh \
#       [RESULT_DIR] [LATENT_DIM] [LEVEL] [NUM_ITERS] [DEVICE] [RENDER_CHECKPOINT]
#
# Примеры:
#   bash scripts/activesgm/03_train_gaussian_lang_field.sh \
#       results/Replica/office0/ActiveOpenVocab/run_0 64
#
#   bash scripts/activesgm/03_train_gaussian_lang_field.sh \
#       results/Replica/room0/ActiveOpenVocab/run_0 3 s 30000 cuda:1 auto
#
# Аргументы:
#   RESULT_DIR         – папка с результатами SLAM (обязательный)
#   LATENT_DIM         – размерность AE (3/4/8/16/32/64), по умолч. 64
#   LEVEL              – уровень SAM: s/m/l, по умолч. s
#   NUM_ITERS          – итерации, по умолч. 30000
#   DEVICE             – cuda:0 / cuda:1, по умолч. cuda:0
#   RENDER_CHECKPOINT  – auto | on | off (по умолч. auto)
#                        off = склейка графа всех проходов (много VRAM)
#                        on  = checkpoint на каждый проход (мало VRAM)
#                        auto = checkpoint только для latent_dim=64
##################################################

set -e

RESULT_DIR=${1:?"Usage: $0 RESULT_DIR [LATENT_DIM] [LEVEL] [NUM_ITERS] [DEVICE] [RENDER_CHECKPOINT]"}
LATENT_DIM=${2:-64}
LEVEL=${3:-s}
NUM_ITERS=${4:-30000}
DEVICE=${5:-cuda:0}
RENDER_CHECKPOINT=${6:-auto}

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

FINAL_DIR="${RESULT_DIR}/splatam/final"
# SplaTAM может сохранить как params0.npz или params.npz в зависимости от конфига/версии
if [ -f "${FINAL_DIR}/params0.npz" ]; then
    CHECKPOINT="${FINAL_DIR}/params0.npz"
elif [ -f "${FINAL_DIR}/params.npz" ]; then
    CHECKPOINT="${FINAL_DIR}/params.npz"
else
    CHECKPOINT="${FINAL_DIR}/params0.npz"
fi
POSES="${RESULT_DIR}/keyframe_poses.json"
FEATURES_DIR="${RESULT_DIR}/language_features_dim${LATENT_DIM}"
OUTPUT_DIR="${RESULT_DIR}/lang_field_${LEVEL}${LATENT_DIM}"

# ── Проверка зависимостей ─────────────────────────────────────────────────
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Чекпойнт не найден: ни ${FINAL_DIR}/params0.npz, ни ${FINAL_DIR}/params.npz"
    echo "       Сначала выполните 01_slam_exploration.sh"
    exit 1
fi

if [ ! -f "$POSES" ]; then
    echo "ERROR: keyframe_poses.json не найден: $POSES"
    echo "       Сначала выполните 01_slam_exploration.sh"
    exit 1
fi

if [ ! -d "$FEATURES_DIR" ]; then
    echo "ERROR: Сжатые фичи не найдены: $FEATURES_DIR"
    echo "       Сначала выполните 02_train_clip_autoencoder.sh с LATENT_DIM=${LATENT_DIM}"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  Result dir   : $RESULT_DIR"
echo "  Checkpoint   : $CHECKPOINT"
echo "  Poses        : $POSES"
echo "  Features dir : $FEATURES_DIR"
echo "  SAM level    : $LEVEL  (s=small, m=medium, l=large)"
echo "  Latent dim   : $LATENT_DIM"
echo "  Iterations   : $NUM_ITERS"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Device       : $DEVICE"
echo "  Render ckpt  : $RENDER_CHECKPOINT  (auto|on|off)"
echo "=============================================="

python scripts/train_language_field.py \
    --checkpoint         "$CHECKPOINT" \
    --poses              "$POSES" \
    --features_dir       "$FEATURES_DIR" \
    --level              "$LEVEL" \
    --output_dir         "$OUTPUT_DIR" \
    --latent_dim         "$LATENT_DIM" \
    --num_iters          "$NUM_ITERS" \
    --device             "$DEVICE" \
    --render_checkpoint  "$RENDER_CHECKPOINT"

echo ""
echo "=== Language field обучено ==="
echo "    Модель: ${OUTPUT_DIR}/lang_field.pt"
