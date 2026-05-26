#!/bin/bash
##################################################
# Step 3: Train LangSplatV2 language field.
#
# Читает params0.npz + keyframe_poses.json + language_features/,
# обучает codebook + sparse coefficients и сохраняет lang_field.pt.
#
# CorrCLIP (merge/suppress масок SAM+CLIP) задаётся только на этапе 01 —
# он влияет на содержимое language_features/. Отдельного флага здесь нет:
# укажите RESULT_DIR от нужного прогона (01_slam_exploration.sh vs
# 01_slam_exploration_with_corr_clip.sh и разные RESULT_RUN).
#
# Использование:
#   bash scripts/aov-gs/03_train_gaussian_lang_field.sh \
#       [RESULT_DIR] [K] [LEVEL] [NUM_ITERS] [DEVICE] [L] [TOPK] [RENDER_CHECKPOINT] [TRAIN_DOWNSCALE] [LOG_EVERY]
#
# Примеры:
#   bash scripts/aov-gs/03_train_gaussian_lang_field.sh \
#       results/Replica/office0/ActiveOpenSem/run_0 64
#
#   bash scripts/aov-gs/03_train_gaussian_lang_field.sh \
#       results/Replica/room0/ActiveOpenSem/run_0 64 s 30000 cuda:1 1 4 auto
#
# Все уровни SAM подряд (s → m → l):
#   bash scripts/aov-gs/03_train_gaussian_lang_field_all_levels.sh \
#       results/Replica/office0/ActiveOpenSem/run_0 64 12000
#
# nohup важно оборачивать всю цепочку в bash -c '...', иначе nohup действует только на первый вызов:
#   cd /path/to/AOV-GS
#   nohup bash -c 'bash scripts/aov-gs/03_train_gaussian_lang_field.sh results/... 64 s 12000 && \
#                  bash scripts/aov-gs/03_train_gaussian_lang_field.sh results/... 64 m 12000 && \
#                  bash scripts/aov-gs/03_train_gaussian_lang_field.sh results/... 64 l 12000' \
#       > train_lang_field.log 2>&1 &
#
# Аргументы:
#   RESULT_DIR         – папка с результатами SLAM (обязательный)
#   K                  – размер codebook (K), по умолч. 64
#   LEVEL              – уровень SAM: s/m/l, по умолч. s
#   NUM_ITERS          – итерации, по умолч. 30000
#   DEVICE             – cuda:0 / cuda:1, по умолч. cuda:0
#   L                  – число уровней RVQ (L), по умолч. 1
#   TOPK               – разрежение top-k по уровням, по умолч. 4
#   RENDER_CHECKPOINT  – auto | on | off (по умолч. auto)
#   TRAIN_DOWNSCALE    – масштаб рендера при train (0,1], по умолч. 0.5
#                        (меньше — меньше VRAM при том же чекпойнте; см. README)
#   LOG_EVERY          – каждые N итераций: loss-лог + при улучшении скользящего среднего — best.pt
#
#   RENDER_CHECKPOINT (8-й аргумент):
#                        off = склейка графа всех проходов (много VRAM)
#                        on  = checkpoint на каждый проход (мало VRAM)
#                        auto = checkpoint только для latent_dim=64
##################################################

set -e

RESULT_DIR=${1:?"Usage: $0 RESULT_DIR [K] [LEVEL] [NUM_ITERS] [DEVICE] [L] [TOPK] [RENDER_CHECKPOINT] [TRAIN_DOWNSCALE] [LOG_EVERY]"}
CODEBOOK_SIZE=${2:-64}
LEVEL=${3:-s}
NUM_ITERS=${4:-30000}
DEVICE=${5:-cuda:0}
VQ_LAYER_NUM=${6:-1}
TOPK=${7:-4}
RENDER_CHECKPOINT=${8:-auto}
TRAIN_DOWNSCALE=${9:-0.5}
LOG_EVERY=${10:-500}

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
FEATURES_DIR="${RESULT_DIR}/language_features"
OUTPUT_DIR="${RESULT_DIR}/lang_field_${LEVEL}k${CODEBOOK_SIZE}_l${VQ_LAYER_NUM}"

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
    echo "ERROR: RAW фичи не найдены: $FEATURES_DIR"
    echo "       Сначала выполните 01_slam_exploration.sh"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  Result dir   : $RESULT_DIR"
echo "  Checkpoint   : $CHECKPOINT"
echo "  Poses        : $POSES"
echo "  Features dir : $FEATURES_DIR"
echo "  SAM level    : $LEVEL  (s=small, m=medium, l=large)"
echo "  Codebook K   : $CODEBOOK_SIZE"
echo "  VQ layers L  : $VQ_LAYER_NUM"
echo "  Top-k        : $TOPK"
echo "  Iterations   : $NUM_ITERS"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Device       : $DEVICE"
echo "  Render ckpt  : $RENDER_CHECKPOINT  (auto|on|off)"
echo "  Downscale    : $TRAIN_DOWNSCALE"
echo "  Log / best.pt: every $LOG_EVERY iters (rolling window = $LOG_EVERY)"
echo "=============================================="

python scripts/train_language_field.py \
    --checkpoint         "$CHECKPOINT" \
    --poses              "$POSES" \
    --features_dir       "$FEATURES_DIR" \
    --level              "$LEVEL" \
    --output_dir         "$OUTPUT_DIR" \
    --codebook_size      "$CODEBOOK_SIZE" \
    --vq_layer_num       "$VQ_LAYER_NUM" \
    --topk               "$TOPK" \
    --num_iters          "$NUM_ITERS" \
    --device             "$DEVICE" \
    --render_checkpoint  "$RENDER_CHECKPOINT" \
    --train_downscale    "$TRAIN_DOWNSCALE" \
    --log_every          "$LOG_EVERY"

echo ""
echo "=== Language field обучено ==="
echo "    Финал (лучшее по скользящему среднему): ${OUTPUT_DIR}/lang_field.pt"
echo "    Чекпойнты во время обучения (при улучшении): ${OUTPUT_DIR}/best.pt"
