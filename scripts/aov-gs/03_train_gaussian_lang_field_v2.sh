#!/bin/bash
##################################################
# Step 3: Train LangSplatV2 language field.
#
# Reads params0.npz + keyframe_poses.json + language_features/,
# trains codebook + sparse coefficients and saves lang_field.pt.
#
# CorrCLIP (SAM+CLIP mask merge/suppress) is set only at stage 01 —
# it affects language_features/ contents. There is no separate flag here:
# point RESULT_DIR at the desired run (01_slam_exploration.sh vs
# 01_slam_exploration_with_corr_clip.sh and different RESULT_RUN).
#
# Usage:
#   bash scripts/aov-gs/03_train_gaussian_lang_field.sh \
#       [RESULT_DIR] [K] [LEVEL] [NUM_ITERS] [DEVICE] [L] [TOPK] [RENDER_CHECKPOINT] [TRAIN_DOWNSCALE] [LOG_EVERY]
#
# Examples:
#   bash scripts/aov-gs/03_train_gaussian_lang_field.sh \
#       results/Replica/office0/ActiveOpenSem/run_0 64
#
#   bash scripts/aov-gs/03_train_gaussian_lang_field.sh \
#       results/Replica/room0/ActiveOpenSem/run_0 64 s 30000 cuda:1 1 4 auto
#
# All SAM levels in sequence (s → m → l):
#   bash scripts/aov-gs/03_train_gaussian_lang_field_all_levels.sh \
#       results/Replica/office0/ActiveOpenSem/run_0 64 12000
#
# Important: wrap the whole chain in bash -c '...', otherwise nohup applies only to the first command:
#   cd /path/to/AOV-GS
#   nohup bash -c 'bash scripts/aov-gs/03_train_gaussian_lang_field.sh results/... 64 s 12000 && \
#                  bash scripts/aov-gs/03_train_gaussian_lang_field.sh results/... 64 m 12000 && \
#                  bash scripts/aov-gs/03_train_gaussian_lang_field.sh results/... 64 l 12000' \
#       > train_lang_field.log 2>&1 &
#
# Arguments:
#   RESULT_DIR         – SLAM results directory (required)
#   K                  – codebook size (K), default 64
#   LEVEL              – SAM level: s/m/l, default s
#   NUM_ITERS          – iterations, default 30000
#   DEVICE             – cuda:0 (default) or physical GPU index as cuda:N
#                        (cuda:1 → CUDA_VISIBLE_DEVICES=1, logical cuda:0)
#   L                  – number of RVQ levels (L), default 1
#   TOPK               – top-k sparsification per level, default 4
#   RENDER_CHECKPOINT  – auto | on | off (default auto)
#   TRAIN_DOWNSCALE    – render scale during train (0,1], default 0.5
#                        (lower — less VRAM at the same checkpoint; see README)
#   LOG_EVERY          – every N iters: loss log + on rolling-mean improvement — best.pt
#
#   RENDER_CHECKPOINT (8th argument):
#                        off = stitch graph of all passes (high VRAM)
#                        on  = checkpoint every pass (low VRAM)
#                        auto = checkpoint only for latent_dim=64
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
# shellcheck source=_gpu_helpers.sh
source "${PROJ_DIR}/scripts/aov-gs/_gpu_helpers.sh"
resolve_train_device DEVICE

cd "$PROJ_DIR"

FINAL_DIR="${RESULT_DIR}/splatam/final"
# SplaTAM may save as params0.npz or params.npz depending on config/version
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

# ── Dependency checks ─────────────────────────────────────────────────────
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: neither ${FINAL_DIR}/params0.npz nor ${FINAL_DIR}/params.npz"
    echo "       Run 01_slam_exploration.sh first"
    exit 1
fi

if [ ! -f "$POSES" ]; then
    echo "ERROR: keyframe_poses.json not found: $POSES"
    echo "       Run 01_slam_exploration.sh first"
    exit 1
fi

if [ ! -d "$FEATURES_DIR" ]; then
    echo "ERROR: RAW features not found: $FEATURES_DIR"
    echo "       Run 01_slam_exploration.sh first"
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
print_train_device "$DEVICE"
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
echo "=== Language field trained ==="
echo "    Final (best by rolling mean): ${OUTPUT_DIR}/lang_field.pt"
echo "    Training checkpoints (on improvement): ${OUTPUT_DIR}/best.pt"
