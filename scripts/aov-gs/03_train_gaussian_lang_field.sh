#!/bin/bash
##################################################
# Step 3: Train LangSplatV2 language field.
#
# Reads params0.npz + keyframe_poses.json + language_features/ (512-D RAW).
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

# nohup subshells often lose `python` from conda — use explicit interpreter
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PY="${CONDA_PREFIX}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

FINAL_DIR="${RESULT_DIR}/splatam/final"
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

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: ${FINAL_DIR}/params*.npz"
    exit 1
fi
if [ ! -f "$POSES" ]; then
    echo "ERROR: keyframe_poses.json not found: $POSES"
    exit 1
fi
if [ ! -d "$FEATURES_DIR" ]; then
    echo "ERROR: RAW features not found: $FEATURES_DIR"
    echo "       Need language_features/ after 01_slam_exploration (not language_features_dim64)"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  Result dir   : $RESULT_DIR"
echo "  Checkpoint   : $CHECKPOINT"
echo "  Features dir : $FEATURES_DIR  (LangSplatV2 RAW 512-D)"
echo "  SAM level    : $LEVEL"
echo "  Python       : $PY ($($PY --version 2>&1))"
print_train_device "$DEVICE"
echo "  Downscale    : $TRAIN_DOWNSCALE"
echo "=============================================="

"$PY" scripts/train_language_field.py \
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

echo "=== Language field → ${OUTPUT_DIR}/lang_field.pt ==="
