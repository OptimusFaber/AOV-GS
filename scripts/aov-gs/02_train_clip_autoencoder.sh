#!/bin/bash
##################################################
# Step 2: Train CLIP autoencoder (512 → N → 512)
#
# Reads CLIP features from language_features/*_f.npy,
# trains an autoencoder with the given latent size,
# saves compressed features to language_features_dimN/.
#
# Supported LATENT_DIM values: 3, 4, 8, 16, 32, 64
#
# Usage:
#   bash scripts/aov-gs/02_train_clip_autoencoder.sh \
#       RESULT_DIR [LATENT_DIM] [NUM_EPOCHS] [DEVICE] [HIDDEN_DIMS]
#
#   Arguments (in order):
#     1  RESULT_DIR   — SLAM results directory (required)
#     2  LATENT_DIM   — AE output dim: 3/4/8/16/32/64  (default: 64)
#     3  NUM_EPOCHS   — number of training epochs                 (default: 100)
#     4  DEVICE       — cuda:0 / cuda:1                           (default: cuda:0)
#     5  HIDDEN_DIMS  — intermediate layers as space-separated quoted list
#                       if unset — default for LATENT_DIM is used
#                       example: "256 64"  →  512 → 256 → 64 → LATENT_DIM
#
# Examples:
#   bash scripts/aov-gs/02_train_clip_autoencoder.sh \
#       results/Replica/office0/ActiveOpenSem/run_0 64
#
#   bash scripts/aov-gs/02_train_clip_autoencoder.sh \
#       results/Replica/office0/ActiveOpenSem/run_0 4 100 cuda:0
#
#   # Custom architecture: 512 → 128 → 4
#   bash scripts/aov-gs/02_train_clip_autoencoder.sh \
#       results/Replica/office0/ActiveOpenSem/run_0 4 100 cuda:0 "128"
#
#   # Custom architecture: 512 → 256 → 32 → 4
#   bash scripts/aov-gs/02_train_clip_autoencoder.sh \
#       results/Replica/room0/ActiveOpenSem/run_0 4 200 cuda:1 "256 32"
##################################################

set -e

RESULT_DIR=${1:?"Usage: $0 RESULT_DIR [LATENT_DIM] [NUM_EPOCHS] [DEVICE] [HIDDEN_DIMS]"}
LATENT_DIM=${2:-64}
NUM_EPOCHS=${3:-100}
DEVICE=${4:-cuda:0}
HIDDEN_DIMS=${5:-""}   # optional: intermediate layers space-separated

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

# Scene name is taken from the path (for ckpt/ folder)
SCENE=$(basename "$(dirname "$(dirname "$RESULT_DIR")")")

# ── Layer sizes ────────────────────────────────────────────────────────────
# If HIDDEN_DIMS is set — use it, else default by LATENT_DIM
if [ -n "$HIDDEN_DIMS" ]; then
    ENCODER_DIMS="${HIDDEN_DIMS} ${LATENT_DIM}"
    # decoder — reverse hidden order + 512
    DECODER_DIMS="$(echo "$HIDDEN_DIMS" | awk '{for(i=NF;i>0;i--) printf $i" "} END{print "512"}')"
else
    case "$LATENT_DIM" in
        3)  ENCODER_DIMS="256 128 3";    DECODER_DIMS="128 256 512" ;;
        4)  ENCODER_DIMS="256 64 4";     DECODER_DIMS="64 256 512"  ;;
        8)  ENCODER_DIMS="256 64 8";     DECODER_DIMS="64 256 512"  ;;
        16) ENCODER_DIMS="256 128 16";   DECODER_DIMS="128 256 512" ;;
        32) ENCODER_DIMS="256 128 32";   DECODER_DIMS="128 256 512" ;;
        64) ENCODER_DIMS="256 128 64";   DECODER_DIMS="128 256 512" ;;
        *)
            echo "ERROR: LATENT_DIM='${LATENT_DIM}' without HIDDEN_DIMS is not supported."
            echo "       Set HIDDEN_DIMS or use LATENT_DIM from: 3, 4, 8, 16, 32, 64"
            exit 1
            ;;
    esac
fi

if [ ! -d "${RESULT_DIR}/language_features" ]; then
    echo "ERROR: language_features not found in ${RESULT_DIR}"
    echo "       Run 01_slam_exploration.sh first"
    exit 1
fi

echo "=============================================="
echo "  Result dir   : $RESULT_DIR"
echo "  Scene name   : $SCENE"
echo "  Latent dim   : $LATENT_DIM"
echo "  Encoder      : 512 → ${ENCODER_DIMS// / → }"
echo "  Decoder      : ${LATENT_DIM} → ${DECODER_DIMS// / → }"
echo "  Epochs       : $NUM_EPOCHS"
echo "  Device       : $DEVICE"
echo "  Output       : ${RESULT_DIR}/language_features_dim${LATENT_DIM}/"
echo "=============================================="

python scripts/train_language_autoencoder.py \
    --dataset_path "$RESULT_DIR" \
    --dataset_name "$SCENE" \
    --encoder_dims $ENCODER_DIMS \
    --decoder_dims $DECODER_DIMS \
    --num_epochs   "$NUM_EPOCHS" \
    --lr           1e-4 \
    --device       "$DEVICE"

echo ""
echo "=== Autoencoder (512→${LATENT_DIM}) trained ==="
echo "    Compressed features: ${RESULT_DIR}/language_features_dim${LATENT_DIM}/"
echo "    AE checkpoint: ckpt/${SCENE}/${LATENT_DIM}/best_ckpt.pth"
