#!/usr/bin/env bash
##################################################
# Unified open-vocabulary pipeline (ActiveOpenSem)
#
# MASK_COLLECTOR  sam | corrclip   — stage 1
# LANG_MODE       langsplat | langsplatv2 — stages 2–3
#
# Usage:
#   bash scripts/aov-gs/pipeline_gs_open_vocab.sh [SCENE] [SEED] [ENABLE_VIS] [DEBUG] \\
#       [MASK_COLLECTOR] [LANG_MODE] [RESULT_RUN] [extra train args...]
#
# Examples:
#   bash scripts/aov-gs/pipeline_gs_open_vocab.sh
#   bash scripts/aov-gs/pipeline_gs_open_vocab.sh office0 0 0 0 corrclip langsplatv2 run_corrclip
#   bash scripts/aov-gs/pipeline_gs_open_vocab.sh office0 0 0 0 sam langsplat run_ls 64 100 s 30000 cuda:0 auto
##################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE="${1:-office0}"
SEED="${2:-0}"
ENABLE_VIS="${3:-0}"
DEBUG="${4:-0}"
MASK_COLLECTOR="${5:-sam}"
LANG_MODE="${6:-langsplatv2}"
RESULT_RUN="${7:-run_0}"

EXP="ActiveOpenSem"
RESULT_DIR="${PROJ_DIR}/results/Replica/${SCENE}/${EXP}/${RESULT_RUN}"

echo "=============================================="
echo "  Open-vocab pipeline"
echo "  Scene           : ${SCENE}"
echo "  Mask collector  : ${MASK_COLLECTOR}"
echo "  Lang mode       : ${LANG_MODE}"
echo "  Result dir      : ${RESULT_DIR}"
echo "=============================================="

bash "${SCRIPT_DIR}/01_slam_exploration.sh" \
  "${SCENE}" "${EXP}" "${SEED}" "${ENABLE_VIS}" "${DEBUG}" \
  "${RESULT_RUN}" "${MASK_COLLECTOR}"

case "${LANG_MODE,,}" in
  langsplatv2|v2)
    bash "${SCRIPT_DIR}/02_validate_features_langsplatv2.sh" "${RESULT_DIR}"
    K="${8:-64}"
    SAM_LEVEL="${9:-s}"
    LANG_ITERS="${10:-30000}"
    DEVICE="${11:-cuda:0}"
    VQ_L="${12:-1}"
    TOPK="${13:-4}"
    RENDER_CKPT="${14:-auto}"
    TRAIN_DS="${15:-1.0}"
    bash "${SCRIPT_DIR}/03_train_gaussian_lang_field_langsplatv2.sh" \
      "${RESULT_DIR}" "${K}" "${SAM_LEVEL}" "${LANG_ITERS}" "${DEVICE}" \
      "${VQ_L}" "${TOPK}" "${RENDER_CKPT}" "${TRAIN_DS}"
    OUT="${RESULT_DIR}/lang_field_${SAM_LEVEL}k${K}_l${VQ_L}/lang_field.pt"
    ;;
  langsplat|legacy)
    LATENT_DIM="${8:-64}"
    AE_EPOCHS="${9:-100}"
    SAM_LEVEL="${10:-s}"
    LANG_ITERS="${11:-30000}"
    DEVICE="${12:-cuda:0}"
    RENDER_CKPT="${13:-auto}"
    bash "${SCRIPT_DIR}/02_train_clip_autoencoder.sh" \
      "${RESULT_DIR}" "${LATENT_DIM}" "${AE_EPOCHS}" "${DEVICE}"
    bash "${SCRIPT_DIR}/03_train_gaussian_lang_field.sh" \
      "${RESULT_DIR}" "${LATENT_DIM}" "${SAM_LEVEL}" "${LANG_ITERS}" "${DEVICE}" "${RENDER_CKPT}"
    OUT="${RESULT_DIR}/lang_field_${SAM_LEVEL}${LATENT_DIM}/lang_field.pt"
    ;;
  *)
    echo "ERROR: LANG_MODE must be langsplat or langsplatv2, got: ${LANG_MODE}"
    exit 1
    ;;
esac

echo ""
echo "=== Pipeline finished ==="
echo "    Language field: ${OUT}"
