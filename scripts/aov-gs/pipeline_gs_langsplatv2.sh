#!/usr/bin/env bash
##################################################
# Pipeline 4 — CLIP + SAM + LangSplatV2 (codebook + sparse logits, без AE)
#
# Конфиг: ActiveOpenSem. Этапы: 01 → 02_validate_features_langsplatv2 (no-op) →
#         03_train_gaussian_lang_field_langsplatv2.
# Нужен scikit-learn для инициализации codebook (KMeans).
#
# Использование:
#   bash scripts/aov-gs/pipeline_gs_langsplatv2.sh \\
#       [SCENE] [SEED] [ENABLE_VIS] [DEBUG] \\
#       [K] [SAM_LEVEL] [LANG_ITERS] [DEVICE] [L] [TOPK] [RENDER_CKPT] [TRAIN_DOWNSCALE]
#
# Пример:
#   bash scripts/aov-gs/pipeline_gs_langsplatv2.sh office0 0 0 0 64 s 30000 cuda:0 1 4 auto 1.0
##################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE="${1:-office0}"
SEED="${2:-0}"
ENABLE_VIS="${3:-0}"
DEBUG="${4:-0}"
CODEBOOK_K="${5:-64}"
SAM_LEVEL="${6:-s}"
LANG_ITERS="${7:-30000}"
DEVICE="${8:-cuda:0}"
VQ_L="${9:-1}"
TOPK="${10:-4}"
RENDER_CKPT="${11:-auto}"
TRAIN_DS="${12:-1.0}"

EXP="ActiveOpenSem"
RESULT_DIR="${PROJ_DIR}/results/Replica/${SCENE}/${EXP}/run_0"

echo "=============================================="
echo "  Pipeline 4: CLIP+SAM + LangSplatV2"
echo "  Scene         : ${SCENE}"
echo "  Result dir    : ${RESULT_DIR}"
echo "  Codebook K    : ${CODEBOOK_K}"
echo "  VQ layers L   : ${VQ_L}  topk=${TOPK}"
echo "=============================================="

bash "${SCRIPT_DIR}/01_slam_exploration.sh" "${SCENE}" "${EXP}" "${SEED}" "${ENABLE_VIS}" "${DEBUG}"

bash "${SCRIPT_DIR}/02_validate_features_langsplatv2.sh" "${RESULT_DIR}"

bash "${SCRIPT_DIR}/03_train_gaussian_lang_field_langsplatv2.sh" \
  "${RESULT_DIR}" "${CODEBOOK_K}" "${SAM_LEVEL}" "${LANG_ITERS}" "${DEVICE}" \
  "${VQ_L}" "${TOPK}" "${RENDER_CKPT}" "${TRAIN_DS}"

OUT_FIELD="${RESULT_DIR}/lang_field_${SAM_LEVEL}k${CODEBOOK_K}_l${VQ_L}/lang_field.pt"
echo ""
echo "=== Pipeline 4 finished ==="
echo "    Language field: ${OUT_FIELD}"
