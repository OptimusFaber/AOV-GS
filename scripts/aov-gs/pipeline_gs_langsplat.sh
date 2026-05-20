#!/usr/bin/env bash
##################################################
# Pipeline 3 — CLIP + SAM + LangSplat (legacy: автоэнкодер 512→D)
#
# Конфиг: ActiveOpenSem (SplaTAM + SAM/CLIP → language_features/).
# Этапы: 01 SLAM → 02_train_clip_autoencoder → 03_train_gaussian_lang_field (--legacy).
#
# Использование:
#   bash scripts/aov-gs/pipeline_gs_langsplat.sh \\
#       [SCENE] [SEED] [ENABLE_VIS] [DEBUG] \\
#       [LATENT_DIM] [AE_EPOCHS] [DEVICE] [SAM_LEVEL] [LANG_ITERS] [RENDER_CKPT]
#
# Пример (по умолчанию office0, latent 64, уровень SAM s):
#   bash scripts/aov-gs/pipeline_gs_langsplat.sh
#
# Пример с визуализацией:
#   bash scripts/aov-gs/pipeline_gs_langsplat.sh office0 0 1 0 64 100 cuda:0 s 30000 auto
##################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE="${1:-office0}"
SEED="${2:-0}"
ENABLE_VIS="${3:-0}"
DEBUG="${4:-0}"
MASK_COLLECTOR="${5:-sam}"
RESULT_RUN="${6:-run_0}"
LATENT_DIM="${7:-64}"
AE_EPOCHS="${8:-100}"
DEVICE="${9:-cuda:0}"
SAM_LEVEL="${10:-s}"
LANG_ITERS="${11:-30000}"
RENDER_CKPT="${12:-auto}"

EXP="ActiveOpenSem"
RESULT_DIR="${PROJ_DIR}/results/Replica/${SCENE}/${EXP}/${RESULT_RUN}"

echo "=============================================="
echo "  Pipeline 3: CLIP+SAM + LangSplat (AE)"
echo "  Scene         : ${SCENE}"
echo "  Result dir    : ${RESULT_DIR}"
echo "  latent_dim D  : ${LATENT_DIM}"
echo "  SAM level     : ${SAM_LEVEL}"
echo "  Mask collector: ${MASK_COLLECTOR}"
echo "  Result run    : ${RESULT_RUN}"
echo "=============================================="

bash "${SCRIPT_DIR}/01_slam_exploration.sh" "${SCENE}" "${EXP}" "${SEED}" "${ENABLE_VIS}" "${DEBUG}" "${RESULT_RUN}" "${MASK_COLLECTOR}"

bash "${SCRIPT_DIR}/02_train_clip_autoencoder.sh" \
  "${RESULT_DIR}" "${LATENT_DIM}" "${AE_EPOCHS}" "${DEVICE}"

bash "${SCRIPT_DIR}/03_train_gaussian_lang_field.sh" \
  "${RESULT_DIR}" "${LATENT_DIM}" "${SAM_LEVEL}" "${LANG_ITERS}" "${DEVICE}" "${RENDER_CKPT}"

OUT_FIELD="${RESULT_DIR}/lang_field_${SAM_LEVEL}${LATENT_DIM}/lang_field.pt"
echo ""
echo "=== Pipeline 3 finished ==="
echo "    Language field: ${OUT_FIELD}"
