#!/usr/bin/env bash
##################################################
# Pipeline 1 — Gaussian Splatting без сегментатора
#
# Конфиг: ActiveGS (SplaTAM + геометрический active_gs, без SAM/CLIP/OneFormer).
# Результат: только карта гауссианов и оценки SplaTAM, без language_features/.
#
# Использование:
#   bash scripts/aov-gs/pipeline_gs_no_segmenter.sh [SCENE] [SEED] [ENABLE_VIS] [DEBUG]
#
# Пример:
#   bash scripts/aov-gs/pipeline_gs_no_segmenter.sh office0 0 0 0
##################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE="${1:-office0}"
SEED="${2:-0}"
ENABLE_VIS="${3:-0}"
DEBUG="${4:-0}"
EXP="ActiveGS"
RESULT_DIR="${PROJ_DIR}/results/Replica/${SCENE}/${EXP}/run_0"

echo "=============================================="
echo "  Pipeline 1: GS without segmenter"
echo "  Scene       : ${SCENE}"
echo "  Config      : configs/Replica/${SCENE}/${EXP}.py"
echo "  Result dir  : ${RESULT_DIR}"
echo "=============================================="

bash "${SCRIPT_DIR}/01_slam_exploration.sh" "${SCENE}" "${EXP}" "${SEED}" "${ENABLE_VIS}" "${DEBUG}"

echo ""
echo "=== Pipeline 1 finished ==="
echo "    Gaussians: ${RESULT_DIR}/splatam/final/params0.npz (или params.npz)"
echo "    (языковые фичи и этапы 2–3 не применяются)"
