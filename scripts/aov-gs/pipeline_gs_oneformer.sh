#!/usr/bin/env bash
##################################################
# Pipeline 2 — Gaussian Splatting с OneFormer
#
# Конфиг: ActiveSem (semsplatam + OneFormer + семантический планировщик active_gsv2).
# Нужны веса OneFormer / HuggingFace (см. configs/Replica/<scene>/ActiveSem.py).
# Результат: семантическая карта; SAM+CLIP language_features здесь не пишутся.
#
# Использование:
#   bash scripts/aov-gs/pipeline_gs_oneformer.sh [SCENE] [SEED] [ENABLE_VIS] [DEBUG]
#
# Пример:
#   bash scripts/aov-gs/pipeline_gs_oneformer.sh office0 0 0 0
##################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE="${1:-office0}"
SEED="${2:-0}"
ENABLE_VIS="${3:-0}"
DEBUG="${4:-0}"
EXP="ActiveSem"
RESULT_DIR="${PROJ_DIR}/results/Replica/${SCENE}/${EXP}/run_0"

if [ ! -f "${PROJ_DIR}/configs/Replica/${SCENE}/${EXP}.py" ]; then
  echo "ERROR: нет конфига ${PROJ_DIR}/configs/Replica/${SCENE}/${EXP}.py"
  echo "       Скопируйте/адаптируйте ActiveSem.py для этой сцены (см. другие office*/ActiveSem.py)."
  exit 1
fi

echo "=============================================="
echo "  Pipeline 2: GS + OneFormer (ActiveSem)"
echo "  Scene       : ${SCENE}"
echo "  Config      : configs/Replica/${SCENE}/${EXP}.py"
echo "  Result dir  : ${RESULT_DIR}"
echo "=============================================="

bash "${SCRIPT_DIR}/01_slam_exploration.sh" "${SCENE}" "${EXP}" "${SEED}" "${ENABLE_VIS}" "${DEBUG}"

echo ""
echo "=== Pipeline 2 finished ==="
echo "    Gaussians: ${RESULT_DIR}/splatam/final/params0.npz (или params.npz)"
echo "    Для open-vocab языкового поля используйте pipeline 3 или 4 (ActiveOpenSem)."
