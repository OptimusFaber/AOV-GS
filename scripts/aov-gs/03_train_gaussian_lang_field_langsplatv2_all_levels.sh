#!/bin/bash
##################################################
# Последовательно обучает language field для всех уровней SAM: s → m → l
# (отдельные выходные каталоги: lang_field_s64, lang_field_m64, lang_field_l64).
#
# Использование:
#   bash scripts/aov-gs/03_train_gaussian_lang_field_all_levels.sh \
#       RESULT_DIR [K] [NUM_ITERS] [DEVICE] [L] [TOPK] [RENDER_CHECKPOINT]
#
# Пример (из корня AOV-GS):
#   bash scripts/aov-gs/03_train_gaussian_lang_field_all_levels.sh \
#       results/Replica/office0/ActiveOpenSem/run_0 64 12000 cuda:0 1 4 auto
#
# Фон + лог (правильный nohup на всю цепочку):
#   cd /path/to/AOV-GS
#   nohup bash scripts/aov-gs/03_train_gaussian_lang_field_all_levels.sh \
#       results/Replica/office0/ActiveOpenSem/run_0 64 12000 \
#       > train_lang_field_sml.log 2>&1 &
##################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJ_DIR}"

RESULT_DIR=${1:?"Usage: $0 RESULT_DIR [K] [NUM_ITERS] [DEVICE] [L] [TOPK] [RENDER_CHECKPOINT]"}
CODEBOOK_SIZE=${2:-64}
NUM_ITERS=${3:-30000}
DEVICE=${4:-cuda:0}
VQ_LAYER_NUM=${5:-1}
TOPK=${6:-4}
RENDER_CHECKPOINT=${7:-auto}

run_level() {
    local level="$1"
    bash "${SCRIPT_DIR}/03_train_gaussian_lang_field_langsplatv2.sh" \
        "${RESULT_DIR}" "${CODEBOOK_SIZE}" "${level}" "${NUM_ITERS}" "${DEVICE}" "${VQ_LAYER_NUM}" "${TOPK}" "${RENDER_CHECKPOINT}"
}

echo "=== Train language field: levels s → m → l ==="
echo "    RESULT_DIR=${RESULT_DIR}  K=${CODEBOOK_SIZE}  L=${VQ_LAYER_NUM}  TOPK=${TOPK}  NUM_ITERS=${NUM_ITERS}"

run_level s
run_level m
run_level l

echo ""
echo "=== Все три уровня (s, m, l) завершены ==="
