#!/bin/bash
##################################################
# LangSplatV2: sequential s → m → l (or parallel via batch script).
#
#   bash scripts/aov-gs/03_train_gaussian_lang_field_all_levels.sh \
#       RESULT_DIR [K] [NUM_ITERS] [DEVICE] [L] [TOPK] [RENDER_CHECKPOINT]
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
    bash "${SCRIPT_DIR}/03_train_gaussian_lang_field.sh" \
        "${RESULT_DIR}" "${CODEBOOK_SIZE}" "${level}" "${NUM_ITERS}" "${DEVICE}" \
        "${VQ_LAYER_NUM}" "${TOPK}" "${RENDER_CHECKPOINT}"
}

echo "=== Train language field: levels s → m → l ==="
echo "    RESULT_DIR=${RESULT_DIR}  K=${CODEBOOK_SIZE}  NUM_ITERS=${NUM_ITERS}"

run_level s
run_level m
run_level l

echo "=== All three levels finished ==="
