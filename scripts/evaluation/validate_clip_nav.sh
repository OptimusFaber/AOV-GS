#!/bin/bash
##################################################
### Interactive CLIP navigation validator
###
### Loads keyframe images + camera poses from a
### completed ActiveSGM run and lets you type text
### queries to visualise where the robot would go.
###
### Запуск из корня репозитория ActiveSGM.
###
### Usage:
###   bash scripts/evaluation/validate_clip_nav.sh [SCENE] [EXP] [RUN] [DEVICE]
###
### Examples:
###   bash scripts/evaluation/validate_clip_nav.sh office0 ActiveOpenSem 0 cuda:0
###   bash scripts/evaluation/validate_clip_nav.sh office0 ActiveOpenSem bench_clip cuda:0
###   # Без GUI (сервер/headless):
###   bash scripts/evaluation/validate_clip_nav.sh office0 ActiveOpenSem 0 cuda:0 --headless
##################################################

SCENE=${1:-office0}
EXP=${2:-ActiveOpenSem}
RUN=${3:-0}
CLIP_DEVICE=${4:-cuda:0}
EXTRA_ARGS="${@:5}"   # e.g. --headless

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJ_DIR"

DATASET=Replica
RESULT_DIR=results/${DATASET}/${SCENE}/${EXP}/run_${RUN}

echo "=== CLIP Navigation Validator ==="
echo "Scene      : ${SCENE}"
echo "Experiment : ${EXP}  run_${RUN}"
echo "Result dir : ${RESULT_DIR}"
echo "CLIP device: ${CLIP_DEVICE}"
echo ""

python src/evaluation/validate_clip_nav.py \
    --result_dir  "${RESULT_DIR}" \
    --clip_device "${CLIP_DEVICE}" \
    --model_name  ViT-B-32 \
    --pretrained  openai \
    --top_k       5 \
    ${EXTRA_ARGS}
