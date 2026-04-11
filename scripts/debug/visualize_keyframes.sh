#!/bin/bash
# Скрипт для визуализации keyframes из результатов обучения

RESULT_DIR=${1:-"results/Replica/office0/ActiveSem/run_0"}
DATASET_DIR=${2:-"data/Replica/office0"}

echo "Визуализация keyframes..."
echo "  Result dir: $RESULT_DIR"
echo "  Dataset dir: $DATASET_DIR"

python src/debug/visualize_keyframes.py \
    --result_dir "$RESULT_DIR" \
    --dataset_dir "$DATASET_DIR"





