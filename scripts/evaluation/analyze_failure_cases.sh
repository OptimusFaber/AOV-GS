#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <scene> [options]"
    echo "Example: $0 office1 --topk_threshold 8"
    echo "Options: --result_dir <path> to specify custom results path"
    exit 1
fi

SCENE=$1
shift

DATASET="Replica"
CONFIG_PATH="configs/${DATASET}/${SCENE}/ActiveSem.py"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config not found: $CONFIG_PATH"
    exit 1
fi

RESULT_DIR_PATTERN="results/${DATASET}/${SCENE}/ActiveSem/run_*"
RESULT_DIRS=($(ls -d $RESULT_DIR_PATTERN 2>/dev/null | sort -V))

if [ ${#RESULT_DIRS[@]} -gt 0 ]; then
    LATEST_DIR="${RESULT_DIRS[-1]}"
    echo "Found results: $LATEST_DIR"
    
    python src/evaluation/visualize_failure_cases.py \
        --cfg "$CONFIG_PATH" \
        --result_dir "$LATEST_DIR" \
        "$@"
else
    echo "Warning: No results found in $RESULT_DIR_PATTERN"
    echo "Trying default path..."
    
    python src/evaluation/visualize_failure_cases.py \
        --cfg "$CONFIG_PATH" \
        "$@"
fi
