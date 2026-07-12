#!/usr/bin/env bash
##################################################
# MP3D YmJkqBEsHnH — train language field (s, m, l)
# for ActiveGeom and ActiveOpenSem (run_0).
#
# Prerequisites:
#   results/MP3D/YmJkqBEsHnH/{ActiveGeom,ActiveOpenSem}/run_0/
#     splatam/final/params*.npz, keyframe_poses.json, language_features/
#
# 2 GPUs: up to 2 level-jobs in parallel (PARALLEL_TRAIN=2)
#
# Usage:
#   bash scripts/aov-gs/run_mp3d_ymjk_lang_field.sh
#
# train only (no validate):
#   SKIP_VALIDATE=1 bash scripts/aov-gs/run_mp3d_ymjk_lang_field.sh
#
# Background:
#   mkdir -p logs/lang_field_ymjk
#   nohup bash scripts/aov-gs/run_mp3d_ymjk_lang_field.sh \
#     >> logs/lang_field_ymjk/orchestrator.log 2>&1 &
#   echo $! > logs/lang_field_ymjk/orchestrator.pid
#   tail -f logs/lang_field_ymjk/orchestrator.log
#
# Env:
#   GPU0=0 GPU1=1
#   PARALLEL_TRAIN=2 PARALLEL_VALIDATE=2
#   NUM_ITERS=12000 TRAIN_DOWNSCALE=0.5 SKIP_DONE=1
##################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export SCENE="YmJkqBEsHnH"
export METHODS="ActiveGeom ActiveOpenSem"
export RUN_TAG="${RUN_TAG:-run_0}"
export GPU0="${GPU0:-0}"
export GPU1="${GPU1:-1}"
export PARALLEL_TRAIN="${PARALLEL_TRAIN:-2}"
export PARALLEL_VALIDATE="${PARALLEL_VALIDATE:-2}"
export NUM_ITERS="${NUM_ITERS:-12000}"
export TRAIN_DOWNSCALE="${TRAIN_DOWNSCALE:-0.5}"
export SKIP_DONE="${SKIP_DONE:-1}"
export FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
export SKIP_VALIDATE="${SKIP_VALIDATE:-0}"
export LOG_DIR="${LOG_DIR:-${PROJ_DIR}/logs/lang_field_ymjk}"

mkdir -p "${LOG_DIR}"

exec bash "${SCRIPT_DIR}/clore/run_mp3d_lang_field_pipeline.sh"
