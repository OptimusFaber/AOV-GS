#!/bin/bash
##################################################
# While run_lang_field_batch_replica.sh is running:
#   wait until batch reached room2/run_0 AND GPU 1 is idle,
#   then train office2/run_0 (s+m+l parallel) on GPU 1.
#
# Usage:
#   ORCH_LOG=logs/lang_field_batch/orchestrator.log \
#   nohup bash scripts/aov-gs/run_inject_office2_when_gpu1_free.sh \
#       > logs/lang_field_batch/inject_office2.log 2>&1 &
##################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJ_DIR}"

ORCH_LOG="${ORCH_LOG:-${PROJ_DIR}/logs/lang_field_batch/orchestrator.log}"
GPU_ID="${GPU_ID:-1}"
SCENE="${SCENE:-office2}"
RUN_TAG="${RUN_TAG:-run_0}"
EXP="${EXP:-ActiveOpenSem}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-64}"
NUM_ITERS="${NUM_ITERS:-12000}"
VQ_LAYER_NUM="${VQ_LAYER_NUM:-1}"
TOPK="${TOPK:-4}"
RENDER_CHECKPOINT="${RENDER_CHECKPOINT:-auto}"
TRAIN_DOWNSCALE="${TRAIN_DOWNSCALE:-0.5}"
LOG_EVERY="${LOG_EVERY:-500}"
POLL_SEC="${POLL_SEC:-30}"

LOG_DIR="${LOG_DIR:-${PROJ_DIR}/logs/lang_field_batch}"
mkdir -p "${LOG_DIR}"

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$( _ts )] $*"; }

RESULT_DIR="${PROJ_DIR}/results/Replica/${SCENE}/${EXP}/${RUN_TAG}"
JOB_ID="${SCENE}/${RUN_TAG}"

log "Watcher: inject ${JOB_ID} on GPU ${GPU_ID} when room2 started + GPU${GPU_ID} free"
log "Orchestrator log: ${ORCH_LOG}"

room2_started=0
gpu_free=0

while [ "${room2_started}" -eq 0 ] || [ "${gpu_free}" -eq 0 ]; do
  if [ -f "${ORCH_LOG}" ]; then
    grep -qE "LAUNCH room2/run_0|TRAIN room2/run_0" "${ORCH_LOG}" 2>/dev/null && room2_started=1
    grep -qE "DONE room1/run_0 on GPU ${GPU_ID}" "${ORCH_LOG}" 2>/dev/null && gpu_free=1
  fi
  if [ "${room2_started}" -eq 0 ] || [ "${gpu_free}" -eq 0 ]; then
    log "wait room2=${room2_started} gpu${GPU_ID}_free=${gpu_free} (poll ${POLL_SEC}s)"
    sleep "${POLL_SEC}"
  fi
done

log "Conditions met — training ${JOB_ID} on GPU ${GPU_ID} (s+m+l parallel)"

for level in s m l; do
  log_file="${LOG_DIR}/train_${SCENE}_${RUN_TAG}_${level}.log"
  (
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    bash "${SCRIPT_DIR}/03_train_gaussian_lang_field.sh" \
      "${RESULT_DIR}" \
      "${CODEBOOK_SIZE}" \
      "${level}" \
      "${NUM_ITERS}" \
      "cuda:0" \
      "${VQ_LAYER_NUM}" \
      "${TOPK}" \
      "${RENDER_CHECKPOINT}" \
      "${TRAIN_DOWNSCALE}" \
      "${LOG_EVERY}"
  ) > "${log_file}" 2>&1 &
done

wait
log "Done: ${JOB_ID} on GPU ${GPU_ID}"
