#!/bin/bash
##################################################
# 1) Inject office2/run_0 on GPU 1 when batch hits room2 + GPU1 free.
# 2) Wait for batch orchestrator PID to exit.
# 3) Run lang_field_traj validation (exclude office0, room0), 4 jobs/GPU.
#
# Usage:
#   BATCH_PID=$(pgrep -f run_lang_field_batch_replica.sh | head -1)
#   nohup bash scripts/aov-gs/run_lang_field_followup_office2_validate.sh "$BATCH_PID" \
#       > logs/lang_field_batch/followup_office2_validate.log 2>&1 &
#
# Skip inject if office2 already trained:
#   INJECT_OFFICE2=0 bash scripts/aov-gs/run_lang_field_followup_office2_validate.sh "$BATCH_PID"
##################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJ_DIR}"

BATCH_PID="${1:-}"
INJECT_OFFICE2="${INJECT_OFFICE2:-1}"
ORCH_LOG="${ORCH_LOG:-${PROJ_DIR}/logs/lang_field_batch/orchestrator.log}"

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$( _ts )] $*"; }

if [ "${INJECT_OFFICE2}" = "1" ]; then
  log "Step 1/3: inject office2/run_0 on GPU 1"
  ORCH_LOG="${ORCH_LOG}" bash "${SCRIPT_DIR}/run_inject_office2_when_gpu1_free.sh"
  log "Step 1/3 done"
else
  log "Step 1/3: inject office2 skipped (INJECT_OFFICE2=0)"
fi

if [ -n "${BATCH_PID}" ]; then
  log "Step 2/3: waiting for batch PID ${BATCH_PID}"
  while kill -0 "${BATCH_PID}" 2>/dev/null; do
    sleep 60
  done
  log "Step 2/3: batch PID ${BATCH_PID} finished"
else
  log "Step 2/3: no BATCH_PID — skip wait (assume batch already done)"
fi

log "Step 3/3: validation batch (exclude office0 room0; wait for all lang_field s/m/l, then 4 slots/GPU)"
bash "${SCRIPT_DIR}/run_lang_field_validate_batch.sh"
log "All follow-up steps finished."
