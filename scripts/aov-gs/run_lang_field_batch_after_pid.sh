#!/bin/bash
##################################################
# Wait for a running job (e.g. multiseed SLAM), then start lang field batch.
#
# Usage:
#   bash scripts/aov-gs/run_lang_field_batch_after_pid.sh WAIT_PID
#
# Example (after pgrep -af run_multiseed_geom_open_sem_all.sh):
#   cd AOV-GS-V2
#   mkdir -p logs/lang_field_batch
#   nohup bash scripts/aov-gs/run_lang_field_batch_after_pid.sh 2039659 \
#       > logs/lang_field_batch/orchestrator.log 2>&1 &
#
# All env vars from run_lang_field_batch_replica.sh are forwarded (PHASE, GPUS, …).
##################################################

set -eo pipefail

WAIT_PID=${1:?"Usage: $0 WAIT_PID  (PID of run_multiseed_geom_open_sem_all.sh or other blocker)"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_lang_field_batch_replica.sh" "${WAIT_PID}"
