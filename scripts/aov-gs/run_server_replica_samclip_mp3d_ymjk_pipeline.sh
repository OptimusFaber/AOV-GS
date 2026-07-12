#!/usr/bin/env bash
##################################################
# Server pipeline (2×GPU):
#   Phase 1 — Replica SAM+CLIP: office3, office4, room0
#   Phase 2 — MP3D YmJkqBEsHnH lang field: ActiveGeom + ActiveOpenSem
#
# Usage:
#   bash scripts/aov-gs/run_server_replica_samclip_mp3d_ymjk_pipeline.sh
#
# SAM+CLIP only:
#   PHASE=samclip bash scripts/aov-gs/run_server_replica_samclip_mp3d_ymjk_pipeline.sh
#
# lang field only (Replica SAM+CLIP already done):
#   PHASE=langfield bash scripts/aov-gs/run_server_replica_samclip_mp3d_ymjk_pipeline.sh
#
# Background:
#   mkdir -p logs/server_pipeline
#   nohup bash scripts/aov-gs/run_server_replica_samclip_mp3d_ymjk_pipeline.sh \
#     >> logs/server_pipeline/orchestrator.log 2>&1 &
#   echo $! > logs/server_pipeline/orchestrator.pid
#   tail -f logs/server_pipeline/orchestrator.log
#
# Env:
#   PHASE=all|samclip|langfield
#   GPU0=0 GPU1=1
##################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJ_DIR}"

PHASE="${PHASE:-all}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
LOG_DIR="${LOG_DIR:-${PROJ_DIR}/logs/server_pipeline}"
mkdir -p "${LOG_DIR}"

log() { echo "[$(date -Is)] $*"; }

FAIL=0

log "=== Server pipeline ==="
log "  PHASE: ${PHASE}"
log "  GPUs : ${GPU0}, ${GPU1}"

if [[ "${PHASE}" == "all" || "${PHASE}" == "samclip" ]]; then
  log "======== Phase 1/2: Replica SAM+CLIP (office3, office4, room0) ========"
  GPU0="${GPU0}" GPU1="${GPU1}" LOG_DIR="${LOG_DIR}/replica_samclip" \
    bash "${SCRIPT_DIR}/run_replica_office34_room0_samclip_only.sh" || FAIL=1
fi

if [[ "${FAIL}" -eq 0 && ( "${PHASE}" == "all" || "${PHASE}" == "langfield" ) ]]; then
  log "======== Phase 2/2: MP3D YmJkqBEsHnH lang field ========"
  GPU0="${GPU0}" GPU1="${GPU1}" LOG_DIR="${LOG_DIR}/lang_field_ymjk" \
    bash "${SCRIPT_DIR}/run_mp3d_ymjk_lang_field.sh" || FAIL=1
elif [[ "${FAIL}" -ne 0 && "${PHASE}" == "all" ]]; then
  log "Phase 2 skipped: Phase 1 failed"
fi

if [[ "${FAIL}" -eq 0 ]]; then
  log "Pipeline finished successfully."
else
  log "Pipeline finished with errors."
  exit 1
fi
