#!/usr/bin/env bash
##################################################
# Full server pipeline (2×GPU):
#
#   Phase 1 — Replica SAM+CLIP (office3, office4, room0)
#             language_features/*.npy are written IMMEDIATELY into run_0/
#   Phase 2 — MP3D YmJkqBEsHnH lang field (ActiveGeom + ActiveOpenSem, s/m/l)
#
# Usage:
#   bash scripts/aov-gs/run_server_full_pipeline.sh
#
# SAM+CLIP only:
#   PHASE=samclip bash scripts/aov-gs/run_server_full_pipeline.sh
#
# MP3D lang field only (Replica already done):
#   PHASE=langfield bash scripts/aov-gs/run_server_full_pipeline.sh
#
# MP3D in parallel with Replica (YmJkqBEsHnH language_features already present):
#   START_MP3D_PARALLEL=1 bash scripts/aov-gs/run_server_full_pipeline.sh
#
# Background:
#   mkdir -p logs/server_full_pipeline
#   nohup bash scripts/aov-gs/run_server_full_pipeline.sh \
#     >> logs/server_full_pipeline/orchestrator.log 2>&1 &
#   echo $! > logs/server_full_pipeline/orchestrator.pid
#   tail -f logs/server_full_pipeline/orchestrator.log
#
# Env:
#   PHASE=all|samclip|langfield
#   START_MP3D_PARALLEL=0|1
#   GPU0=0 GPU1=1
##################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJ_DIR}"

PHASE="${PHASE:-all}"
START_MP3D_PARALLEL="${START_MP3D_PARALLEL:-0}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
LOG_DIR="${LOG_DIR:-${PROJ_DIR}/logs/server_full_pipeline}"
mkdir -p "${LOG_DIR}"

log() { echo "[$(date -Is)] $*"; }

_run_mp3d_langfield() {
  GPU0="${GPU0}" GPU1="${GPU1}" LOG_DIR="${LOG_DIR}/lang_field_ymjk" \
    bash "${SCRIPT_DIR}/run_mp3d_ymjk_lang_field.sh"
}

FAIL=0
MP3D_PID=""

log "=== Server full pipeline ==="
log "  PHASE              : ${PHASE}"
log "  START_MP3D_PARALLEL: ${START_MP3D_PARALLEL}"
log "  GPUs               : ${GPU0}, ${GPU1}"

if [[ "${START_MP3D_PARALLEL}" == "1" && ( "${PHASE}" == "all" || "${PHASE}" == "langfield" ) ]]; then
  log "======== MP3D YmJkqBEsHnH lang field (parallel, background) ========"
  _run_mp3d_langfield >>"${LOG_DIR}/lang_field_ymjk_parallel.log" 2>&1 &
  MP3D_PID=$!
  log "  MP3D pid=${MP3D_PID} → ${LOG_DIR}/lang_field_ymjk_parallel.log"
fi

if [[ "${PHASE}" == "all" || "${PHASE}" == "samclip" ]]; then
  log "======== Phase 1: Replica SAM+CLIP (office3, office4, room0) ========"
  GPU0="${GPU0}" GPU1="${GPU1}" LOG_DIR="${LOG_DIR}/replica_samclip" \
    bash "${SCRIPT_DIR}/run_replica_office34_room0_samclip_only.sh" || FAIL=1
fi

if [[ -n "${MP3D_PID}" ]]; then
  log "Waiting for parallel MP3D lang field (pid=${MP3D_PID})..."
  set +e
  wait "${MP3D_PID}"
  MP3D_RC=$?
  set -e
  [[ "${MP3D_RC}" -eq 0 ]] || { log "FAIL MP3D parallel (rc=${MP3D_RC})"; FAIL=1; }
elif [[ "${FAIL}" -eq 0 && ( "${PHASE}" == "all" || "${PHASE}" == "langfield" ) ]]; then
  log "======== Phase 2: MP3D YmJkqBEsHnH lang field ========"
  _run_mp3d_langfield || FAIL=1
elif [[ "${FAIL}" -ne 0 && "${PHASE}" == "all" && "${START_MP3D_PARALLEL}" != "1" ]]; then
  log "Phase 2 skipped: Phase 1 failed"
fi

if [[ "${FAIL}" -eq 0 ]]; then
  log "Pipeline finished successfully."
else
  log "Pipeline finished with errors."
  exit 1
fi
