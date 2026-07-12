#!/usr/bin/env bash
##################################################
# Replica ActiveSem: SLAM (poses) → SAM+CLIP (language_features)
#
# Two phases (PHASE=all):
#   Phase 1 — ActiveSGM on all scenes (poses), parallel on 2 GPUs:
#     cuda:0 → office3 → room0
#     cuda:1 → office4
#   Phase 2 — SAM+CLIP (only after phase 1 finishes):
#     cuda:0 → office4
#     cuda:1 → sleep 900s → office3 → room0
#
# Usage:
#   bash scripts/aov-gs/run_replica_activesem_poses_samclip_batch.sh
#
# poses only:  PHASE=slam  ...
# SAM+CLIP only (poses already present):  PHASE=samclip  ...
#
# Env:
#   PHASE=all|slam|samclip
#   GPU0=0 GPU1=1
#   GPU1_SAMCLIP_DELAY=900
#   FORCE_RERUN=1  SKIP_DONE=1  CONTINUE_ON_FAIL=1
##################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJ_DIR}"

PHASE="${PHASE:-all}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU1_SAMCLIP_DELAY="${GPU1_SAMCLIP_DELAY:-900}"

SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-run_0}"
SKIP_DONE="${SKIP_DONE:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-1}"
ENABLE_VIS="${ENABLE_VIS:-0}"

LOG_DIR="${LOG_DIR:-${PROJ_DIR}/logs/replica_poses_samclip}"
mkdir -p "${LOG_DIR}"

set +u
if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /opt/conda/etc/profile.d/conda.sh
  conda activate aov-gs 2>/dev/null || true
elif [[ -f /home/optimus/anaconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/optimus/anaconda3/etc/profile.d/conda.sh
  conda activate aov-gs 2>/dev/null || true
fi
set -u

if [[ -f /.dockerenv ]]; then
  bash "${PROJ_DIR}/docker/ensure_habitat_egl.sh" 2>/dev/null || true
fi

if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -x "${CONDA_PREFIX}/bin/python" ]]; then
  PY="${CONDA_PREFIX}/bin/python"
elif [[ -x /home/optimus/anaconda3/envs/aov-gs/bin/python ]]; then
  PY=/home/optimus/anaconda3/envs/aov-gs/bin/python
elif [[ -x /home/optimus/anaconda3/envs/active-gs/bin/python ]]; then
  PY=/home/optimus/anaconda3/envs/active-gs/bin/python
elif [[ -x /home/optimus/anaconda3/envs/active-sgm/bin/python ]]; then
  PY=/home/optimus/anaconda3/envs/active-sgm/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

export PYTHONPATH="${PROJ_DIR}:${PROJ_DIR}/third_parties/splatam:${PYTHONPATH:-}"

log() { echo "[$(date -Is)] $*" >&2; }

_run_dir() {
  echo "${PROJ_DIR}/results/Replica/${1}/ActiveSem/${RUN_TAG}"
}

_poses_ok() {
  local rd="$1" pj="${rd}/exploration_path_poses.json"
  [[ -f "${pj}" ]] || return 1
  "${PY}" - "${pj}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
n = d.get("num_poses", len(d.get("poses_c2w", [])))
sys.exit(0 if n > 0 else 1)
PY
}

_samclip_ok() {
  local rd="$1" lf n_s n_f
  lf="${rd}/language_features"
  [[ -d "${lf}" && -f "${rd}/keyframe_poses.json" ]] || return 1
  n_s=$(find "${lf}" -maxdepth 1 -name '*_s.npy' 2>/dev/null | wc -l)
  n_f=$(find "${lf}" -maxdepth 1 -name '*_f.npy' 2>/dev/null | wc -l)
  [[ "${n_s}" -gt 0 && "${n_s}" -eq "${n_f}" ]]
}

_should_run_slam() {
  local scene="$1"
  local rd
  rd="$(_run_dir "${scene}")"
  if [[ "${FORCE_RERUN}" == "1" ]]; then
    return 0
  fi
  if [[ "${SKIP_DONE}" == "1" ]] && _poses_ok "${rd}"; then
    return 1
  fi
  return 0
}

_should_run_samclip() {
  local scene="$1"
  local rd
  rd="$(_run_dir "${scene}")"
  if ! _poses_ok "${rd}"; then
    return 1
  fi
  if [[ "${FORCE_RERUN}" == "1" ]]; then
    return 0
  fi
  if [[ "${SKIP_DONE}" == "1" ]] && _samclip_ok "${rd}"; then
    return 1
  fi
  return 0
}

_run_activesgm_slam() {
  local scene="$1" gpu="$2"
  local rd cfg log_file
  rd="$(_run_dir "${scene}")"
  cfg="${PROJ_DIR}/configs/Replica/${scene}/ActiveSem.py"
  log_file="${LOG_DIR}/activesgm_${scene}_gpu${gpu}.log"

  if ! _should_run_slam "${scene}"; then
    log "[${scene}] SKIP ActiveSGM: exploration_path_poses.json already present"
    return 0
  fi

  if [[ "${FORCE_RERUN}" == "1" ]]; then
    rm -rf "${rd}"
  fi

  [[ -f "${cfg}" ]] || { log "[${scene}] ERROR: missing ${cfg}"; return 1; }

  log "[${scene}] ActiveSGM SLAM on gpu${gpu} → ${log_file}"
  mkdir -p "${rd}"
  set +e
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export ACTIVESGM_SLAM_DEVICE="cuda:0"
    export ACTIVESGM_SEMANTIC_DEVICE="cuda:0"
    "${PY}" src/main/activesgm.py \
      --cfg "configs/Replica/${scene}/ActiveSem.py" \
      --seed "${SEED}" \
      --result_dir "results/Replica/${scene}/ActiveSem/${RUN_TAG}" \
      --enable_vis "${ENABLE_VIS}" \
      --corrclip 0
  ) >"${log_file}" 2>&1
  local rc=$?
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    if _poses_ok "${rd}"; then
      local n
      n="$("${PY}" - "${rd}/exploration_path_poses.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("num_poses", len(d.get("poses_c2w", []))))
PY
)"
      log "[${scene}] WARN ActiveSGM rc=${rc}, but poses saved (${n}) — OK for SAM+CLIP"
      return 0
    fi
    log "[${scene}] FAIL ActiveSGM (rc=${rc})"
    tail -30 "${log_file}" 2>/dev/null || true
    return "${rc}"
  fi

  if ! _poses_ok "${rd}"; then
    log "[${scene}] FAIL ActiveSGM: exploration_path_poses.json not created"
    return 1
  fi
  local n
  n="$("${PY}" - "${rd}/exploration_path_poses.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("num_poses", len(d.get("poses_c2w", []))))
PY
)"
  log "[${scene}] OK ActiveSGM: exploration_path_poses.json (${n} poses)"
  return 0
}

_run_samclip() {
  local scene="$1" gpu="$2"
  local rd force_flag=0
  rd="$(_run_dir "${scene}")"

  if ! _should_run_samclip "${scene}"; then
    if ! _poses_ok "${rd}"; then
      log "[${scene}] SKIP SAM+CLIP: no exploration_path_poses.json"
      return 1
    fi
    log "[${scene}] SKIP SAM+CLIP: language_features already present"
    return 0
  fi

  [[ "${FORCE_RERUN}" == "1" ]] && force_flag=1

  log "[${scene}] SAM+CLIP on gpu${gpu} (poses → language_features)"
  SCENE="${scene}" \
  GPU="${gpu}" \
  RUN_TAG="${RUN_TAG}" \
  ACTIVESEM_RUN_TAG="${RUN_TAG}" \
  SKIP_DONE="${SKIP_DONE}" \
  FORCE="${force_flag}" \
  EXPORT_FORCE=1 \
  LOG_DIR="${LOG_DIR}" \
  bash "${SCRIPT_DIR}/run_replica_activesem_samclip.sh"
}

_run_job() {
  local kind="$1" scene="$2" gpu="$3"
  case "${kind}" in
    slam) _run_activesgm_slam "${scene}" "${gpu}" ;;
    samclip) _run_samclip "${scene}" "${gpu}" ;;
    sleep)
      log "gpu${gpu}: sleep ${scene}s"
      sleep "${scene}"
      ;;
    *) log "Unknown job kind: ${kind}"; return 1 ;;
  esac
}

_run_gpu_queue() {
  local gpu="$1"
  shift
  local -a jobs=("$@")
  local job fail=0 rc=0
  local kind scene

  for job in "${jobs[@]}"; do
    kind="${job%%:*}"
    scene="${job#*:}"
    set +e
    _run_job "${kind}" "${scene}" "${gpu}"
    rc=$?
    set -e
    if [[ "${rc}" -ne 0 ]]; then
      log "FAIL gpu${gpu} job ${job} (rc=${rc})"
      fail=1
      [[ "${CONTINUE_ON_FAIL}" == "1" ]] || return "${fail}"
    fi
  done
  return "${fail}"
}

_run_phase_parallel() {
  local label="$1"
  local gpu0="$2"
  local gpu1="$3"
  shift 3
  local -a gpu0_jobs=()
  local -a gpu1_jobs=()
  local parsing=0

  for arg in "$@"; do
    if [[ "${arg}" == "--gpu1" ]]; then
      parsing=1
      continue
    fi
    if [[ "${parsing}" -eq 0 ]]; then
      gpu0_jobs+=("${arg}")
    else
      gpu1_jobs+=("${arg}")
    fi
  done

  log "=== ${label} ==="
  log "  gpu${gpu0}: ${gpu0_jobs[*]:-(idle)}"
  log "  gpu${gpu1}: ${gpu1_jobs[*]:-(idle)}"

  local fail=0
  local pids=()

  if [[ "${#gpu0_jobs[@]}" -gt 0 ]]; then
    _run_gpu_queue "${gpu0}" "${gpu0_jobs[@]}" &
    pids+=($!)
  fi
  if [[ "${#gpu1_jobs[@]}" -gt 0 ]]; then
    _run_gpu_queue "${gpu1}" "${gpu1_jobs[@]}" &
    pids+=($!)
  fi

  local pid rc
  for pid in "${pids[@]}"; do
    set +e
    wait "${pid}"
    rc=$?
    set -e
    [[ "${rc}" -eq 0 ]] || fail=1
  done
  return "${fail}"
}

_run_phase_slam() {
  _run_phase_parallel "Phase 1/2: ActiveSGM (poses)" "${GPU0}" "${GPU1}" \
    slam:office3 slam:room0 --gpu1 slam:office4
}

_run_phase_samclip() {
  local scene rd missing=0
  for scene in office3 office4 room0; do
    rd="$(_run_dir "${scene}")"
    if ! _poses_ok "${rd}"; then
      log "[${scene}] WARN: no poses — SAM+CLIP will be skipped"
      missing=1
    fi
  done
  [[ "${missing}" -eq 0 ]] || log "Phase 2: not all scenes have poses (see above)"

  _run_phase_parallel "Phase 2/2: SAM+CLIP (language_features)" "${GPU0}" "${GPU1}" \
    samclip:office4 --gpu1 "sleep:${GPU1_SAMCLIP_DELAY}" samclip:office3 samclip:room0
}

# ── Main ────────────────────────────────────────────────────────────────
log "=== Replica ActiveSem batch ==="
log "  PHASE  : ${PHASE}"
log "  GPUs   : ${GPU0}, ${GPU1}"
log "  Log dir: ${LOG_DIR}"

FAIL=0

case "${PHASE}" in
  slam)
    _run_phase_slam || FAIL=1
    ;;
  samclip)
    _run_phase_samclip || FAIL=1
    ;;
  all|*)
    _run_phase_slam || FAIL=1
    if [[ "${FAIL}" -eq 0 || "${CONTINUE_ON_FAIL}" == "1" ]]; then
      _run_phase_samclip || FAIL=1
    else
      log "Phase 2 skipped: Phase 1 failed (CONTINUE_ON_FAIL=0)"
    fi
    ;;
esac

if [[ "${FAIL}" -eq 0 ]]; then
  log "Done."
  for scene in office3 room0 office4; do
    rd="$(_run_dir "${scene}")"
    _poses_ok "${rd}" && log "  ${scene} poses → ${rd}/exploration_path_poses.json"
    _samclip_ok "${rd}" && log "  ${scene} lang   → ${rd}/language_features/"
  done
else
  log "Finished with errors."
  exit 1
fi
