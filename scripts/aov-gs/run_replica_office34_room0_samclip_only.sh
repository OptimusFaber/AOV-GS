#!/usr/bin/env bash
##################################################
# Replica ActiveSem — SAM+CLIP ONLY (no SLAM)
#
# Reads a prepared trajectory:
#   results/Replica/<scene>/ActiveSem/run_0/exploration_path_poses.json
#
# Passive replay (ActiveSemTrajLangPassive) + SAM+CLIP on keyframes.
# Writes immediately into run_0/ (as frames are processed):
#   language_features/{frame:06d}_{s,f}.npy
#   keyframe_poses.json  (at end of run)
#
# do not touch splatam/ or exploration_path_poses.json
# (temporary splatam only under run_0/.samclip_passive/splatam/)
#
# Scenes: office3, office4, room0
# 2 GPUs: cuda:0 and cuda:1 — one job per GPU
#
# Usage (from repo root):
#   bash scripts/aov-gs/run_replica_office34_room0_samclip_only.sh
#
# Single scene:
#   SCENE=office3 GPU=0 bash scripts/aov-gs/run_replica_office34_room0_samclip_only.sh
#
# Background:
#   mkdir -p logs/replica_samclip_office34_room0
#   nohup bash scripts/aov-gs/run_replica_office34_room0_samclip_only.sh \
#     >> logs/replica_samclip_office34_room0/orchestrator.log 2>&1 &
#   echo $! > logs/replica_samclip_office34_room0/orchestrator.pid
#   tail -f logs/replica_samclip_office34_room0/samclip_office4_gpu1.log
#
# Env:
#   GPU0=0 GPU1=1
#   RUN_TAG=run_0
#   SKIP_DONE=1
#   FORCE=0
#   SCENES="office3 office4 room0"
##################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJ_DIR}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
RUN_TAG="${RUN_TAG:-run_0}"
SKIP_DONE="${SKIP_DONE:-1}"
FORCE="${FORCE:-0}"
LOG_DIR="${LOG_DIR:-${PROJ_DIR}/logs/replica_samclip_office34_room0}"

if [[ -n "${SCENES:-}" ]]; then
  # shellcheck disable=SC2206
  SCENE_LIST=(${SCENES})
elif [[ -n "${SCENE:-}" ]]; then
  SCENE_LIST=("${SCENE}")
else
  SCENE_LIST=(office3 office4 room0)
fi

mkdir -p "${LOG_DIR}"

set +u
if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /opt/conda/etc/profile.d/conda.sh
  conda activate aov-gs 2>/dev/null || true
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  conda activate aov-gs 2>/dev/null || true
fi
set -u

if [[ -f /.dockerenv ]]; then
  bash "${PROJ_DIR}/docker/ensure_habitat_egl.sh" 2>/dev/null || true
fi

if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -x "${CONDA_PREFIX}/bin/python" ]]; then
  PY="${CONDA_PREFIX}/bin/python"
elif [[ -x "${HOME}/anaconda3/envs/aov-gs/bin/python" ]]; then
  PY="${HOME}/anaconda3/envs/aov-gs/bin/python"
elif [[ -x "${HOME}/anaconda3/envs/active-gs/bin/python" ]]; then
  PY="${HOME}/anaconda3/envs/active-gs/bin/python"
elif [[ -x "${HOME}/anaconda3/envs/active-sgm/bin/python" ]]; then
  PY="${HOME}/anaconda3/envs/active-sgm/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

export PYTHONPATH="${PROJ_DIR}:${PROJ_DIR}/third_parties/splatam:${PYTHONPATH:-}"

log() { echo "[$(date -Is)] $*"; }

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

_preflight() {
  local scene="$1"
  local cfg="${PROJ_DIR}/configs/Replica/${scene}/ActiveSemTrajLangPassive.py"
  local rd
  rd="$(_run_dir "${scene}")"

  if [[ ! -f "${cfg}" ]]; then
    log "ERROR [${scene}]: missing ${cfg}"
    log "  Need ActiveSemTrajLangPassive config (passive replay, NOT ActiveSem.py)."
    log "  Copy from AOV-GS-V2/configs/Replica/${scene}/ActiveSemTrajLangPassive.py"
    return 1
  fi

  if ! _poses_ok "${rd}"; then
    log "ERROR [${scene}]: missing ${rd}/exploration_path_poses.json"
    return 1
  fi

  return 0
}

_run_scene() {
  local scene="$1" gpu="$2"
  local rd work_rd cfg log_file poses traj
  rd="$(_run_dir "${scene}")"
  work_rd="${rd}/.samclip_passive"
  cfg="configs/Replica/${scene}/ActiveSemTrajLangPassive.py"
  log_file="${LOG_DIR}/samclip_${scene}_gpu${gpu}.log"
  poses="${rd}/exploration_path_poses.json"
  traj="${PROJ_DIR}/data/replica_activesem_traj/${scene}/traj.txt"

  _preflight "${scene}" || return 1

  if [[ "${FORCE}" != "1" && "${SKIP_DONE}" == "1" ]] && _samclip_ok "${rd}"; then
    log "[${scene}] SKIP: language_features already present"
    return 0
  fi

  log "[${scene}] export poses → ${traj}"
  local export_flags=()
  [[ "${FORCE}" == "1" ]] && export_flags=(--force)
  "${PY}" scripts/data/export_replica_activesem_traj.py \
    --scene "${scene}" \
    --run-tag "${RUN_TAG}" \
    --out-traj "${traj}" \
    "${export_flags[@]}" >>"${log_file}" 2>&1

  if [[ "${FORCE}" == "1" ]]; then
    rm -rf "${rd}/language_features" "${rd}/keyframe_poses.json"
  fi
  mkdir -p "${rd}/language_features"

  rm -rf "${work_rd}"
  mkdir -p "${work_rd}"
  ln -sfn "${rd}/language_features" "${work_rd}/language_features"

  log "[${scene}] PASSIVE replay + SAM+CLIP on gpu${gpu}"
  log "  cfg  : ${cfg}  (predefined_traj, enable_active_planning=False)"
  log "  poses: ${poses}"
  log "  out  : ${rd}/language_features/  (npy written immediately, splatam → ${work_rd}/)"
  log "  log  : ${log_file}"

  set +e
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PY}" src/main/activesgm.py \
      --cfg "${cfg}" \
      --seed 0 \
      --result_dir "results/Replica/${scene}/ActiveSem/${RUN_TAG}/.samclip_passive" \
      --enable_vis 0 \
      --corrclip 0
  ) >>"${log_file}" 2>&1
  local rc=$?
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    log "[${scene}] FAIL passive replay (rc=${rc}) — tail ${log_file}"
    tail -30 "${log_file}" 2>/dev/null || true
    return "${rc}"
  fi

  if [[ ! -d "${rd}/language_features" ]] || [[ -z "$(ls -A "${rd}/language_features" 2>/dev/null)" ]]; then
    log "[${scene}] FAIL: ${rd}/language_features empty after replay"
    return 1
  fi

  if [[ -f "${work_rd}/keyframe_poses.json" ]]; then
    mv -f "${work_rd}/keyframe_poses.json" "${rd}/keyframe_poses.json"
  fi
  rm -rf "${work_rd}"

  local n
  n=$(find "${rd}/language_features" -maxdepth 1 -name '*_s.npy' | wc -l)
  log "[${scene}] OK: ${n} keyframes → ${rd}/language_features/"
  return 0
}

_run_pool() {
  local fail=0 qi=0
  local -a SLOT_PIDS=("" "")
  local -a SLOT_SCENES=("" "")

  _reap() {
    local i pid rc scene gpu
    for i in 0 1; do
      pid="${SLOT_PIDS[$i]}"
      [[ -n "${pid}" ]] || continue
      kill -0 "${pid}" 2>/dev/null && continue
      set +e
      wait "${pid}"
      rc=$?
      set -e
      scene="${SLOT_SCENES[$i]}"
      gpu="${GPU0}"
      [[ "${i}" -eq 1 ]] && gpu="${GPU1}"
      [[ "${rc}" -eq 0 ]] || { log "FAIL ${scene} gpu${gpu} rc=${rc}"; fail=1; }
      SLOT_PIDS[$i]=""
      SLOT_SCENES[$i]=""
    done
  }

  _start() {
    local slot="$1"
    [[ -z "${SLOT_PIDS[$slot]}" ]] || return 1
    [[ "${qi}" -lt ${#SCENE_LIST[@]} ]] || return 1
    local scene="${SCENE_LIST[$qi]}"
    local gpu="${GPU0}"
    [[ "${slot}" -eq 1 ]] && gpu="${GPU1}"
    qi=$((qi + 1))
    _run_scene "${scene}" "${gpu}" &
    SLOT_PIDS[$slot]=$!
    SLOT_SCENES[$slot]="${scene}"
    log "START ${scene} gpu${gpu} pid=${SLOT_PIDS[$slot]}"
  }

  _running() { [[ -n "${SLOT_PIDS[0]}" || -n "${SLOT_PIDS[1]}" ]]; }

  while [[ "${qi}" -lt ${#SCENE_LIST[@]} ]] || _running; do
    _reap
    _start 0 || true
    _start 1 || true
    _running && sleep 3
  done
  return "${fail}"
}

log "=== Replica SAM+CLIP ONLY (office3, office4, room0) ==="
log "  Scenes : ${SCENE_LIST[*]}"
log "  GPUs   : ${GPU0}, ${GPU1}"
log "  Log dir: ${LOG_DIR}"

FAIL=0
if [[ "${#SCENE_LIST[@]}" -eq 1 ]]; then
  _run_scene "${SCENE_LIST[0]}" "${GPU:-${GPU0}}" || FAIL=1
else
  _run_pool || FAIL=1
fi

if [[ "${FAIL}" -eq 0 ]]; then
  log "Done."
  for scene in "${SCENE_LIST[@]}"; do
    rd="$(_run_dir "${scene}")"
    _samclip_ok "${rd}" && log "  ${scene} → ${rd}/language_features/"
  done
else
  log "Finished with errors."
  exit 1
fi
