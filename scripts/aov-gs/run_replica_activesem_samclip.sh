#!/usr/bin/env bash
##################################################
# Replica ActiveSem → SAM+CLIP → language_features
#
# Job (exactly one):
#   1) Take exploration poses from
#        results/Replica/<SCENE>/ActiveSem/run_0/exploration_path_poses.json
#   2) Replay the trajectory in Habitat (passive SplaTAM replay)
#   3) On SplaTAM keyframes — SAM+CLIP
#   4) Write into the same run_0/:
#        language_features/{frame:06d}_s.npy
#        language_features/{frame:06d}_f.npy
#        keyframe_poses.json
#
# Do not touch: splatam/, exploration_path_poses.json (SemSplaTAM ActiveSem backup).
# Next lang-field: bash scripts/aov-gs/03_train_gaussian_lang_field.sh ...
#
# Usage (from AOV-GS-V2 root):
#   bash scripts/aov-gs/run_replica_activesem_samclip.sh
#
# Single scene:
#   SCENE=office0 bash scripts/aov-gs/run_replica_activesem_samclip.sh
#
# SLAM + SAM+CLIP for room0/office3/office4 (if poses missing):
#   bash scripts/aov-gs/run_replica_activesem_poses_samclip_batch.sh
#
# Background + monitoring:
#   mkdir -p logs/replica_activesem_samclip
#   nohup bash scripts/aov-gs/run_replica_activesem_samclip.sh \
#     >> logs/replica_activesem_samclip/orchestrator.log 2>&1 &
#   tail -f logs/replica_activesem_samclip/orchestrator.log
#   tail -f logs/replica_activesem_samclip/office0.log
#
# Env:
#   SCENES="office0 office2 ..."   default: all 8 Replica
#   SCENE=office0                  single scene
#   GPU=0
#   RUN_TAG=run_0
#   ACTIVESEM_RUN_TAG=run_0
#   SKIP_DONE=1
#   FORCE=0                        rebuild language_features
#   LOG_BASENAME=samclip_office3_gpu0   (optional per-scene log stem)
##################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJ_DIR}"

GPU="${GPU:-0}"
GPU0="${GPU0:-${GPU}}"
GPU1="${GPU1:-1}"
RUN_TAG="${RUN_TAG:-run_0}"
ACTIVESEM_RUN_TAG="${ACTIVESEM_RUN_TAG:-run_0}"
SKIP_DONE="${SKIP_DONE:-1}"
FORCE="${FORCE:-0}"
EXPORT_FORCE="${EXPORT_FORCE:-0}"
SEED="${SEED:-0}"
SLAM_CFG="${SLAM_CFG:-ActiveSemTrajLangPassive}"
WORK_SUBDIR="${WORK_SUBDIR:-.samclip_passive}"

DEFAULT_SCENES=(office0 office2 office3 office4 room0 room1 room2)
if [[ -n "${SCENES:-}" ]]; then
  # shellcheck disable=SC2206
  SCENE_LIST=(${SCENES})
elif [[ -n "${SCENE:-}" ]]; then
  SCENE_LIST=("${SCENE}")
else
  SCENE_LIST=("${DEFAULT_SCENES[@]}")
fi

LOG_DIR="${LOG_DIR:-${PROJ_DIR}/logs/replica_activesem_samclip}"
mkdir -p "${LOG_DIR}"

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

log() { echo "[$(date -Is)] $*"; }

_activesem_dir() {
  echo "${PROJ_DIR}/results/Replica/${1}/ActiveSem/${RUN_TAG}"
}

_poses_json() {
  echo "${PROJ_DIR}/results/Replica/${1}/ActiveSem/${ACTIVESEM_RUN_TAG}/exploration_path_poses.json"
}

_traj_txt() {
  echo "${PROJ_DIR}/data/replica_activesem_traj/${1}/traj.txt"
}

_samclip_done() {
  local scene="$1"
  local rd lf n_s n_f
  rd="$(_activesem_dir "${scene}")"
  lf="${rd}/language_features"
  [[ -d "${lf}" && -f "${rd}/keyframe_poses.json" ]] || return 1
  n_s=$(find "${lf}" -maxdepth 1 -name '*_s.npy' 2>/dev/null | wc -l)
  n_f=$(find "${lf}" -maxdepth 1 -name '*_f.npy' 2>/dev/null | wc -l)
  [[ "${n_s}" -gt 0 && "${n_s}" -eq "${n_f}" ]]
}

_export_traj() {
  local scene="$1"
  local poses traj force_flag=()
  poses="$(_poses_json "${scene}")"
  traj="$(_traj_txt "${scene}")"
  [[ "${EXPORT_FORCE}" == "1" ]] && force_flag=(--force)

  if [[ ! -f "${poses}" ]]; then
    log "ERROR ${scene}: missing ${poses}"
    log "  ActiveSem SLAM required first (ActiveSGM scene exploration)."
    return 1
  fi

  log "${scene}: export poses → ${traj}"
  "${PY}" scripts/data/export_replica_activesem_traj.py \
    --scene "${scene}" \
    --run-tag "${ACTIVESEM_RUN_TAG}" \
    --out-traj "${traj}" \
    "${force_flag[@]}"
}

_run_scene() {
  local scene="$1"
  local gpu="${2:-${GPU0}}"
  local rd work_rd cfg log_file poses
  rd="$(_activesem_dir "${scene}")"
  work_rd="${rd}/${WORK_SUBDIR}"
  cfg="configs/Replica/${scene}/${SLAM_CFG}.py"
  log_file="${LOG_DIR}/${LOG_BASENAME:-${scene}}.log"
  poses="$(_poses_json "${scene}")"

  if [[ "${FORCE}" != "1" && "${SKIP_DONE}" == "1" ]] && _samclip_done "${scene}"; then
    log "SKIP ${scene}: language_features already present"
    return 0
  fi

  if [[ ! -f "${cfg}" ]]; then
    log "ERROR ${scene}: missing ${cfg}"
    return 1
  fi

  _export_traj "${scene}" || return 1

  if [[ "${FORCE}" == "1" ]]; then
    rm -rf "${rd}/language_features" "${rd}/keyframe_poses.json"
  fi
  mkdir -p "${rd}/language_features"

  rm -rf "${work_rd}"
  mkdir -p "${work_rd}"
  ln -sfn "${rd}/language_features" "${work_rd}/language_features"

  log "${scene}: passive replay + SAM+CLIP → ${log_file}"
  log "  poses : ${poses}"
  log "  out   : ${rd}/language_features/  (npy written immediately, splatam → ${work_rd}/)"

  set +e
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PY}" src/main/activesgm.py \
      --cfg "${cfg}" \
      --seed "${SEED}" \
      --result_dir "results/Replica/${scene}/ActiveSem/${RUN_TAG}/${WORK_SUBDIR}" \
      --enable_vis 0 \
      --corrclip 0
  ) >"${log_file}" 2>&1
  local rc=$?
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    log "FAIL ${scene} (rc=${rc}) — tail ${log_file}"
    tail -40 "${log_file}" 2>/dev/null || true
    return "${rc}"
  fi

  if [[ ! -d "${rd}/language_features" ]] || [[ -z "$(ls -A "${rd}/language_features" 2>/dev/null)" ]]; then
    log "FAIL ${scene}: ${rd}/language_features empty after replay"
    return 1
  fi

  if [[ -f "${work_rd}/keyframe_poses.json" ]]; then
    mv -f "${work_rd}/keyframe_poses.json" "${rd}/keyframe_poses.json"
  fi
  rm -rf "${work_rd}"

  if _samclip_done "${scene}"; then
    local n
    n=$(find "${rd}/language_features" -maxdepth 1 -name '*_s.npy' | wc -l)
    log "OK ${scene}: ${n} keyframes → ${rd}/language_features/"
    return 0
  fi

  log "FAIL ${scene}: audit language_features failed"
  return 1
}

_run_scene_pool() {
  local fail=0 qi=0
  local -a SLOT_PIDS=("" "")
  local -a SLOT_SCENES=("" "")

  _reap_samclip() {
    local i pid rc scene gpu
    for i in 0 1; do
      pid="${SLOT_PIDS[$i]}"
      [[ -n "${pid}" ]] || continue
      if kill -0 "${pid}" 2>/dev/null; then
        continue
      fi
      set +e
      wait "${pid}"
      rc=$?
      set -e
      scene="${SLOT_SCENES[$i]}"
      gpu="${GPU0}"
      [[ "${i}" -eq 1 ]] && gpu="${GPU1:-1}"
      if [[ "${rc}" -ne 0 ]]; then
        log "FAIL ${scene} gpu${gpu} (rc=${rc})"
        fail=1
      fi
      SLOT_PIDS[$i]=""
      SLOT_SCENES[$i]=""
    done
  }

  _start_samclip() {
    local slot="$1"
    [[ -z "${SLOT_PIDS[$slot]}" ]] || return 1
    [[ "${qi}" -lt ${#SCENE_LIST[@]} ]] || return 1
    local scene="${SCENE_LIST[$qi]}"
    local gpu="${GPU0}"
    [[ "${slot}" -eq 1 ]] && gpu="${GPU1:-1}"
    qi=$((qi + 1))
    _run_scene "${scene}" "${gpu}" &
    SLOT_PIDS[$slot]=$!
    SLOT_SCENES[$slot]="${scene}"
    log "START ${scene} on cuda:${gpu} pid=${SLOT_PIDS[$slot]}"
  }

  _any_samclip() { [[ -n "${SLOT_PIDS[0]}" || -n "${SLOT_PIDS[1]}" ]]; }

  while [[ "${qi}" -lt ${#SCENE_LIST[@]} ]] || _any_samclip; do
    _reap_samclip
    _start_samclip 0 || true
    _start_samclip 1 || true
    _any_samclip && sleep 3
  done
  return "${fail}"
}

log "=== Replica ActiveSem traj → SAM+CLIP → language_features ==="
log "  Scenes : ${SCENE_LIST[*]}"
log "  GPU pool : cuda:${GPU0}, cuda:${GPU1} (multi-scene parallel)"
log "  Log dir: ${LOG_DIR}"

FAIL=0
if [[ "${#SCENE_LIST[@]}" -eq 1 ]]; then
  _run_scene "${SCENE_LIST[0]}" "${GPU:-${GPU0}}" || FAIL=1
else
  _run_scene_pool || FAIL=1
fi

if [[ "${FAIL}" -eq 0 ]]; then
  log "Done. Next: train lang field on results/Replica/<scene>/ActiveSem/run_0/"
else
  log "Finished with errors."
  exit 1
fi
