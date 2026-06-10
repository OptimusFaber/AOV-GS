#!/bin/bash
##################################################
# Batch validate lang_field_traj for Replica ActiveOpenSem runs.
#
# Default: all scenes except office0 and room0; run_0 + run_1/2/3.
# Waits until every target (scene, run) has SLAM + lang_field s/m/l before validating.
# Parallelism: VALIDATE_SLOTS_PER_GPU jobs per GPU (default 4), dynamic pool on 2 GPUs.
#
# Usage:
#   bash scripts/aov-gs/run_lang_field_validate_batch.sh
#
#   nohup bash scripts/aov-gs/run_lang_field_validate_batch.sh \
#       > logs/lang_field_batch/validate_batch.log 2>&1 &
#
# Env:
#   GPUS="0 1"
#   VALIDATE_SLOTS_PER_GPU=4
#   EXCLUDE_SCENES="office0 room0"
#   SCENES="office1 office2 ..."   override scene list
#   RUN_TAGS="run_0 run_1 run_2 run_3"
#   WAIT_FOR_ALL=1             wait until all targets have lang_field s/m/l (default)
#   WAIT_POLL_SEC=120          poll while waiting for training to finish
#   SKIP_DONE=1                skip validate if miou_summary.txt exists
#   EXP=ActiveOpenSem
##################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJ_DIR}"

GPUS="${GPUS:-0 1}"
VALIDATE_SLOTS_PER_GPU="${VALIDATE_SLOTS_PER_GPU:-4}"
EXCLUDE_SCENES="${EXCLUDE_SCENES:-office0 room0}"
EXP="${EXP:-ActiveOpenSem}"
RUN_TAGS="${RUN_TAGS:-run_0 run_1 run_2 run_3}"
SKIP_DONE="${SKIP_DONE:-1}"
WAIT_FOR_ALL="${WAIT_FOR_ALL:-1}"
WAIT_POLL_SEC="${WAIT_POLL_SEC:-120}"
DRY_RUN="${DRY_RUN:-0}"

LOG_DIR="${LOG_DIR:-${PROJ_DIR}/logs/lang_field_batch}"
mkdir -p "${LOG_DIR}"

if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PY="${CONDA_PREFIX}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

# shellcheck disable=SC2206
GPU_ARR=($GPUS)
NUM_GPUS=${#GPU_ARR[@]}
# shellcheck disable=SC2206
EXCLUDE_ARR=($EXCLUDE_SCENES)
# shellcheck disable=SC2206
RUN_TAG_ARR=($RUN_TAGS)

DEFAULT_SCENES=(office0 office1 office2 office3 office4 room0 room1 room2)
if [ -n "${SCENES:-}" ]; then
  # shellcheck disable=SC2206
  SCENE_LIST=($SCENES)
else
  SCENE_LIST=("${DEFAULT_SCENES[@]}")
fi

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$( _ts )] $*"; }

_is_excluded() {
  local scene="$1"
  local s
  for s in "${EXCLUDE_ARR[@]}"; do
    [ "${scene}" = "${s}" ] && return 0
  done
  return 1
}

_lang_field_ready() {
  local rd="$1"
  local k="${CODEBOOK_SIZE:-64}"
  local l="${VQ_LAYER_NUM:-1}"
  [ -f "${rd}/lang_field_sk${k}_l${l}/lang_field.pt" ] \
    && [ -f "${rd}/lang_field_mk${k}_l${l}/lang_field.pt" ] \
    && [ -f "${rd}/lang_field_lk${k}_l${l}/lang_field.pt" ]
}

_slam_ready() {
  local rd="$1"
  [ -f "${rd}/keyframe_poses.json" ] \
    && [ -d "${rd}/language_features" ] \
    && { [ -f "${rd}/splatam/final/params.npz" ] || [ -f "${rd}/splatam/final/params0.npz" ]; }
}

_result_dir() {
  local scene="$1"
  local run_tag="$2"
  echo "${PROJ_DIR}/results/Replica/${scene}/${EXP}/${run_tag}"
}

# All (scene, run) pairs we require before validation (excludes office0, room0).
_build_required_targets() {
  REQUIRED_TARGETS=()
  local scene run_tag
  for scene in "${SCENE_LIST[@]}"; do
    _is_excluded "${scene}" && continue
    for run_tag in "${RUN_TAG_ARR[@]}"; do
      REQUIRED_TARGETS+=("${scene}/${run_tag}")
    done
  done
}

_missing_lang_fields() {
  MISSING_LANG=()
  local job scene run_tag rd
  for job in "${REQUIRED_TARGETS[@]}"; do
    scene="${job%%/*}"
    run_tag="${job#*/}"
    rd="$(_result_dir "${scene}" "${run_tag}")"
    if ! _slam_ready "${rd}"; then
      MISSING_LANG+=("${job} (no SLAM/language_features yet)")
    elif ! _lang_field_ready "${rd}"; then
      MISSING_LANG+=("${job} (lang_field s/m/l incomplete)")
    fi
  done
}

_wait_all_lang_fields() {
  _build_required_targets
  log "Waiting for lang_field s/m/l on ${#REQUIRED_TARGETS[@]} targets (exclude: ${EXCLUDE_SCENES})..."
  log "  Runs: ${RUN_TAGS}"
  log "  Poll : ${WAIT_POLL_SEC}s"

  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi

  while true; do
    _missing_lang_fields
    if [ "${#MISSING_LANG[@]}" -eq 0 ]; then
      log "All required lang_field checkpoints ready — starting validation."
      break
    fi
    log "Still waiting (${#MISSING_LANG[@]} pending): ${MISSING_LANG[*]}"
    sleep "${WAIT_POLL_SEC}"
  done
}

CODEBOOK_SIZE="${CODEBOOK_SIZE:-64}"
VQ_LAYER_NUM="${VQ_LAYER_NUM:-1}"

build_queue() {
  QUEUE=()
  local scene run_tag rd traj out summary job
  _build_required_targets
  for job in "${REQUIRED_TARGETS[@]}"; do
    scene="${job%%/*}"
    run_tag="${job#*/}"
    rd="$(_result_dir "${scene}" "${run_tag}")"
    traj="${PROJ_DIR}/data/replica_sim_nvs/${scene}/traj.txt"
    out="${rd}/lang_field_traj_eval"
    summary="${out}/miou_summary.txt"

    [ -f "${traj}" ] || { log "WARN ${job}: missing ${traj}"; continue; }

    if [ "${SKIP_DONE}" = "1" ] && [ -f "${summary}" ]; then
      log "SKIP ${job}: ${summary} exists"
      continue
    fi

    QUEUE+=("${scene}|${run_tag}|${rd}|${traj}|${out}")
  done
}

_validate_one() {
  local gpu="$1"
  local scene="$2"
  local run_tag="$3"
  local rd="$4"
  local traj="$5"
  local out="$6"
  local log_file="${LOG_DIR}/validate_${scene}_${run_tag}.log"

  log "VALIDATE ${scene}/${run_tag} on GPU ${gpu} → ${out}"
  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PY}" scripts/validate_lang_field_traj.py \
      --scene "${scene}" \
      --result_dir "${rd}" \
      --traj_txt "${traj}" \
      --out_dir "${out}" \
      --device cuda:0
  ) > "${log_file}" 2>&1
}

_run_validate_pool() {
  local num_jobs=${#QUEUE[@]}
  if [ "${num_jobs}" -eq 0 ]; then
    log "No validation jobs in queue."
    return 0
  fi

  log "Validation pool: ${num_jobs} jobs, ${VALIDATE_SLOTS_PER_GPU} slots/GPU, GPUs=${GPU_ARR[*]}"

  if [ "${DRY_RUN}" = "1" ]; then
    local entry
    for entry in "${QUEUE[@]}"; do
      log "[DRY_RUN] ${entry%%|*}"
    done
    return 0
  fi

  local pending_idx=0
  declare -a SLOT_PID=()
  declare -a SLOT_JOB=()
  local gi slot running max_slots

  max_slots=$((NUM_GPUS * VALIDATE_SLOTS_PER_GPU))
  for ((slot = 0; slot < max_slots; slot++)); do
    SLOT_PID[slot]=""
    SLOT_JOB[slot]=""
  done

  _launch_slot() {
    local slot=$1
    local gi=$((slot % NUM_GPUS))
    local gpu="${GPU_ARR[gi]}"
    local entry="${QUEUE[$pending_idx]}"
    IFS='|' read -r scene run_tag rd traj out <<< "${entry}"
    pending_idx=$((pending_idx + 1))
    (
      _validate_one "${gpu}" "${scene}" "${run_tag}" "${rd}" "${traj}" "${out}"
    ) &
    SLOT_PID[slot]=$!
    SLOT_JOB[slot]="${scene}/${run_tag}@gpu${gpu}"
    log "LAUNCH validate ${scene}/${run_tag} slot=${slot} gpu=${gpu} (${pending_idx}/${num_jobs})"
  }

  while true; do
    for ((slot = 0; slot < max_slots; slot++)); do
      if [ -z "${SLOT_PID[slot]}" ] && [ "${pending_idx}" -lt "${num_jobs}" ]; then
        _launch_slot "${slot}"
      fi
    done

    running=0
    for ((slot = 0; slot < max_slots; slot++)); do
      [ -n "${SLOT_PID[slot]}" ] && running=1 && break
    done
    [ "${running}" -eq 0 ] && break

    wait -n
    for ((slot = 0; slot < max_slots; slot++)); do
      if [ -n "${SLOT_PID[slot]}" ] && ! kill -0 "${SLOT_PID[slot]}" 2>/dev/null; then
        wait "${SLOT_PID[slot]}" || true
        log "DONE validate ${SLOT_JOB[slot]}"
        SLOT_PID[slot]=""
        SLOT_JOB[slot]=""
      fi
    done
  done
}

log "Lang field validate batch"
log "  Python   : ${PY}"
log "  Exclude  : ${EXCLUDE_SCENES}"
log "  Runs     : ${RUN_TAGS}"
log "  Wait all : ${WAIT_FOR_ALL}"
log "  Slots/GPU: ${VALIDATE_SLOTS_PER_GPU}"

if [ "${WAIT_FOR_ALL}" = "1" ]; then
  _wait_all_lang_fields
fi

build_queue
log "  Queue    : ${#QUEUE[@]} validation jobs"

_run_validate_pool
log "Validation batch finished."
