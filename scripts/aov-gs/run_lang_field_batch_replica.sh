#!/bin/bash
##################################################
# Batch train LangSplatV2 language field on Replica ActiveOpenSem runs.
#
# Parallelism (2× 32GB GPU):
#   - Dynamic pool: freed GPU picks next scene from the queue
#   - On each GPU per scene: s, m, l in parallel (3 processes)
#
# Phase order (PHASE=all):
#   1) All run_0 jobs first (prereqs required upfront)
#   2) Then run_1/run_2/run_3 — wait for SLAM outputs per job (post-factum)
#
# Phase 1 — run_0: office1, office3, office4, room1, room2
# Phase 2 — run_1/2/3: office0, office1, office3, office4, room0, room1, room2
#
# Usage:
#   bash scripts/aov-gs/run_lang_field_batch_replica.sh [WAIT_PID]
#
# Env:
#   WAIT_PID=...             wait for PID before start (poll 60s)
#   PHASE=1|2|all            default: all (phase 1 then phase 2)
#   GPUS="0 1"
#   SLAM_WAIT_POLL_SEC=120   poll interval for run_1/2/3 prereqs
#   SLAM_WAIT_TIMEOUT=0      0 = wait forever until SLAM dir is ready
#   LEVELS_PARALLEL=1
#   SKIP_DONE=1
#   PHASE2_RUNS="run_1 run_2 run_3"
##################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJ_DIR}"

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  WAIT_PID="$1"
  shift
fi
WAIT_PID="${WAIT_PID:-}"

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

if [ -n "${WAIT_PID}" ]; then
  if kill -0 "${WAIT_PID}" 2>/dev/null; then
    echo "[$( _ts )] Waiting for PID ${WAIT_PID} to finish (poll every 60s)..."
    while kill -0 "${WAIT_PID}" 2>/dev/null; do
      sleep 60
    done
    echo "[$( _ts )] PID ${WAIT_PID} finished — starting lang field batch."
  else
    echo "[$( _ts )] PID ${WAIT_PID} not running — starting batch immediately."
  fi
fi

PHASE="${PHASE:-all}"
GPUS="${GPUS:-0 1}"
LEVELS_PARALLEL="${LEVELS_PARALLEL:-1}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-64}"
NUM_ITERS="${NUM_ITERS:-12000}"
VQ_LAYER_NUM="${VQ_LAYER_NUM:-1}"
TOPK="${TOPK:-4}"
RENDER_CHECKPOINT="${RENDER_CHECKPOINT:-auto}"
TRAIN_DOWNSCALE="${TRAIN_DOWNSCALE:-0.5}"
EXP="${EXP:-ActiveOpenSem}"
SKIP_DONE="${SKIP_DONE:-1}"
START_FROM="${START_FROM:-}"
DRY_RUN="${DRY_RUN:-0}"
LOG_EVERY="${LOG_EVERY:-500}"
SLAM_WAIT_POLL_SEC="${SLAM_WAIT_POLL_SEC:-120}"
SLAM_WAIT_TIMEOUT="${SLAM_WAIT_TIMEOUT:-0}"

LOG_DIR="${LOG_DIR:-${PROJ_DIR}/logs/lang_field_batch}"
mkdir -p "${LOG_DIR}"

# shellcheck disable=SC2206
GPU_ARR=($GPUS)
NUM_GPUS=${#GPU_ARR[@]}
if [ "${NUM_GPUS}" -lt 1 ]; then
  echo "ERROR: GPUS is empty" >&2
  exit 2
fi

PHASE1_JOBS=(
  "office1/run_0"
  "office3/run_0"
  "office4/run_0"
  "room1/run_0"
  "room2/run_0"
)
# office2/run_0 — via run_inject_office2_when_gpu1_free.sh (followup)

PHASE2_SCENES=(office0 office1 office3 office4 room0 room1 room2)
PHASE2_RUNS=(${PHASE2_RUNS:-run_1 run_2 run_3})

build_phase2_jobs() {
  PHASE2_JOBS=()
  local scene run_tag
  for scene in "${PHASE2_SCENES[@]}"; do
    for run_tag in "${PHASE2_RUNS[@]}"; do
      PHASE2_JOBS+=("${scene}/${run_tag}")
    done
  done
}
build_phase2_jobs

STATE_FILE="${LOG_DIR}/batch_state_$(date +%Y%m%d_%H%M%S).txt"
: > "${STATE_FILE}"

_splatam_ckpt() {
  local result_dir="$1"
  if [ -f "${result_dir}/splatam/final/params0.npz" ]; then
    echo "${result_dir}/splatam/final/params0.npz"
  elif [ -f "${result_dir}/splatam/final/params.npz" ]; then
    echo "${result_dir}/splatam/final/params.npz"
  else
    return 1
  fi
}

_prereqs_missing() {
  local result_dir="$1"
  local -n _out=$2
  _out=()
  if ! _splatam_ckpt "${result_dir}" >/dev/null 2>&1; then
    _out+=("splatam/final/params*.npz")
  fi
  [ -f "${result_dir}/keyframe_poses.json" ] || _out+=("keyframe_poses.json")
  [ -d "${result_dir}/language_features" ] || _out+=("language_features/")
}

_prereqs_ready() {
  local result_dir="$1"
  local missing=()
  _prereqs_missing "${result_dir}" missing
  [ "${#missing[@]}" -eq 0 ]
}

_is_run0_job() {
  local run_tag="$1"
  [ "${run_tag}" = "run_0" ]
}

_level_done() {
  local result_dir="$1"
  local level="$2"
  local k="$3"
  local l="$4"
  [ -f "${result_dir}/lang_field_${level}k${k}_l${l}/lang_field.pt" ]
}

_lang_field_done() {
  local result_dir="$1"
  local k="$2"
  local l="$3"
  _level_done "${result_dir}" s "${k}" "${l}" \
    && _level_done "${result_dir}" m "${k}" "${l}" \
    && _level_done "${result_dir}" l "${k}" "${l}"
}

_ensure_prereqs() {
  local result_dir="$1"
  local job_id="$2"
  local run_tag="$3"

  if _prereqs_ready "${result_dir}"; then
    return 0
  fi

  local missing=()
  _prereqs_missing "${result_dir}" missing

  if _is_run0_job "${run_tag}"; then
    echo "[$( _ts )] SKIP ${job_id}: run_0 missing ${missing[*]}"
    echo "${job_id}|skip|prereq" >> "${STATE_FILE}"
    return 1
  fi

  # run_1/2/3: wait for SLAM to finish (post-factum), do not skip immediately
  echo "[$( _ts )] WAIT ${job_id}: SLAM not ready (${missing[*]}), polling every ${SLAM_WAIT_POLL_SEC}s..."
  local waited=0
  while ! _prereqs_ready "${result_dir}"; do
    if [ "${SLAM_WAIT_TIMEOUT}" -gt 0 ] && [ "${waited}" -ge "${SLAM_WAIT_TIMEOUT}" ]; then
      echo "[$( _ts )] PENDING ${job_id}: SLAM still not ready after ${waited}s"
      echo "${job_id}|pending|slam_timeout" >> "${STATE_FILE}"
      return 1
    fi
    sleep "${SLAM_WAIT_POLL_SEC}"
    waited=$((waited + SLAM_WAIT_POLL_SEC))
    _prereqs_missing "${result_dir}" missing
    echo "[$( _ts )] WAIT ${job_id}: still missing ${missing[*]} (${waited}s elapsed)"
  done
  echo "[$( _ts )] READY ${job_id}: SLAM inputs appeared"
  return 0
}

_tail_level_logs() {
  local scene="$1"
  local run_tag="$2"
  local level log
  for level in s m l; do
    log="${LOG_DIR}/train_${scene}_${run_tag}_${level}.log"
    if [ -f "${log}" ]; then
      echo "--- tail ${log} ---" >&2
      tail -15 "${log}" >&2 || true
    fi
  done
}

_train_level() {
  local gpu="$1"
  local result_dir="$2"
  local scene="$3"
  local run_tag="$4"
  local level="$5"

  local log_file="${LOG_DIR}/train_${scene}_${run_tag}_${level}.log"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    bash "${SCRIPT_DIR}/03_train_gaussian_lang_field.sh" \
      "${result_dir}" \
      "${CODEBOOK_SIZE}" \
      "${level}" \
      "${NUM_ITERS}" \
      "cuda:0" \
      "${VQ_LAYER_NUM}" \
      "${TOPK}" \
      "${RENDER_CHECKPOINT}" \
      "${TRAIN_DOWNSCALE}" \
      "${LOG_EVERY}"
  ) > "${log_file}" 2>&1
}

_train_scene_run() {
  local gpu="$1"
  local scene="$2"
  local run_tag="$3"
  local job_id="${scene}/${run_tag}"
  local result_dir="${PROJ_DIR}/results/Replica/${scene}/${EXP}/${run_tag}"

  if ! _ensure_prereqs "${result_dir}" "${job_id}" "${run_tag}"; then
    return 0
  fi

  if [ "${SKIP_DONE}" = "1" ] && _lang_field_done "${result_dir}" "${CODEBOOK_SIZE}" "${VQ_LAYER_NUM}"; then
    echo "[$( _ts )] GPU${gpu} SKIP ${job_id}: s/m/l already trained"
    echo "${job_id}|skip|done" >> "${STATE_FILE}"
    return 0
  fi

  local -a levels_todo=()
  local level
  for level in s m l; do
    if [ "${SKIP_DONE}" = "1" ] && _level_done "${result_dir}" "${level}" "${CODEBOOK_SIZE}" "${VQ_LAYER_NUM}"; then
      echo "[$( _ts )] GPU${gpu} SKIP ${job_id}/${level}: already done"
      continue
    fi
    levels_todo+=("${level}")
  done

  if [ "${#levels_todo[@]}" -eq 0 ]; then
    echo "[$( _ts )] GPU${gpu} OK ${job_id}: all levels present"
    echo "${job_id}|ok|all_present" >> "${STATE_FILE}"
    return 0
  fi

  echo ""
  echo "=============================================="
  echo "[$( _ts )] GPU${gpu} TRAIN ${job_id}"
  echo "  Result dir : ${result_dir}"
  echo "  Levels     : ${levels_todo[*]} ($([ "${LEVELS_PARALLEL}" = "1" ] && echo parallel || echo sequential))"
  echo "  K/iters    : ${CODEBOOK_SIZE} / ${NUM_ITERS}"
  echo "  Downscale  : ${TRAIN_DOWNSCALE}"
  echo "  Render ckpt: ${RENDER_CHECKPOINT}"
  echo "=============================================="

  if [ "${DRY_RUN}" = "1" ]; then
    echo "${job_id}|dry-run|planned" >> "${STATE_FILE}"
    return 0
  fi

  local fail=0
  if [ "${LEVELS_PARALLEL}" = "1" ]; then
    local -a pids=()
    for level in "${levels_todo[@]}"; do
      _train_level "${gpu}" "${result_dir}" "${scene}" "${run_tag}" "${level}" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "${pid}" || fail=1
    done
  else
    for level in "${levels_todo[@]}"; do
      _train_level "${gpu}" "${result_dir}" "${scene}" "${run_tag}" "${level}" || fail=1
    done
  fi

  if [ "${fail}" -eq 0 ] && _lang_field_done "${result_dir}" "${CODEBOOK_SIZE}" "${VQ_LAYER_NUM}"; then
    echo "[$( _ts )] GPU${gpu} OK ${job_id}"
    echo "${job_id}|ok|trained" >> "${STATE_FILE}"
  else
    echo "[$( _ts )] GPU${gpu} FAILED ${job_id} (see ${LOG_DIR}/train_${scene}_${run_tag}_*.log)" >&2
    _tail_level_logs "${scene}" "${run_tag}"
    echo "${job_id}|fail|check_logs" >> "${STATE_FILE}"
    return 1
  fi
}

_filter_jobs() {
  local -n _src=$1
  local -n _dst=$2
  _dst=()
  local STARTED=0
  if [ -z "${START_FROM}" ]; then
    STARTED=1
  fi
  local job
  for job in "${_src[@]}"; do
    if [ "${STARTED}" -eq 0 ]; then
      if [ "${job}" = "${START_FROM}" ]; then
        STARTED=1
      else
        echo "[$( _ts )] SKIP ${job} (before START_FROM=${START_FROM})"
        continue
      fi
    fi
    _dst+=("${job}")
  done
}

_run_phase_pool() {
  local phase_label="$1"
  shift
  local -a QUEUE=("$@")

  if [ "${#QUEUE[@]}" -eq 0 ]; then
    echo "[$( _ts )] ${phase_label}: no jobs"
    return 0
  fi

  echo ""
  echo "[$( _ts )] === ${phase_label}: ${#QUEUE[@]} jobs, dynamic GPU pool ==="
  echo "  Queue: ${QUEUE[*]}"

  if [ "${DRY_RUN}" = "1" ]; then
    local job
    for job in "${QUEUE[@]}"; do
      echo "${job}|dry-run|${phase_label}" >> "${STATE_FILE}"
    done
    return 0
  fi

  local pending_idx=0
  local num_jobs=${#QUEUE[@]}
  declare -a GPU_PID=()
  declare -a GPU_JOB=()
  local gi running

  for ((gi = 0; gi < NUM_GPUS; gi++)); do
    GPU_PID[gi]=""
    GPU_JOB[gi]=""
  done

  _launch_on_gpu() {
    local gi=$1
    local job="${QUEUE[$pending_idx]}"
    local scene="${job%%/*}"
    local run_tag="${job#*/}"
    local gpu="${GPU_ARR[gi]}"

    pending_idx=$((pending_idx + 1))
    echo "[$( _ts )] LAUNCH ${job} on GPU ${gpu} (${pending_idx}/${num_jobs} dequeued)"
    (
      _train_scene_run "${gpu}" "${scene}" "${run_tag}"
    ) &
    GPU_PID[gi]=$!
    GPU_JOB[gi]="${job}"
  }

  while true; do
    for ((gi = 0; gi < NUM_GPUS; gi++)); do
      if [ -z "${GPU_PID[gi]}" ] && [ "${pending_idx}" -lt "${num_jobs}" ]; then
        _launch_on_gpu "${gi}"
      fi
    done

    running=0
    for ((gi = 0; gi < NUM_GPUS; gi++)); do
      if [ -n "${GPU_PID[gi]}" ]; then
        running=1
        break
      fi
    done
    [ "${running}" -eq 0 ] && break

    wait -n
    for ((gi = 0; gi < NUM_GPUS; gi++)); do
      if [ -n "${GPU_PID[gi]}" ] && ! kill -0 "${GPU_PID[gi]}" 2>/dev/null; then
        wait "${GPU_PID[gi]}" || true
        echo "[$( _ts )] DONE ${GPU_JOB[gi]} on GPU ${GPU_ARR[gi]}"
        GPU_PID[gi]=""
        GPU_JOB[gi]=""
      fi
    done
  done
}

FILTERED_P1=()
FILTERED_P2=()
_filter_jobs PHASE1_JOBS FILTERED_P1
_filter_jobs PHASE2_JOBS FILTERED_P2

echo "[$( _ts )] Lang field batch (phase=${PHASE})"
echo "  Phase 1    : ${#FILTERED_P1[@]} jobs (run_0, prereqs required)"
echo "  Phase 2    : ${#FILTERED_P2[@]} jobs (run_1/2/3, wait for SLAM)"
echo "  GPUs       : ${GPU_ARR[*]} (dynamic pool)"
echo "  SLAM wait  : poll=${SLAM_WAIT_POLL_SEC}s timeout=${SLAM_WAIT_TIMEOUT:-0}"
echo "  Log dir    : ${LOG_DIR}"
echo "  State file : ${STATE_FILE}"

case "${PHASE}" in
  1)
    _run_phase_pool "Phase 1 (run_0)" "${FILTERED_P1[@]}"
    ;;
  2)
    _run_phase_pool "Phase 2 (run_1/2/3)" "${FILTERED_P2[@]}"
    ;;
  all|*)
    _run_phase_pool "Phase 1 (run_0)" "${FILTERED_P1[@]}"
    _run_phase_pool "Phase 2 (run_1/2/3)" "${FILTERED_P2[@]}"
    ;;
esac

OK_JOBS=()
FAILED_JOBS=()
SKIPPED_JOBS=()
PENDING_JOBS=()
while IFS='|' read -r job_id status _rest; do
  case "${status}" in
    ok) OK_JOBS+=("${job_id}") ;;
    fail) FAILED_JOBS+=("${job_id}") ;;
    skip) SKIPPED_JOBS+=("${job_id}") ;;
    pending) PENDING_JOBS+=("${job_id}") ;;
  esac
done < "${STATE_FILE}"

echo ""
echo "[$( _ts )] Batch finished."
echo "  OK (${#OK_JOBS[@]})       : ${OK_JOBS[*]:-<none>}"
echo "  Failed (${#FAILED_JOBS[@]})  : ${FAILED_JOBS[*]:-<none>}"
echo "  Skipped (${#SKIPPED_JOBS[@]}) : ${SKIPPED_JOBS[*]:-<none>}"
echo "  Pending (${#PENDING_JOBS[@]}) : ${PENDING_JOBS[*]:-<none>}"
echo "  Per-level logs: ${LOG_DIR}/train_{scene}_{run}_{s|m|l}.log"

if [ "${#FAILED_JOBS[@]}" -gt 0 ]; then
  exit 1
fi
