#!/bin/bash
##################################################
# Multiseed benchmark: ActiveGeom + ActiveOpenSem on all Replica scenes.
#
# Default 9 seeds (all != 0): 1 2 3 4 5 7 11 13 17
# Total jobs: 8 scenes × 9 seeds × 2 methods = 144
#
# Parallelism (2× V100-32GB, tuned for throughput):
#   GEOM_PER_GPU=3       — up to 3 ActiveGeom jobs per GPU (6 total on 2 GPUs)
#   OPEN_SEM_PER_GPU=2   — up to 2 ActiveOpenSem jobs per GPU (4 total on 2 GPUs)
#   Slots refill as jobs finish (wait -n), not fixed batches.
#   OOM jobs retried with OOM_RETRY_SLOTS_PER_GPU=1 per GPU.
#
# Usage:
#   bash scripts/aov-gs/run_multiseed_geom_open_sem_all.sh
#
# Background:
#   mkdir -p logs/multiseed
#   cd AOV-GS && nohup bash scripts/aov-gs/run_multiseed_geom_open_sem_all.sh \
#     > logs/multiseed/orchestrator.log 2>&1 &
#
# Env:
#   SEEDS="1 2 3 4 5 7 11 13 17"
#   SCENES="office0 office1 ..."     default: all 8 Replica scenes
#   METHODS="geom open_sem"          default: both
#   GEOM_PER_GPU=3                   ActiveGeom slots per GPU
#   OPEN_SEM_PER_GPU=2               ActiveOpenSem slots per GPU
#   GPUS="0 1"                       GPU ids
#   SKIP_DONE=1                      skip finished runs
#   OOM_RETRY_ROUNDS=3               retry passes for OOM/failed jobs
#   OOM_RETRY_SLOTS_PER_GPU=1        slots per GPU on retry passes
#   DELETE_AFTER_RECORD=1            remove run dir after CSV update (default)
#   MULTISEED_CSV=...                stats table (default results/Replica/multiseed_experiments.csv)
#   DRY_RUN=1                        print tasks only
##################################################

set -eo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

# 9 distinct non-zero seeds (fixed for reproducibility across machines)
SEEDS="${SEEDS:-1 2 3 4 5 7 11 13 17}"
SCENES="${SCENES:-office0 office1 office2 office3 office4 room0 room1 room2}"
METHODS="${METHODS:-geom open_sem}"
GEOM_PER_GPU="${GEOM_PER_GPU:-3}"
OPEN_SEM_PER_GPU="${OPEN_SEM_PER_GPU:-2}"
OOM_RETRY_SLOTS_PER_GPU="${OOM_RETRY_SLOTS_PER_GPU:-1}"
SKIP_DONE="${SKIP_DONE:-1}"
OOM_RETRY_ROUNDS="${OOM_RETRY_ROUNDS:-3}"
DRY_RUN="${DRY_RUN:-0}"
GPUS="${GPUS:-0 1}"

LOG_DIR="${PROJ_DIR}/logs/multiseed"
STATE_DIR="${LOG_DIR}/state"
mkdir -p "$LOG_DIR" "$STATE_DIR"

STATS_PY="${PROJ_DIR}/scripts/aov-gs/append_multiseed_stats.py"
MULTISEED_CSV="${MULTISEED_CSV:-${PROJ_DIR}/results/Replica/multiseed_experiments.csv}"
export MULTISEED_CSV DELETE_AFTER_RECORD="${DELETE_AFTER_RECORD:-1}"

JOB_SCRIPT="${PROJ_DIR}/scripts/aov-gs/run_multiseed_single_job.sh"
chmod +x "$JOB_SCRIPT"
chmod +x "$STATS_PY"

echo "[$(date -Is)] Generating configs..."
python3 scripts/aov-gs/gen_open_sem_scene_configs.py

# shellcheck disable=SC2206
SCENE_ARR=($SCENES)
# shellcheck disable=SC2206
SEED_ARR=($SEEDS)
# shellcheck disable=SC2206
METHOD_ARR=($METHODS)
# shellcheck disable=SC2206
GPU_ARR=($GPUS)
NUM_GPUS=${#GPU_ARR[@]}

build_task_list() {
  local f="$1"
  : > "$f"
  for method in "${METHOD_ARR[@]}"; do
    for scene in "${SCENE_ARR[@]}"; do
      for seed in "${SEED_ARR[@]}"; do
        if [[ "$seed" == "0" ]]; then
          echo "[$(date -Is)] WARN: skipping seed 0" >&2
          continue
        fi
        echo "${method} ${scene} ${seed}" >> "$f"
      done
    done
  done
}

is_done() {
  local method=$1 scene=$2 seed=$3
  local experiment
  case "$method" in
    geom) experiment="ActiveGeom" ;;
    open_sem) experiment="ActiveOpenSem" ;;
    *) return 1 ;;
  esac
  python3 "$STATS_PY" --check \
    --csv "$MULTISEED_CSV" \
    --experiment "$experiment" \
    --seed "$seed" \
    --scene "$scene"
}

slots_for_method() {
  local method=$1
  local override=${2:-}
  if [[ -n "$override" ]]; then
    echo "$override"
    return
  fi
  case "$method" in
    geom) echo "$GEOM_PER_GPU" ;;
    open_sem) echo "$OPEN_SEM_PER_GPU" ;;
    *) echo 1 ;;
  esac
}

pick_gpu_for_method() {
  local method=$1
  local slots_override=$2
  local limit gi best_gi=-1 best_load=999999

  limit="$(slots_for_method "$method" "$slots_override")"
  for ((gi = 0; gi < NUM_GPUS; gi++)); do
    if (( GPU_RUNNING[gi] < limit && GPU_RUNNING[gi] < best_load )); then
      best_load=${GPU_RUNNING[gi]}
      best_gi=$gi
    fi
  done
  echo "$best_gi"
}

run_pool() {
  local task_file=$1
  local pass_name=$2
  local failed_file=$3
  local slots_override=${4:-}

  : > "$failed_file"

  local -a PENDING=()
  local launched=0
  local skipped=0

  while IFS=' ' read -r method scene seed || [[ -n "${method:-}" ]]; do
    [[ -z "${method:-}" ]] && continue

    if [[ "$SKIP_DONE" == "1" ]] && is_done "$method" "$scene" "$seed"; then
      skipped=$((skipped + 1))
      echo "[$(date -Is)] [$pass_name] SKIP done: $method $scene seed=$seed"
      continue
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
      local limit
      limit="$(slots_for_method "$method" "$slots_override")"
      echo "[DRY_RUN] $method $scene seed=$seed (slots/gpu=$limit)"
      continue
    fi

    PENDING+=("${method} ${scene} ${seed}")
  done < "$task_file"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[$(date -Is)] [$pass_name] pending=$((${#PENDING[@]})) skipped=$skipped"
    return 0
  fi

  declare -a GPU_RUNNING=()
  for ((gi = 0; gi < NUM_GPUS; gi++)); do
    GPU_RUNNING[gi]=0
  done

  declare -A PID_GPU_IDX=()
  declare -A PID_LABEL=()
  local -a RUNNING_PIDS=()
  local pending_idx=0

  launch_pending_at() {
    local gi=$1
    local line="${PENDING[$pending_idx]}"
    local method scene seed gpu label

    read -r method scene seed <<< "$line"
    gpu="${GPU_ARR[$gi]}"
    label="${method}:${scene}:seed${seed}"

    echo "[$(date -Is)] [$pass_name] LAUNCH $label gpu=$gpu slot=$((GPU_RUNNING[gi] + 1))/$(slots_for_method "$method" "$slots_override")"
    (
      export SKIP_DONE=0
      bash "$JOB_SCRIPT" "$method" "$scene" "$seed" "$gpu"
    ) &
    local pid=$!

    PID_GPU_IDX[$pid]=$gi
    PID_LABEL[$pid]=$label
    RUNNING_PIDS+=("$pid")
    GPU_RUNNING[gi]=$((GPU_RUNNING[gi] + 1))
    pending_idx=$((pending_idx + 1))
    launched=$((launched + 1))
  }

  try_fill_slots() {
    local gi gi_pick method
    while (( pending_idx < ${#PENDING[@]} )); do
      read -r method _ _ <<< "${PENDING[$pending_idx]}"
      gi_pick="$(pick_gpu_for_method "$method" "$slots_override")"
      [[ "$gi_pick" -lt 0 ]] && break
      launch_pending_at "$gi_pick"
    done
  }

  on_job_finished() {
    local pid=$1 rc=$2
    local gi=${PID_GPU_IDX[$pid]}
    local label=${PID_LABEL[$pid]}

    GPU_RUNNING[gi]=$((GPU_RUNNING[gi] - 1))
    unset "PID_GPU_IDX[$pid]" "PID_LABEL[$pid]"

    local -a next_pids=()
    for p in "${RUNNING_PIDS[@]}"; do
      [[ "$p" != "$pid" ]] && next_pids+=("$p")
    done
    RUNNING_PIDS=("${next_pids[@]}")

    if [[ $rc -ne 0 ]]; then
      echo "$label" >> "$failed_file"
      echo "[$(date -Is)] [$pass_name] FAIL $label rc=$rc"
    else
      echo "[$(date -Is)] [$pass_name] OK $label"
    fi
  }

  try_fill_slots

  while (( ${#RUNNING_PIDS[@]} > 0 )); do
    set +e
    finished_pid=""
    wait -n -p finished_pid 2>/dev/null
    rc=$?
    if [[ -z "$finished_pid" ]]; then
      finished_pid=${RUNNING_PIDS[0]}
      wait "$finished_pid"
      rc=$?
    fi
    set -e
    on_job_finished "$finished_pid" "$rc"
    try_fill_slots
  done

  echo "[$(date -Is)] [$pass_name] launched=$launched skipped=$skipped failed=$(wc -l < "$failed_file" | tr -d ' ')"
}

TASK_FILE="${STATE_DIR}/tasks_all.txt"
FAILED_FILE="${STATE_DIR}/failed.txt"
build_task_list "$TASK_FILE"

TOTAL=$(wc -l < "$TASK_FILE" | tr -d ' ')
GEOM_TOTAL=$((GEOM_PER_GPU * NUM_GPUS))
OPEN_SEM_TOTAL=$((OPEN_SEM_PER_GPU * NUM_GPUS))

echo "[$(date -Is)] Multiseed benchmark"
echo "  Seeds           : ${SEED_ARR[*]}"
echo "  Scenes          : ${SCENE_ARR[*]}"
echo "  Methods         : ${METHOD_ARR[*]}"
echo "  Tasks           : $TOTAL"
echo "  GPUs            : ${GPU_ARR[*]}"
echo "  Geom slots/GPU  : $GEOM_PER_GPU (max $GEOM_TOTAL concurrent)"
echo "  OpenSem slots/GPU: $OPEN_SEM_PER_GPU (max $OPEN_SEM_TOTAL concurrent)"
echo "  Skip done       : $SKIP_DONE"
echo "  Stats CSV       : $MULTISEED_CSV"
echo "  Delete runs     : ${DELETE_AFTER_RECORD:-1} (after stats recorded)"
echo ""

if [[ "$DRY_RUN" == "1" ]]; then
  run_pool "$TASK_FILE" "dry-run" "$FAILED_FILE"
  exit 0
fi

run_pool "$TASK_FILE" "pass-parallel" "$FAILED_FILE"

round=1
while [[ -s "$FAILED_FILE" && "$round" -le "$OOM_RETRY_ROUNDS" ]]; do
  RETRY_FILE="${STATE_DIR}/retry_round_${round}.txt"
  : > "$RETRY_FILE"
  while IFS= read -r label; do
    [[ -z "$label" ]] && continue
    method="${label%%:*}"
    rest="${label#*:}"
    scene="${rest%%:seed*}"
    seed="${rest##*seed}"
    echo "$method $scene $seed" >> "$RETRY_FILE"
  done < "$FAILED_FILE"

  echo ""
  echo "[$(date -Is)] Retry round $round/$OOM_RETRY_ROUNDS ($(wc -l < "$RETRY_FILE" | tr -d ' ') jobs), slots/GPU=$OOM_RETRY_SLOTS_PER_GPU"
  sleep 10
  run_pool "$RETRY_FILE" "retry-${round}" "$FAILED_FILE" "$OOM_RETRY_SLOTS_PER_GPU"
  round=$((round + 1))
done

if [[ -s "$FAILED_FILE" ]]; then
  echo ""
  echo "[$(date -Is)] FINISHED WITH FAILURES:"
  cat "$FAILED_FILE"
  echo "Logs: ${LOG_DIR}/"
  exit 1
fi

echo ""
echo "[$(date -Is)] All multiseed jobs complete."
echo "Stats table: ${MULTISEED_CSV}"
