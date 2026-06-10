#!/bin/bash
##################################################
# Run one multiseed benchmark job: ActiveGeom or ActiveOpenSem.
#
# Usage:
#   bash scripts/aov-gs/run_multiseed_single_job.sh <geom|open_sem> <scene> <seed> [gpu_id]
#
# After success:
#   - append row to results/Replica/multiseed_experiments.csv
#   - delete run folder (DELETE_AFTER_RECORD=1, default)
#
# Env:
#   SAM_CLIP_DEVICE=cuda:0
#   SKIP_DONE=1              skip if row already in CSV
#   DELETE_AFTER_RECORD=1    free disk after recording stats
#   MULTISEED_CSV=...        stats table path
##################################################

set -eo pipefail

METHOD=${1:?method: geom|open_sem}
SCENE=${2:?scene}
SEED=${3:?seed}
GPU_ID=${4:-0}

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export SAM_CLIP_DEVICE="${SAM_CLIP_DEVICE:-cuda:0}"

RESULT_RUN="seed_${SEED}"
SKIP_DONE="${SKIP_DONE:-1}"
DELETE_AFTER_RECORD="${DELETE_AFTER_RECORD:-1}"
MULTISEED_CSV="${MULTISEED_CSV:-${PROJ_DIR}/results/Replica/multiseed_experiments.csv}"
STATS_PY="${PROJ_DIR}/scripts/aov-gs/append_multiseed_stats.py"

case "$METHOD" in
  geom)
    EXP="ActiveOpenSemGeom"
    RESULT_ROOT="ActiveGeom"
    EXPERIMENT="ActiveGeom"
    CORRCLIP=0
    ;;
  open_sem)
    EXP="ActiveOpenSem"
    RESULT_ROOT="ActiveOpenSem"
    EXPERIMENT="ActiveOpenSem"
    CORRCLIP=1
    ;;
  *)
    echo "Unknown method: $METHOD (expected geom or open_sem)" >&2
    exit 2
    ;;
esac

CFG="configs/Replica/${SCENE}/${EXP}.py"
RESULT_DIR="${PROJ_DIR}/results/Replica/${SCENE}/${RESULT_ROOT}/${RESULT_RUN}"
DONE_MARKER="${RESULT_DIR}/splatam/eval_final/render_result.txt"
LOG_DIR="${PROJ_DIR}/logs/multiseed"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${METHOD}_${SCENE}_seed${SEED}_gpu${GPU_ID}.log"

record_and_cleanup() {
  python3 "$STATS_PY" \
    --csv "$MULTISEED_CSV" \
    --experiment "$EXPERIMENT" \
    --seed "$SEED" \
    --scene "$SCENE" \
    --run-dir "$RESULT_DIR" \
    --delete "$DELETE_AFTER_RECORD" \
    | tee -a "$LOG_FILE"
}

if [[ ! -f "$CFG" ]]; then
  echo "[$(date -Is)] MISSING CFG: $CFG" | tee -a "$LOG_FILE"
  exit 3
fi

if [[ "$SKIP_DONE" == "1" ]] && python3 "$STATS_PY" --check \
    --csv "$MULTISEED_CSV" \
    --experiment "$EXPERIMENT" \
    --seed "$SEED" \
    --scene "$SCENE"; then
  echo "[$(date -Is)] SKIP (in CSV): $EXPERIMENT $SCENE seed=$SEED" | tee -a "$LOG_FILE"
  # stale folder from interrupted run
  if [[ -d "$RESULT_DIR" ]]; then
    echo "[$(date -Is)] removing stale dir $RESULT_DIR" | tee -a "$LOG_FILE"
    rm -rf "$RESULT_DIR"
  fi
  exit 0
fi

mkdir -p "$RESULT_DIR"

{
  echo "[$(date -Is)] START method=$METHOD scene=$SCENE seed=$SEED gpu=$GPU_ID"
  echo "  experiment : $EXPERIMENT"
  echo "  cfg        : $CFG"
  echo "  result_dir : $RESULT_DIR"
  echo "  stats_csv  : $MULTISEED_CSV"
  echo "  corrclip   : $CORRCLIP"
  echo "  sam_device : $SAM_CLIP_DEVICE"
  echo "  visible    : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
} | tee "$LOG_FILE"

set +e
python3 src/main/activesgm.py \
  --cfg "configs/Replica/${SCENE}/${EXP}.py" \
  --seed "$SEED" \
  --result_dir "$RESULT_DIR" \
  --enable_vis 0 \
  --corrclip "$CORRCLIP" \
  >> "$LOG_FILE" 2>&1
RC=$?
set -e

if [[ $RC -eq 0 && -f "$DONE_MARKER" ]]; then
  record_and_cleanup
  echo "[$(date -Is)] OK method=$METHOD scene=$SCENE seed=$SEED" | tee -a "$LOG_FILE"
  exit 0
fi

if grep -qiE 'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|std::bad_alloc|Killed' "$LOG_FILE"; then
  echo "[$(date -Is)] OOM method=$METHOD scene=$SCENE seed=$SEED rc=$RC" | tee -a "$LOG_FILE"
  exit 137
fi

echo "[$(date -Is)] FAILED method=$METHOD scene=$SCENE seed=$SEED rc=$RC" | tee -a "$LOG_FILE"
if [[ -f "$LOG_FILE" ]]; then
  echo "--- last 30 lines of ${LOG_FILE} ---" | tee -a "$LOG_FILE"
  tail -n 30 "$LOG_FILE" | tee -a "$LOG_FILE" >&2
fi
exit "${RC:-1}"
