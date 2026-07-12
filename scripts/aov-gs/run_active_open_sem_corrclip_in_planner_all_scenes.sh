#!/bin/bash
##################################################
# ActiveOpenSem on all Replica scenes except room0:
#   - CorrCLIP ON (--corrclip 1)
#   - active_gs_hybrid_v3 planner (fixed in ActiveOpenSem.py)
#   - Fixed output: results/Replica/{SCENE}/ActiveOpenSem/run_1_corrclip_in_planner/
#
# Usage:
#   bash scripts/aov-gs/run_active_open_sem_corrclip_in_planner_all_scenes.sh
#
# Background + log:
#   cd AOV-GS && nohup bash scripts/aov-gs/run_active_open_sem_corrclip_in_planner_all_scenes.sh > active_open_sem_corrclip_in_planner_all.log 2>&1 &
#
# Env overrides:
#   SEED=0              random seed (default 0)
#   RESULT_RUN=...      result subfolder (default run_1_corrclip_in_planner)
#   SCENES="..."        space-separated scene list (default: all except room0)
#   START_FROM=office1  skip scenes before this one (resume batch)
#   SKIP_DONE=1         skip scenes whose eval_final already exists
##################################################

set -eo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

SEED="${SEED:-0}"
EXP="ActiveOpenSem"
RESULT_RUN="${RESULT_RUN:-run_1_corrclip_in_planner}"
START_FROM="${START_FROM:-}"
SKIP_DONE="${SKIP_DONE:-0}"

DEFAULT_SCENES=(room2)
if [ -n "${SCENES:-}" ]; then
  # shellcheck disable=SC2206
  SCENE_LIST=($SCENES)
else
  SCENE_LIST=("${DEFAULT_SCENES[@]}")
fi

echo "[$(date -Is)] Generating ActiveOpenSem_base / Geom / ActiveOpenSem configs..."
python scripts/aov-gs/gen_open_sem_scene_configs.py

echo "[$(date -Is)] Batch: ActiveOpenSem + CorrCLIP -> ${RESULT_RUN}"
echo "  Scenes     : ${SCENE_LIST[*]}"
echo "  Seed       : ${SEED}"
echo "  Start from : ${START_FROM:-<first>}"
echo "  Skip done  : ${SKIP_DONE}"
echo ""

FAILED_SCENES=()
SKIPPED_SCENES=()
STARTED=0
if [ -z "$START_FROM" ]; then
  STARTED=1
fi

for SCENE in "${SCENE_LIST[@]}"; do
  if [ "$STARTED" -eq 0 ]; then
    if [ "$SCENE" = "$START_FROM" ]; then
      STARTED=1
    else
      echo "[$(date -Is)] SKIP ${SCENE} (before START_FROM=${START_FROM})"
      continue
    fi
  fi

  CFG="configs/Replica/${SCENE}/${EXP}.py"
  RESULT_DIR="${PROJ_DIR}/results/Replica/${SCENE}/${EXP}/${RESULT_RUN}"
  DONE_MARKER="${RESULT_DIR}/splatam/eval_final/render_result.txt"

  if [ ! -f "$CFG" ]; then
    echo "[$(date -Is)] SKIP ${SCENE}: missing ${CFG}"
    SKIPPED_SCENES+=("$SCENE")
    continue
  fi

  if [ "$SKIP_DONE" = "1" ] && [ -f "$DONE_MARKER" ]; then
    echo "[$(date -Is)] SKIP ${SCENE}: already done (${DONE_MARKER})"
    SKIPPED_SCENES+=("$SCENE")
    continue
  fi

  echo ""
  echo "=============================================="
  echo "[$(date -Is)] SCENE=${SCENE}"
  echo "  Config     : ${CFG}"
  echo "  Result dir : results/Replica/${SCENE}/${EXP}/${RESULT_RUN}"
  echo "  CorrCLIP   : ON"
  echo "  Planner    : active_gs_hybrid_v3"
  echo "=============================================="

  if bash scripts/aov-gs/01_slam_exploration_with_corr_clip.sh \
      "$SCENE" "$EXP" "$SEED" 0 0 "$RESULT_RUN"; then
    echo "[$(date -Is)] OK ${SCENE}"
  else
    rc=$?
    echo "[$(date -Is)] FAILED ${SCENE} (exit ${rc})" >&2
    FAILED_SCENES+=("$SCENE")
  fi
done

echo ""
echo "[$(date -Is)] Batch finished."
echo "  Failed  : ${FAILED_SCENES[*]:-<none>}"
echo "  Skipped : ${SKIPPED_SCENES[*]:-<none>}"
echo "Results under: results/Replica/{scene}/${EXP}/${RESULT_RUN}/"

if [ "${#FAILED_SCENES[@]}" -gt 0 ]; then
  exit 1
fi
