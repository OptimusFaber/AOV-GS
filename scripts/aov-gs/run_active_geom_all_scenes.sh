#!/bin/bash
##################################################
# ActiveGeom (ActiveOpenSemGeom config) on all Replica scenes:
#   - geometry-only active_gs planner
#   - CorrCLIP OFF
#   - post-refinement capped at post_refine_steps=200 (from ActiveOpenSem_base)
#   - Output: results/Replica/{SCENE}/ActiveGeom/{RESULT_RUN}/
#
# Usage:
#   bash scripts/aov-gs/run_active_geom_all_scenes.sh
#
# Background + log:
#   cd AOV-GS-V2 && nohup bash scripts/aov-gs/run_active_geom_all_scenes.sh > active_geom_all_scenes.log 2>&1 &
#
# Env overrides:
#   SEED=0              random seed (default 0)
#   RESULT_RUN=run_0    result subfolder (default run_0)
#   SCENES="..."        space-separated scene list (default: all 8 Replica scenes)
#   START_FROM=office1  skip scenes before this one (resume batch)
#   SKIP_DONE=1         skip scenes whose eval_final already exists
##################################################

set -eo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

SEED="${SEED:-0}"
EXP="ActiveOpenSemGeom"
RESULT_ROOT="ActiveGeom"
RESULT_RUN="${RESULT_RUN:-run_0}"
START_FROM="${START_FROM:-}"
SKIP_DONE="${SKIP_DONE:-0}"

DEFAULT_SCENES=(office0 office1 office2 office3 office4 room0 room1 room2)
if [ -n "${SCENES:-}" ]; then
  # shellcheck disable=SC2206
  SCENE_LIST=($SCENES)
else
  SCENE_LIST=("${DEFAULT_SCENES[@]}")
fi

echo "[$(date -Is)] Generating ActiveOpenSem_base / Geom / ActiveOpenSem configs..."
python3 scripts/aov-gs/gen_open_sem_scene_configs.py

echo "[$(date -Is)] Batch: ActiveGeom -> ${RESULT_RUN}"
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
  RESULT_DIR="${PROJ_DIR}/results/Replica/${SCENE}/${RESULT_ROOT}/${RESULT_RUN}"
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
  echo "  Result dir : results/Replica/${SCENE}/${RESULT_ROOT}/${RESULT_RUN}"
  echo "  Planner    : active_gs (geometry-only)"
  echo "  CorrCLIP   : OFF"
  echo "=============================================="

  mkdir -p "$RESULT_DIR"

  if python3 src/main/activesgm.py \
      --cfg "configs/Replica/${SCENE}/${EXP}.py" \
      --seed "$SEED" \
      --result_dir "$RESULT_DIR" \
      --enable_vis 0 \
      --corrclip 0; then
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
echo "Results under: results/Replica/{scene}/${RESULT_ROOT}/${RESULT_RUN}/"

if [ "${#FAILED_SCENES[@]}" -gt 0 ]; then
  exit 1
fi
