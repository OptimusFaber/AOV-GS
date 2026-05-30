#!/bin/bash
##################################################
# Sequential ActiveGeom + ActiveOpenSem on remaining Replica scenes.
# office0 is skipped (already done).
#
# Usage:
#   bash scripts/aov-gs/run_geom_active_open_sem_remaining_scenes.sh
#
# One-liner (background + log):
#   cd AOV-GS-V2 && nohup bash scripts/aov-gs/run_geom_active_open_sem_remaining_scenes.sh > geom_active_open_sem_all_scenes.log 2>&1 &
##################################################

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

SCENES=(office1 office2 office3 office4 room0 room1 room2)
SEED="${SEED:-0}"

echo "[$(date -Is)] Generating ActiveOpenSem_base / Geom / ActiveOpenSem configs..."
python scripts/aov-gs/gen_open_sem_scene_configs.py

for SCENE in "${SCENES[@]}"; do
  echo ""
  echo "=============================================="
  echo "[$(date -Is)] SCENE=${SCENE} — ActiveGeom (ActiveOpenSemGeom)"
  echo "=============================================="
  bash scripts/aov-gs/01_slam_exploration.sh "$SCENE" ActiveOpenSemGeom "$SEED" 0 0

  echo ""
  echo "=============================================="
  echo "[$(date -Is)] SCENE=${SCENE} — ActiveOpenSem"
  echo "=============================================="
  bash scripts/aov-gs/01_slam_exploration.sh "$SCENE" ActiveOpenSem "$SEED" 0 0
done

echo ""
echo "[$(date -Is)] All scenes finished."
