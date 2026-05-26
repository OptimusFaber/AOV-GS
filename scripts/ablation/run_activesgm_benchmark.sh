#!/bin/bash
##################################################
# ActiveSGM (ActiveSem / OneFormer) benchmark on Replica.
#
# Usage:
#   bash scripts/ablation/run_activesgm_benchmark.sh [SCENES...]
#   bash scripts/ablation/run_activesgm_benchmark.sh all
#   bash scripts/ablation/run_activesgm_benchmark.sh office0 room2
#
# Env:
#   SEED=0          random seed (default 0)
#   ENABLE_VIS=0    visualization
#   GPU=0           CUDA_VISIBLE_DEVICES (default 0,1)
#
# Output:
#   results/Replica/<scene>/ActiveSem/run_N/
#   results/Replica/table_activesgm.csv
##################################################

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

export CUDA_VISIBLE_DEVICES="${GPU:-0,1}"
SEED="${SEED:-0}"
ENABLE_VIS="${ENABLE_VIS:-0}"
EXP="ActiveSem"

ALL_SCENES=(office0 office1 office2 office3 office4 room0 room1 room2)

if [[ $# -eq 0 ]] || [[ "${1:-}" == "all" ]]; then
  SCENES=("${ALL_SCENES[@]}")
else
  SCENES=("$@")
fi

echo "[$(date -Is)] ActiveSGM benchmark — scenes: ${SCENES[*]}"

for SCENE in "${SCENES[@]}"; do
  # auto run_N
  BASE="results/Replica/${SCENE}/${EXP}"
  mkdir -p "$BASE"
  RUN_ID=0
  while [[ -d "${BASE}/run_${RUN_ID}" ]]; do
    RUN_ID=$((RUN_ID + 1))
  done
  RESULT_DIR="${BASE}/run_${RUN_ID}"
  mkdir -p "$RESULT_DIR"

  echo ""
  echo "=============================================="
  echo "[$(date -Is)] SCENE=${SCENE}  EXP=${EXP}  run_${RUN_ID}"
  echo "=============================================="

  bash scripts/ablation/run_with_metrics.sh slam "$RESULT_DIR" \
    python src/main/activesgm.py \
      --cfg "configs/Replica/${SCENE}/${EXP}.py" \
      --seed "$SEED" \
      --result_dir "$RESULT_DIR" \
      --enable_vis "$ENABLE_VIS"

  # optional 3D recon eval (same as run_replica.sh)
  DASHSCENE="${SCENE:0:-1}_${SCENE: -1}"
  GT_MESH="data/replica_v1/${DASHSCENE}/mesh.ply"
  if [[ -f "$GT_MESH" ]]; then
    for stage in exploration_stage_0 exploration_stage_1 final; do
      ckpt="${RESULT_DIR}/splatam/${stage}/params.npz"
      if [[ -f "$ckpt" ]]; then
        python src/evaluation/eval_splatam_recon_v2.py \
          --ckpt "$ckpt" \
          --gt_mesh "$GT_MESH" \
          --transform_traj "data/Replica/${SCENE}/traj.txt" \
          --result_dir "${RESULT_DIR}/eval_3d/${stage}" || true
      fi
    done
  fi
done

echo ""
echo "[$(date -Is)] Summarizing → results/Replica/table_activesgm.csv"
python3 scripts/ablation/summarize_benchmark.py --algo activesgm --scenes "${SCENES[@]}"
echo "[$(date -Is)] Done."
