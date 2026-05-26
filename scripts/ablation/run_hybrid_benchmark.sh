#!/bin/bash
##################################################
# Hybrid (ActiveOpenSemHybrid) benchmark on Replica.
#
# Usage:
#   bash scripts/ablation/run_hybrid_benchmark.sh [SCENES...]
#   bash scripts/ablation/run_hybrid_benchmark.sh all
#   HYBRID_K=16 bash scripts/ablation/run_hybrid_benchmark.sh office0
#
# Env:
#   HYBRID_K=8     top-K geometry candidates for SAM+CLIP (default 8)
#   SEED=0
#   ENABLE_VIS=0
#   GPU=0,1
#
# Output:
#   results/Replica/<scene>/ActiveOpenSemHybrid/run_N/
#   results/Replica/table_hybrid.csv
##################################################

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

export CUDA_VISIBLE_DEVICES="${GPU:-0,1}"
SEED="${SEED:-0}"
ENABLE_VIS="${ENABLE_VIS:-0}"
HYBRID_K="${HYBRID_K:-8}"
EXP="ActiveOpenSemHybrid"

ALL_SCENES=(office0 office1 office2 office3 office4 room0 room1 room2)

if [[ $# -eq 0 ]] || [[ "${1:-}" == "all" ]]; then
  SCENES=("${ALL_SCENES[@]}")
else
  SCENES=("$@")
fi

echo "[$(date -Is)] Generating / updating OpenSem configs..."
python3 scripts/ablation/gen_open_sem_scene_configs.py

echo "[$(date -Is)] Hybrid benchmark — K=${HYBRID_K}, scenes: ${SCENES[*]}"

for SCENE in "${SCENES[@]}"; do
  python3 <<PY
import sys
sys.path.insert(0, "scripts/ablation")
from benchmark_utils import patch_hybrid_k
patch_hybrid_k("${SCENE}", ${HYBRID_K})
print("  patched ${SCENE}: max_semantic_candidates=${HYBRID_K}")
PY

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
  echo "[$(date -Is)] SCENE=${SCENE}  K=${HYBRID_K}  run_${RUN_ID}"
  echo "=============================================="

  bash scripts/ablation/run_with_metrics.sh slam "$RESULT_DIR" \
    python src/main/activesgm.py \
      --cfg "configs/Replica/${SCENE}/${EXP}.py" \
      --seed "$SEED" \
      --result_dir "$RESULT_DIR" \
      --enable_vis "$ENABLE_VIS" \
      --corrclip 0
done

echo ""
echo "[$(date -Is)] Summarizing → results/Replica/table_hybrid.csv"
python3 scripts/ablation/summarize_benchmark.py --algo hybrid --scenes "${SCENES[@]}"
echo "[$(date -Is)] Done."
