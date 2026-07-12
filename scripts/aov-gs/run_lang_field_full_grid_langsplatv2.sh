#!/bin/bash
##################################################
# LangSplatV2 — language field training grid (stage 3)
#
# Only RESULT_DIR/language_features/ (raw SAM+CLIP, 512-D) is required after stage 1.
# Autoencoder step (language_features_dim*) is not needed.
#
# Grid parameters are set via env:
#   GRID_K_VALUES   — space-separated K (codebook_size), default "64"
#   GRID_L_VALUES   — space-separated L (vq_layer_num), default "1"
#   GRID_TOPK       — top-k sparse per level, default "4"
#   GRID_LEVELS     — SAM levels, default "s m l"
#   NUM_ITERS       — iterations per job, default 30000
#   DEVICE          — cuda:0
#   RENDER_CHECKPOINT — auto | on | off
#   LOG_DIR         — stdout logs (default: RESULT_DIR/lang_field_grid_logs)
#
# Example:
#   GRID_K_VALUES="32 64" NUM_ITERS=12000 bash scripts/aov-gs/run_lang_field_full_grid.sh \
#       results/Replica/office0/ActiveOpenSem/run_0
##################################################

set -euo pipefail

RESULT_DIR=${1:?"Usage: $0 RESULT_DIR"}

NUM_ITERS=${NUM_ITERS:-30000}
DEVICE=${DEVICE:-cuda:0}
RENDER_CHECKPOINT=${RENDER_CHECKPOINT:-auto}
LOG_EVERY=${LOG_EVERY:-500}

GRID_K_VALUES=${GRID_K_VALUES:-64}
GRID_L_VALUES=${GRID_L_VALUES:-1}
GRID_TOPK=${GRID_TOPK:-4}
GRID_LEVELS=${GRID_LEVELS:-s m l}

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

LOG_DIR="${LOG_DIR:-${RESULT_DIR}/lang_field_grid_logs}"
mkdir -p "$LOG_DIR"

if [ ! -d "${RESULT_DIR}/language_features" ]; then
    echo "ERROR: missing ${RESULT_DIR}/language_features — run stage 1 first (SLAM + SAM/CLIP)."
    exit 1
fi

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

_pick_ckpt() {
    local fd="${RESULT_DIR}/splatam/final"
    if [ -f "${fd}/params0.npz" ]; then echo "${fd}/params0.npz"
    elif [ -f "${fd}/params.npz" ]; then echo "${fd}/params.npz"
    else echo "${fd}/params0.npz"; fi
}

CHECKPOINT="$(_pick_ckpt)"
POSES="${RESULT_DIR}/keyframe_poses.json"

if [ ! -f "$POSES" ]; then
    echo "ERROR: missing $POSES"
    exit 1
fi

if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "ERROR: python3/python not found"
    exit 127
fi

echo "=============================================="
echo "  Lang field grid (LangSplatV2)"
echo "  RESULT_DIR         : $RESULT_DIR"
echo "  Features           : ${RESULT_DIR}/language_features"
echo "  K values           : $GRID_K_VALUES"
echo "  L values           : $GRID_L_VALUES"
echo "  TOPK               : $GRID_TOPK"
echo "  Levels             : $GRID_LEVELS"
echo "  NUM_ITERS          : $NUM_ITERS"
echo "  DEVICE             : $DEVICE"
echo "  RENDER_CHECKPOINT  : $RENDER_CHECKPOINT"
echo "  LOG_DIR            : $LOG_DIR"
echo "=============================================="

_failed=0

for K in $GRID_K_VALUES; do
  for L in $GRID_L_VALUES; do
    for TOPK in $GRID_TOPK; do
      pids=()
      for lev in $GRID_LEVELS; do
        out="${RESULT_DIR}/lang_field_${lev}k${K}_l${L}"
        log="${LOG_DIR}/stdout_k${K}_l${L}_${lev}.log"
        echo "[$(_ts)] START  K=${K} L=${L} topk=${TOPK} level=${lev} → ${out}"
        (
          set +e
          "$PY" scripts/train_language_field.py \
            --checkpoint         "$CHECKPOINT" \
            --poses              "$POSES" \
            --features_dir       "${RESULT_DIR}/language_features" \
            --level              "$lev" \
            --output_dir         "$out" \
            --codebook_size      "$K" \
            --vq_layer_num       "$L" \
            --topk               "$TOPK" \
            --num_iters          "$NUM_ITERS" \
            --device             "$DEVICE" \
            --render_checkpoint  "$RENDER_CHECKPOINT" \
            --log_every          "$LOG_EVERY" \
            >"$log" 2>&1
          ec=$?
          if [ "$ec" -eq 0 ]; then
            echo "[$(_ts)] OK     K=${K} L=${L} ${lev}"
          else
            echo "[$(_ts)] FAIL   K=${K} L=${L} ${lev} exit=$ec → $log"
            exit "$ec"
          fi
        ) &
        pids+=($!)
      done
      for pid in "${pids[@]}"; do
        wait "$pid" || _failed=1
      done
    done
  done
done

echo ""
if [ "$_failed" -eq 0 ]; then
    echo "[$(_ts)] Done. lang_field.pt in lang_field_*k*_l*/ ; stdout: $LOG_DIR"
else
    echo "[$(_ts)] Errors occurred — see $LOG_DIR/stdout_*.log"
    exit 1
fi
