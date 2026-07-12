#!/bin/bash
##################################################
# Stage 3 — selective grid + fixed parallelism waves
#
# Iterations:
#   D ∈ {3,4,8,16}  — all levels s,m,l  →  NUM_ITERS=30000
#   D ∈ {32,64}     — level m only       →  NUM_ITERS=20000
#
# Waves (one GPU; within a wave — parallel, wall-time = max over jobs):
#   1) 4s 4m 4l 8s 8m
#   2) 8l 16s 16m
#   3) 3s 3m 3l 16l
#   4) 32m
#   5) 64m
#
# Loss logs: each output_dir gets loss_{D}{level}.txt
# (every LOG_EVERY iterations, default 500) — see train_language_field / LangSplatam.
#
# Usage:
#   cd AOV-GS
#   bash scripts/aov-gs/run_lang_field_full_grid.sh \
#       results/Replica/office0/ActiveOpenSem/run_0
#
# Env:
#   NUM_ITERS_SMALL=30000   # for D=3,4,8,16
#   NUM_ITERS_LARGE=20000   # for D=32,64
#   LOG_EVERY=500
#   DEVICE=cuda:0
#   RENDER_CHECKPOINT=auto
#   LOG_DIR=...             # stdout logs, default: RESULT_DIR/lang_field_grid_logs
#
##################################################

set -euo pipefail

RESULT_DIR=${1:?"Usage: $0 RESULT_DIR"}

NUM_ITERS_SMALL=${NUM_ITERS_SMALL:-30000}
NUM_ITERS_LARGE=${NUM_ITERS_LARGE:-20000}
LOG_EVERY=${LOG_EVERY:-500}
DEVICE=${DEVICE:-cuda:0}
RENDER_CHECKPOINT=${RENDER_CHECKPOINT:-auto}

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

LOG_DIR="${LOG_DIR:-${RESULT_DIR}/lang_field_grid_logs}"
mkdir -p "$LOG_DIR"

REQUIRED_DIMS=(3 4 8 16 32 64)

echo "=============================================="
echo "  Lang field grid (custom waves)"
echo "  RESULT_DIR        : $RESULT_DIR"
echo "  NUM_ITERS_SMALL   : $NUM_ITERS_SMALL  (D=3,4,8,16)"
echo "  NUM_ITERS_LARGE   : $NUM_ITERS_LARGE  (D=32,64)"
echo "  LOG_EVERY         : $LOG_EVERY"
echo "  DEVICE            : $DEVICE"
echo "  RENDER_CHECKPOINT : $RENDER_CHECKPOINT"
echo "  LOG_DIR (stdout)  : $LOG_DIR"
echo "=============================================="

_missing=()
for D in "${REQUIRED_DIMS[@]}"; do
    _fd="${RESULT_DIR}/language_features_dim${D}"
    if [ ! -d "$_fd" ]; then
        _missing+=("language_features_dim${D}")
    fi
done
if [ "${#_missing[@]}" -gt 0 ]; then
    echo "ERROR: autoencoder folders missing:"
    printf '  - %s\n' "${_missing[@]}"
    exit 1
fi

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

# Launch one job: D, level, num_iters → 03_train_gaussian_lang_field.sh
# loss_*.txt is written to output_dir inside train_language_field (see LangSplatam)
_run_one() {
    local D=$1
    local L=$2
    local niters=$3
    local out="${RESULT_DIR}/lang_field_${L}${D}"
    local log="${LOG_DIR}/stdout_${D}${L}.log"
    echo "[$(_ts)] START  D=${D} level=${L} iters=${niters} out=${out}"
    set +e
    # shellcheck disable=SC2086
    python scripts/train_language_field.py \
        --checkpoint         "$( _pick_ckpt )" \
        --poses              "${RESULT_DIR}/keyframe_poses.json" \
        --features_dir       "${RESULT_DIR}/language_features_dim${D}" \
        --level              "${L}" \
        --output_dir         "${out}" \
        --latent_dim         "${D}" \
        --num_iters          "${niters}" \
        --device             "${DEVICE}" \
        --render_checkpoint  "${RENDER_CHECKPOINT}" \
        --log_every          "${LOG_EVERY}" \
        --legacy \
        >"${log}" 2>&1
    local ec=$?
    set -e
    if [ "$ec" -eq 0 ]; then
        echo "[$(_ts)] OK     D=${D} level=${L}"
    else
        echo "[$(_ts)] FAIL   D=${D} level=${L} exit=$ec  →  ${log}"
    fi
    return "$ec"
}

_pick_ckpt() {
    local fd="${RESULT_DIR}/splatam/final"
    if [ -f "${fd}/params0.npz" ]; then echo "${fd}/params0.npz"
    elif [ -f "${fd}/params.npz" ]; then echo "${fd}/params.npz"
    else echo "${fd}/params0.npz"; fi
}

_wait_all() {
    local ec_all=0
    for pid in "$@"; do
        wait "$pid" || ec_all=1
    done
    return "$ec_all"
}

_failed=0

echo ""
echo "[$(_ts)] === Wave 1: 4s 4m 4l 8s 8m (parallel) ==="
pids=()
_run_one 4 s "$NUM_ITERS_SMALL" & pids+=($!)
_run_one 4 m "$NUM_ITERS_SMALL" & pids+=($!)
_run_one 4 l "$NUM_ITERS_SMALL" & pids+=($!)
_run_one 8 s "$NUM_ITERS_SMALL" & pids+=($!)
_run_one 8 m "$NUM_ITERS_SMALL" & pids+=($!)
_wait_all "${pids[@]}" || _failed=1

echo ""
echo "[$(_ts)] === Wave 2: 8l 16s 16m (parallel) ==="
pids=()
_run_one 8 l "$NUM_ITERS_SMALL" & pids+=($!)
_run_one 16 s "$NUM_ITERS_SMALL" & pids+=($!)
_run_one 16 m "$NUM_ITERS_SMALL" & pids+=($!)
_wait_all "${pids[@]}" || _failed=1

echo ""
echo "[$(_ts)] === Wave 3: 3s 3m 3l 16l (parallel) ==="
pids=()
_run_one 3 s "$NUM_ITERS_SMALL" & pids+=($!)
_run_one 3 m "$NUM_ITERS_SMALL" & pids+=($!)
_run_one 3 l "$NUM_ITERS_SMALL" & pids+=($!)
_run_one 16 l "$NUM_ITERS_SMALL" & pids+=($!)
_wait_all "${pids[@]}" || _failed=1

echo ""
echo "[$(_ts)] === Wave 4: 32m ==="
_run_one 32 m "$NUM_ITERS_LARGE" || _failed=1

echo ""
echo "[$(_ts)] === Wave 5: 64m ==="
_run_one 64 m "$NUM_ITERS_LARGE" || _failed=1

echo ""
if [ "$_failed" -eq 0 ]; then
    echo "[$(_ts)] Done. Loss files: \${RESULT_DIR}/lang_field_*/loss_*.txt ; stdout: $LOG_DIR"
else
    echo "[$(_ts)] Errors occurred — see $LOG_DIR/stdout_*.log"
    exit 1
fi
