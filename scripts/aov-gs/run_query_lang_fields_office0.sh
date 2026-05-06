#!/usr/bin/env bash
##################################################
# Batch query evaluation for all trained language fields in a RESULT_DIR.
#
# Preferred (default) mode:
#   runs `scripts/activesgm/eval_run_lang_fields.py` which:
#   - picks the correct AE checkpoint by latent_dim
#   - runs `scripts/query_language_field.py` for each query
#   - writes `RESULT_DIR/lang_field_query_eval/summary.csv`
#
# Usage:
#   cd New-Proj
#   bash scripts/activesgm/run_query_lang_fields_office0.sh \
#     results/Replica/office0/ActiveOpenSem/run_0
#
# Env (optional):
#   DEVICE=cuda:0
#   SCENE=office0
#   POSE_SELECT=centroid|relevancy   (default centroid; cheaper)
#   TOP_PERCENTILE=2.0
#   DBSCAN_EPS=0.15
#   DBSCAN_MIN=30
#   TOP_K_VIEWS=3
#   OUT_DIR=...   (default: RESULT_DIR/lang_field_query_eval)
#   CKPT_ROOT=... (default: <repo>/ckpt ; expects CKPT_ROOT/<SCENE>/.../*.pth)
#   QUERIES="a sofa|a table|a window|the chair"  (pipe-separated)
#
# Legacy mode (kept for compatibility):
#   LEGACY=1 uses the old output layout under output/{DATASET}/{SCENE}/Field_{N}{SIZE}/...
#
# Queries are fixed to:
#   "a sofa" "a table" "a window" "the chair"
#
# AE checkpoints:
#   By default it will use: <repo>/ckpt/${SCENE}/.../*.pth
#   Your setup: /home/optimus/Desktop/Work/MIPT/Diploma/ACTIVE-SGM/New-Proj/ckpt/office0/*
##################################################

set -euo pipefail

RESULT_DIR=${1:?"Usage: $0 RESULT_DIR"}

if command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "[error] Neither 'python' nor 'python3' found in PATH."
  exit 127
fi

DEVICE=${DEVICE:-cuda:0}
SCENE=${SCENE:-office0}
POSE_SELECT=${POSE_SELECT:-centroid}
TOP_PERCENTILE=${TOP_PERCENTILE:-2.0}
DBSCAN_EPS=${DBSCAN_EPS:-0.15}
DBSCAN_MIN=${DBSCAN_MIN:-30}
TOP_K_VIEWS=${TOP_K_VIEWS:-3}
OUT_DIR=${OUT_DIR:-}
CKPT_ROOT=${CKPT_ROOT:-}
LEGACY=${LEGACY:-0}
QUERIES_ENV=${QUERIES:-}

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

if [ -n "$CKPT_ROOT" ]; then
  export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"
fi

_slug() {
  # lower + replace non-alnum with underscore
  "$PY" - "$@" <<'PY'
import re, sys
s=" ".join(sys.argv[1:]).strip().lower()
s=re.sub(r"[^a-z0-9]+","_",s).strip("_")
print(s or "q")
PY
}

_pick_ckpt() {
  local fd="${RESULT_DIR}/splatam/final"
  if [ -f "${fd}/params0.npz" ]; then echo "${fd}/params0.npz"
  elif [ -f "${fd}/params.npz" ]; then echo "${fd}/params.npz"
  else echo "${fd}/params0.npz"; fi
}

CHECKPOINT="$(_pick_ckpt)"
POSES=""
if [ -f "${RESULT_DIR}/keyframe_poses.json" ]; then
  POSES="--poses ${RESULT_DIR}/keyframe_poses.json"
fi

if [ -n "$QUERIES_ENV" ]; then
  IFS='|' read -r -a QUERIES <<<"$QUERIES_ENV"
else
  QUERIES=("a sofa" "a table" "a window" "the chair")
fi

echo "=============================================="
echo "Query all language fields"
echo "  RESULT_DIR     : $RESULT_DIR"
echo "  DEVICE         : $DEVICE"
echo "  CHECKPOINT     : $CHECKPOINT"
echo "  PYTHON         : $PY"
echo "  POSE_SELECT    : $POSE_SELECT"
echo "  POSES          : ${POSES:-<auto>}"
echo "  QUERIES        : ${QUERIES[*]}"
echo "  OUT_DIR        : ${OUT_DIR:-<default>}"
echo "  CKPT_ROOT      : ${CKPT_ROOT:-<repo>/ckpt}"
echo "  LEGACY         : $LEGACY"
echo "=============================================="

if [ "$LEGACY" -eq 1 ]; then
  echo "[legacy] Using old per-dim checkpoint layout under ckpt/${SCENE}/<D>/best_ckpt.pth"
  DATASET=${DATASET:-Replica}
  OUT_ROOT="${PROJ_DIR}/output/${DATASET}/${SCENE}"
  mkdir -p "$OUT_ROOT"

  SIZES=(s m l)
  DIMS=(3 4 8 16 32 64)
  failed=0

  for N in "${DIMS[@]}"; do
    AE_CKPT="ckpt/${SCENE}/${N}/best_ckpt.pth"
    if [ ! -f "$AE_CKPT" ]; then
      echo "[skip] missing AE checkpoint: $AE_CKPT"
      continue
    fi
    for SIZE in "${SIZES[@]}"; do
      LF_DIR="${RESULT_DIR}/lang_field_${SIZE}${N}"
      LF_PT="${LF_DIR}/lang_field.pt"
      if [ ! -f "$LF_PT" ]; then
        continue
      fi
      FIELD_OUT_DIR="${OUT_ROOT}/Field_${N}${SIZE}"
      mkdir -p "$FIELD_OUT_DIR"
      for Q in "${QUERIES[@]}"; do
        QSLUG="$(_slug "$Q")"
        OUT_PREFIX="${FIELD_OUT_DIR}/${QSLUG}"
        echo ""
        echo "==> Field_${N}${SIZE} | query='${Q}'"
        set +e
        # shellcheck disable=SC2086
        "$PY" scripts/query_language_field.py \
          --checkpoint "$CHECKPOINT" \
          --lang_field "$LF_PT" \
          --text "$Q" \
          --ae_ckpt "$AE_CKPT" \
          --device "$DEVICE" \
          --pose_select "$POSE_SELECT" \
          --top_percentile "$TOP_PERCENTILE" \
          --dbscan_eps "$DBSCAN_EPS" \
          --dbscan_min "$DBSCAN_MIN" \
          --top_k_views "$TOP_K_VIEWS" \
          ${POSES} \
          --out "$OUT_PREFIX"
        ec=$?
        set -e
        if [ "$ec" -ne 0 ]; then
          echo "[FAIL] Field_${N}${SIZE} query='${Q}' exit=$ec"
          failed=1
        fi
      done
    done
  done

  echo ""
  if [ "$failed" -eq 0 ]; then
    echo "Done. Outputs → $OUT_ROOT"
  else
    echo "Done with failures. Check stdout above. Outputs (partial) → $OUT_ROOT"
    exit 1
  fi
  exit 0
fi

# Preferred path: python batch evaluator + summary.csv
EVAL_CMD=("$PY" scripts/activesgm/eval_run_lang_fields.py
  --result_dir "$RESULT_DIR"
  --scene_name "$SCENE"
  --device "$DEVICE"
  --pose_select "$POSE_SELECT"
  --top_percentile "$TOP_PERCENTILE"
  --dbscan_eps "$DBSCAN_EPS"
  --dbscan_min "$DBSCAN_MIN"
  --top_k_views "$TOP_K_VIEWS"
  --queries "${QUERIES[@]}"
)

if [ -n "$OUT_DIR" ]; then
  EVAL_CMD+=(--out_dir "$OUT_DIR")
fi

if [ -n "$CKPT_ROOT" ]; then
  # eval_run_lang_fields.py expects ckpt under <repo>/ckpt/<scene>/...
  # We support an override by symlinking via an env var that the python script reads
  # (implemented inside eval_run_lang_fields.py). If not present there, it will just
  # fall back to repo ckpt dir.
  export ACTIVESGM_CKPT_ROOT="$CKPT_ROOT"
fi

set +e
# shellcheck disable=SC2086
"${EVAL_CMD[@]}" ${POSES}
ec=$?
set -e

if [ "$ec" -ne 0 ]; then
  echo "Done with failures (exit=$ec). Check stdout above."
  exit "$ec"
fi

echo "Done. Outputs → ${OUT_DIR:-${RESULT_DIR}/lang_field_query_eval}"
echo "Summary CSV → ${OUT_DIR:-${RESULT_DIR}/lang_field_query_eval}/summary.csv"

