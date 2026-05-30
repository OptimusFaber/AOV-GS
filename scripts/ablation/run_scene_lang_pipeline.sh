#!/bin/bash
##################################################
# Single scene: SLAM exploration → lang field train → traj validation.
#
# Usage:
#   bash scripts/ablation/run_scene_lang_pipeline.sh SCENE [EXP] [OPTIONS...]
#
# Examples:
#   bash scripts/ablation/run_scene_lang_pipeline.sh office0
#   bash scripts/ablation/run_scene_lang_pipeline.sh office0 ActiveOpenSem
#   TRAIN_DOWNSCALE=0.25 LEVELS=s bash scripts/ablation/run_scene_lang_pipeline.sh room0
#
# Env:
#   TRAIN_DOWNSCALE=0.5   render downscale during lang-field training
#   CODEBOOK_K=64         LangSplatV2 codebook size
#   LEVELS=all            SAM levels: s, m, l, or all (default s)
#   NUM_ITERS=30000
#   HYBRID_K=8            only for ActiveOpenSem
#   PARALLEL_LANG=auto    auto|0|1 — parallelize s/m/l on free GPUs
#   SEED=0
#   GPU=0,1
#   SLAM_DEVICE=cuda:0  SEM_DEVICE=cuda:1   override segmenter / SLAM GPU
#
# Output:
#   results/Replica/<scene>/<EXP>/run_N/
#   lang_field_*k*_l*/lang_field.pt
#   lang_field_traj_eval/miou_summary.txt, miou_per_class.csv
#   results/Replica/ablation_<scene>_lang.csv
##################################################

set -euo pipefail

SCENE="${1:?Usage: $0 SCENE [EXP]}"
EXP="${2:-ActiveOpenSem}"
shift 2 || true

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
SEED="${SEED:-0}"
ENABLE_VIS="${ENABLE_VIS:-0}"
TRAIN_DOWNSCALE="${TRAIN_DOWNSCALE:-0.5}"
CODEBOOK_K="${CODEBOOK_K:-64}"
NUM_ITERS="${NUM_ITERS:-30000}"
LEVELS="${LEVELS:-s}"
PARALLEL_LANG="${PARALLEL_LANG:-auto}"
HYBRID_K="${HYBRID_K:-8}"

if [[ "$EXP" == "ActiveOpenSem" ]]; then
  python3 scripts/ablation/gen_open_sem_scene_configs.py
  python3 <<PY
import sys
sys.path.insert(0, "scripts/ablation")
from benchmark_utils import patch_hybrid_k
patch_hybrid_k("${SCENE}", ${HYBRID_K})
PY
fi

BASE="results/Replica/${SCENE}/${EXP}"
mkdir -p "$BASE"
RUN_ID=0
while [[ -d "${BASE}/run_${RUN_ID}" ]]; do
  RUN_ID=$((RUN_ID + 1))
done
RESULT_DIR="${BASE}/run_${RUN_ID}"
mkdir -p "$RESULT_DIR"

echo "=============================================="
echo "  Scene         : $SCENE"
echo "  EXP           : $EXP"
echo "  Result dir    : $RESULT_DIR"
echo "  Downscale     : $TRAIN_DOWNSCALE"
echo "  Levels        : $LEVELS"
echo "  Codebook K    : $CODEBOOK_K"
echo "=============================================="

# ── Stage 1: SLAM + SAM/CLIP ─────────────────────────────────────────────
bash scripts/ablation/run_with_metrics.sh slam "$RESULT_DIR" \
  python src/main/activesgm.py \
    --cfg "configs/Replica/${SCENE}/${EXP}.py" \
    --seed "$SEED" \
    --result_dir "$RESULT_DIR" \
    --enable_vis "$ENABLE_VIS" \
    --corrclip 0 \
    ${SLAM_DEVICE:+--slam_device "$SLAM_DEVICE"} \
    ${SEM_DEVICE:+--seg_device "$SEM_DEVICE"}

# ── Stage 2: validate raw features (LangSplatV2 no-op) ─────────────────────
bash scripts/aov-gs/02_train_clip_autoencoder_v2_stub.sh "$RESULT_DIR"

# ── Stage 3: lang field training ───────────────────────────────────────────
pick_levels() {
  case "$LEVELS" in
    all) echo "s m l" ;;
    *) echo "$LEVELS" ;;
  esac
}

mapfile -t GPUS < <(python3 -c "
import sys
sys.path.insert(0, 'scripts/ablation')
from benchmark_utils import available_gpus
print('\n'.join(available_gpus(6000)))
")

train_one_level() {
  local lvl="$1"
  local dev="$2"
  bash scripts/ablation/run_with_metrics.sh "lang_field_${lvl}" "$RESULT_DIR" \
    bash scripts/aov-gs/03_train_gaussian_lang_field_v2.sh \
      "$RESULT_DIR" "$CODEBOOK_K" "$lvl" "$NUM_ITERS" "$dev" 1 4 auto "$TRAIN_DOWNSCALE"
}

LEVEL_LIST=($(pick_levels))
N_LEVELS=${#LEVEL_LIST[@]}

if [[ "$PARALLEL_LANG" == "auto" ]] && [[ ${#GPUS[@]} -ge $N_LEVELS ]] && [[ $N_LEVELS -gt 1 ]]; then
  echo "[$(date -Is)] Parallel lang-field training on ${#GPUS[@]} GPU(s)"
  PIDS=()
  for i in "${!LEVEL_LIST[@]}"; do
    lvl="${LEVEL_LIST[$i]}"
    dev="${GPUS[$i]}"
    ( train_one_level "$lvl" "$dev" ) &
    PIDS+=($!)
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
else
  echo "[$(date -Is)] Sequential lang-field training"
  dev="${GPUS[0]:-cuda:0}"
  for lvl in "${LEVEL_LIST[@]}"; do
    train_one_level "$lvl" "$dev"
  done
fi

# ── Stage 4: validation on NVS traj ────────────────────────────────────────
LANG_DIR=""
for lvl in "${LEVEL_LIST[@]}"; do
  cand="${RESULT_DIR}/lang_field_${lvl}k${CODEBOOK_K}_l1"
  if [[ -d "$cand" ]] && [[ -f "${cand}/lang_field.pt" ]]; then
    LANG_DIR="$cand"
  fi
done
if [[ -z "$LANG_DIR" ]]; then
  LANG_DIR=$(find "$RESULT_DIR" -maxdepth 1 -type d -name 'lang_field_*' | head -1)
fi

if [[ -z "$LANG_DIR" ]] || [[ ! -f "${LANG_DIR}/lang_field.pt" ]]; then
  echo "ERROR: lang_field.pt not found under $RESULT_DIR"
  exit 1
fi

VAL_OUT="${RESULT_DIR}/lang_field_traj_eval"
mkdir -p "$VAL_OUT"

bash scripts/ablation/run_with_metrics.sh validate "$RESULT_DIR" \
  python scripts/validate_lang_field_traj.py \
    --scene "$SCENE" \
    --result_dir "$RESULT_DIR" \
    --traj_txt "data/replica_sim_nvs/${SCENE}/traj.txt" \
    --align_gs_train_frame \
    --levels "$LEVELS" \
    --codebook_size "$CODEBOOK_K" \
    --out_dir "$VAL_OUT"

# ── Per-class table ────────────────────────────────────────────────────────
ABLATION_CSV="results/Replica/ablation_${SCENE}_lang.csv"
python3 <<PY
import csv
from pathlib import Path

result = Path("${RESULT_DIR}")
val = result / "lang_field_traj_eval"
miou_txt = val / "miou_summary.txt"
per_class = val / "miou_per_class.csv"
out = Path("${ABLATION_CSV}")

rows = []
overall = ""
if miou_txt.exists():
    for line in miou_txt.read_text(encoding="utf-8").splitlines():
        if line.startswith("Overall mIoU"):
            overall = f"{float(line.split(':')[1].strip()) * 100:.2f}"

import json
metrics = {}
mp = result / "ablation_metrics.json"
if mp.exists():
    metrics = json.loads(mp.read_text(encoding="utf-8"))

stages = metrics.get("stages", {})
def t(name):
    s = stages.get(name, {}).get("wall_s")
    return f"{s/3600:.2f}" if s else "—"
def v(name):
    x = stages.get(name, {}).get("vram_peak_mb")
    return f"{x:.0f}" if x else "—"

header = ["scene", "exp", "run", "overall_miou_pct", "time_slam_h", "time_lang_h", "time_validate_h",
          "vram_slam_mb", "vram_lang_mb", "num_gaussians", "train_downscale", "lang_field_dir"]
row = {
    "scene": "${SCENE}",
    "exp": "${EXP}",
    "run": result.name,
    "overall_miou_pct": overall,
    "time_slam_h": t("slam"),
    "time_lang_h": t("lang_field_s") if t("lang_field_s") != "—" else t("lang_field"),
    "time_validate_h": t("validate"),
    "vram_slam_mb": v("slam"),
    "vram_lang_mb": v("lang_field_s") if v("lang_field_s") != "—" else v("lang_field"),
    "num_gaussians": str(metrics.get("num_gaussians", "—")),
    "train_downscale": "${TRAIN_DOWNSCALE}",
    "lang_field_dir": "${LANG_DIR}",
}
rows.append(row)

if per_class.exists():
    with per_class.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for pr in reader:
            rows.append({**row, "class": pr.get("class", ""), "class_miou_pct": pr.get("miou_pct", pr.get("miou", ""))})

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    fields = list(header)
    if rows and "class" in rows[-1]:
        fields += ["class", "class_miou_pct"]
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows[:1])
    if per_class.exists():
        with per_class.open(encoding="utf-8") as pf:
            r = csv.DictReader(pf)
            w.writerow({})
            w.writerow({"scene": "per-class mIoU"})
            for pr in r:
                w.writerow({"class": pr.get("class", ""), "class_miou_pct": pr.get("miou_pct", pr.get("miou", ""))})

print(f"Wrote {out}")
if miou_txt.exists():
    print(miou_txt.read_text(encoding="utf-8"))
PY

echo ""
echo "[$(date -Is)] Pipeline complete: $RESULT_DIR"
echo "  Summary CSV: $ABLATION_CSV"
