#!/usr/bin/env bash
# One-shot setup: third_parties + conda env + pip extras + SAM + Replica data.
# For pipeline / GPU / Docker see README.md and linked sub-guides.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SKIP_ENV=0
SKIP_DATA=0
SKIP_SAM=0
ASSUME_YES=0

usage() {
  cat <<'EOF'
Usage: bash scripts/installation/quick_start.sh [OPTIONS]

  Installs everything needed before running the SLAM / open-vocab pipeline.

  Disk space (rough estimate):
    conda env (active-sgm)     ~12 GB
    Replica meshes (v1)        ~2 GB
    ReplicaSLAM + Habitat/NVS  ~5–15 GB (depends on scenes)
    SAM checkpoint             ~0.4 GB
    third_parties (git clone)  ~0.2 GB
    ─────────────────────────────────
    plan for ≥ 30 GB free on disk

Options:
  --yes              skip confirmation prompt
  --skip-env         skip conda build (env must already exist)
  --skip-data        skip Replica / ReplicaSLAM / Habitat / NVS download
  --skip-sam         skip SAM checkpoint download
  -h, --help         this message

After completion:
  conda activate active-sgm
  See scripts/aov-gs/README.md to run the pipeline.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes) ASSUME_YES=1; shift ;;
    --skip-env) SKIP_ENV=1; shift ;;
    --skip-data) SKIP_DATA=1; shift ;;
    --skip-sam) SKIP_SAM=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

echo "=============================================="
echo "  AOV-GS — quick start"
echo "  Repo: https://github.com/OptimusFaber/AOV-GS"
echo "=============================================="
echo ""
echo "Estimated disk usage:"
echo "  conda env              ~12 GB"
echo "  Replica + derived data ~7–17 GB"
echo "  SAM + third_parties    ~0.6 GB"
echo "  Recommended free space ≥ 30 GB"
echo ""

if [[ "$ASSUME_YES" -ne 1 ]]; then
  read -r -p "Continue? [y/N] " ans
  [[ "${ans,,}" == "y" || "${ans,,}" == "yes" ]] || { echo "Aborted."; exit 0; }
fi

echo ""
echo "=== 1/5 third_parties ==="
bash scripts/installation/setup_third_parties.sh

echo ""
echo "=== 2/5 conda environment ==="
if [[ "$SKIP_ENV" -eq 1 ]]; then
  echo "[skip] --skip-env"
else
  if command -v conda >/dev/null 2>&1 && conda env list | grep -qE '(^| )active-sgm( |$)'; then
    echo "[quick_start] conda env active-sgm already exists — skipping build_sem.sh"
    echo "              Re-run build manually if needed: bash scripts/installation/conda_env/build_sem.sh"
  else
    bash scripts/installation/conda_env/build_sem.sh
  fi
fi

eval "$(conda shell.bash hook)"
conda activate active-sgm

echo ""
echo "=== 3/5 pip packages (open-vocab) ==="
pip install -q open_clip_torch scikit-learn tqdm
pip install -q git+https://github.com/facebookresearch/segment-anything.git

echo ""
echo "=== 4/5 SAM checkpoint ==="
if [[ "$SKIP_SAM" -eq 1 ]]; then
  echo "[skip] --skip-sam"
else
  bash scripts/installation/download_sam_ckpt.sh
fi

echo ""
echo "=== 5/5 Replica data ==="
if [[ "$SKIP_DATA" -eq 1 ]]; then
  echo "[skip] --skip-data"
else
  bash scripts/data/replica_download.sh data/replica_v1
  bash scripts/data/replica_update.sh data/replica_v1
  bash scripts/data/replica_slam_download.sh
  bash scripts/data/generate_replica_habitat.sh all
  bash scripts/data/generate_replica_nvs.sh all
fi

echo ""
echo "=============================================="
echo "  Quick start finished."
echo "  Next: conda activate active-sgm"
echo "  Pipeline: scripts/aov-gs/README.md"
echo "=============================================="
