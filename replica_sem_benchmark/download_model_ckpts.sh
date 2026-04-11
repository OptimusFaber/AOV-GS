#!/usr/bin/env bash
# Download SAM1, SAM2 (Hugging Face snapshots), optional SAM3, and warm OpenCLIP cache.
# Canonical layout::
#   $ROOT/sam/<model>/            one folder per HF model (config.json)
#   $ROOT/sam/sam1/*.pth          segment_anything
#   $ROOT/clip/open_clip/         OPENCLIP_CACHE
#   $ROOT/clip/hub/               HF_HOME
#
# Usage:
#   ./replica_sem_benchmark/download_model_ckpts.sh
#   ROOT=/other/path ./replica_sem_benchmark/download_model_ckpts.sh
#   ./replica_sem_benchmark/download_model_ckpts.sh --with-sam3    # after HF login + license
#   ./replica_sem_benchmark/download_model_ckpts.sh --with-sam2-1
#
# Requirements: hf or huggingface-cli (pip install -U "huggingface_hub[cli]"), wget, python3.
#   Prefer ``hf download`` (new); script falls back to ``huggingface-cli download``.
#
# Gated models (facebook/sam3, facebook/sam3.1): open the model page on huggingface.co,
# accept Meta's terms, then:
#   hf auth login   # or: huggingface-cli login
#   ./download_model_ckpts.sh --with-sam3
# Without login you get HTTP 401 / GatedRepoError — this script skips and removes partial dirs.

set -euo pipefail

ROOT="${ROOT:-/mnt/data/model-ckpts}"
# Default: do NOT fetch SAM3 (gated + large). Use --with-sam3 after `huggingface-cli login`.
WITH_SAM3="${WITH_SAM3:-0}"
WITH_SAM2_1="${WITH_SAM2_1:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-sam3)     WITH_SAM3=0; shift ;;
    --with-sam3)   WITH_SAM3=1; shift ;;
    --with-sam2-1) WITH_SAM2_1=1; shift ;;
    --no-sam2-1)   WITH_SAM2_1=0; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "$ROOT/sam/sam1" "$ROOT/clip/open_clip" "$ROOT/clip/huggingface_hub/hub"

echo "=== Root: $ROOT ==="

# ── SAM1 (segment_anything) ───────────────────────────────────────────────────
# Avoid ``wget -c`` on an already-complete file — many CDNs return HTTP 416.
SAM1_PTH="$ROOT/sam/sam1/sam_vit_b_01ec64.pth"
SAM1_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM1_MIN_BYTES=50000000   # ~50 MiB floor (full ViT-B ~375 MiB)
echo "=== SAM1 ViT-B ==="
if [[ -f "$SAM1_PTH" ]] && [[ $(wc -c < "$SAM1_PTH") -gt $SAM1_MIN_BYTES ]]; then
  echo "SAM1 already present ($(du -h "$SAM1_PTH" | cut -f1)), skip download"
else
  wget -O "$SAM1_PTH" "$SAM1_URL"
fi

# ── SAM2 / SAM3: full Hugging Face snapshots (for transformers mask-generation) ─
# sam2_video / Sam2VideoModel needs transformers>=4.56 at runtime.

SAM2_REPOS=(
  facebook/sam2-hiera-tiny
  facebook/sam2-hiera-small
  facebook/sam2-hiera-base-plus
  facebook/sam2-hiera-large
)

if [[ "$WITH_SAM2_1" == "1" ]]; then
  SAM2_REPOS+=(
    facebook/sam2.1-hiera-tiny
    facebook/sam2.1-hiera-small
    facebook/sam2.1-hiera-base-plus
    facebook/sam2.1-hiera-large
  )
fi

if ! command -v hf &>/dev/null && ! command -v huggingface-cli &>/dev/null; then
  echo "Install: pip install -U 'huggingface_hub[cli]'  (provides: hf, huggingface-cli)"
  exit 1
fi

# Quieter Hub client; tracebacks on stderr are still possible on failure — we clean partial dirs.
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

download_repo() {
  local repo="$1"
  local name="${repo##*/}"
  local dst="$ROOT/sam/$name"
  local log
  log=$(mktemp)
  if command -v hf &>/dev/null; then
    echo "=== hf download $repo -> $dst ==="
  else
    echo "=== huggingface-cli download $repo -> $dst ==="
  fi
  # Remove stale partial snapshot from a previous failed run (e.g. 401 after LICENSE only).
  rm -rf "$dst"
  set +e
  local rc
  # ``hf download`` (new CLI) does NOT support ``--local-dir-use-symlinks`` — only
  # ``huggingface-cli download`` does. Use plain ``--local-dir`` for ``hf``.
  if command -v hf &>/dev/null; then
    hf download "$repo" --local-dir "$dst" 2>"$log"
    rc=$?
    if [[ $rc -ne 0 ]] && command -v huggingface-cli &>/dev/null; then
      echo "  (hf failed, retrying with huggingface-cli download …)"
      rm -rf "$dst"
      huggingface-cli download "$repo" \
        --local-dir "$dst" \
        --local-dir-use-symlinks False \
        2>"$log"
      rc=$?
    fi
  else
    huggingface-cli download "$repo" \
      --local-dir "$dst" \
      --local-dir-use-symlinks False \
      2>"$log"
    rc=$?
  fi
  set -e
  if [[ $rc -eq 0 ]] && [[ -f "$dst/config.json" ]]; then
    echo "OK: $name"
    rm -f "$log"
    return 0
  fi
  echo "SKIP: $repo (exit $rc). If 401: gated repo → hf auth login + accept license on huggingface.co"
  if [[ -s "$log" ]]; then
    echo "---- (last lines of stderr) ----"
    tail -n 8 "$log" | sed 's/^/  /'
    echo "--------------------------------"
  fi
  rm -f "$log"
  rm -rf "$dst"
  return 0
}

for repo in "${SAM2_REPOS[@]}"; do
  download_repo "$repo"
done

if [[ "$WITH_SAM3" == "1" ]]; then
  echo "=== SAM3 (gated — requires hf auth login + license on the model page) ==="
  for repo in facebook/sam3 facebook/sam3.1; do
    download_repo "$repo"
  done
else
  echo "=== SAM3 skipped (default). To download after HF login:  --with-sam3 ==="
fi

# ── OpenCLIP pretrained cache ─────────────────────────────────────────────────
export OPENCLIP_CACHE="$ROOT/clip/open_clip"
export HF_HOME="${HF_HOME:-$ROOT/clip/hub}"
mkdir -p "$OPENCLIP_CACHE" "$HF_HOME"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=== OpenCLIP cache warm-up (same list as eval_clip_sam_systematic) ==="
if [[ -f "$SCRIPT_DIR/download_openclip_weights.py" ]]; then
  ROOT="$ROOT" python3 "$SCRIPT_DIR/download_openclip_weights.py"
else
  echo "  (download_openclip_weights.py missing — skip)"
fi

echo ""
echo "Done. HF snapshots for eval:"
echo "  python replica_sem_benchmark/eval_clip_sam_systematic.py ... \\"
echo "    --sam_models preset:hf_local_no_sam3   # sam1/*.pth + HF; skip SAM3* dirs"
echo "  or --sam_models preset:hf_local          # sam1/*.pth + all HF under $ROOT/sam/"
echo ""
echo "SAM1 default path is detected automatically if present:"
echo "  $ROOT/sam/sam1/sam_vit_b_01ec64.pth"
echo ""
echo "HF SAM2/SAM3 need: pip install -U 'transformers>=4.56.0'"
