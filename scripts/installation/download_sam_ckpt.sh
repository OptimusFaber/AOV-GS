#!/usr/bin/env bash
# Download SAM ViT-B checkpoint into ckpts/ (~375 MB).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_DIR="${ROOT}/ckpts"
CKPT="${CKPT_DIR}/sam_vit_b_01ec64.pth"
URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

mkdir -p "$CKPT_DIR"

if [[ -f "$CKPT" ]]; then
  echo "[download_sam_ckpt] Already exists: ${CKPT}"
  exit 0
fi

echo "[download_sam_ckpt] Downloading ~375 MB -> ${CKPT}"
wget --continue -O "$CKPT" "$URL"
echo "[download_sam_ckpt] Done."
