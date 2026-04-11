#!/usr/bin/env bash
# Normalize checkpoint tree to::
#
#   $ROOT/sam/<model_name>/     HF snapshots (config.json) + sam1/ for segment_anything .pth
#   $ROOT/clip/open_clip/       OPENCLIP_CACHE
#   $ROOT/clip/hub/             HF_HOME
#
# Removes: legacy symlinks at $ROOT, bucket ``sam/hf/``, empty ``sam/native/``, top-level ``hub/``
# after moving data. Re-run safe.
#
# Usage:
#   ./replica_sem_benchmark/organize_model_ckpts.sh
#   ROOT=/mnt/data/model-ckpts ./replica_sem_benchmark/organize_model_ckpts.sh
#
set -euo pipefail

ROOT="${ROOT:-/mnt/data/model-ckpts}"

echo "=== Normalize checkpoints: $ROOT ==="

mkdir -p "$ROOT/sam" "$ROOT/clip/open_clip" "$ROOT/clip/hub"

# ── Remove legacy symlinks at root (keep only sam/ and clip/) ─────────────────
for name in sam1 hf sam2_pt open_clip huggingface; do
  if [[ -L "$ROOT/$name" ]]; then
    echo "  rm symlink $ROOT/$name"
    rm -f "$ROOT/$name"
  fi
done

# ── sam1 at repo root → sam/sam1 ────────────────────────────────────────────
if [[ -d "$ROOT/sam1" ]] && [[ ! -L "$ROOT/sam1" ]]; then
  if [[ ! -d "$ROOT/sam/sam1" ]]; then
    echo "  mv sam1/ → sam/sam1/"
    mv "$ROOT/sam1" "$ROOT/sam/sam1"
  else
    echo "  WARN: sam/sam1 exists; leave root sam1/ in place — merge manually"
  fi
fi

# ── Flatten sam/hf/<model>/ → sam/<model>/ ───────────────────────────────────
if [[ -d "$ROOT/sam/hf" ]] && [[ ! -L "$ROOT/sam/hf" ]]; then
  shopt -s nullglob
  for d in "$ROOT/sam/hf"/*; do
    [[ -d "$d" ]] || continue
    base=$(basename "$d")
    if [[ -e "$ROOT/sam/$base" ]]; then
      echo "  WARN: skip flatten $base (already exists under sam/)"
      continue
    fi
    echo "  mv sam/hf/$base → sam/$base"
    mv "$d" "$ROOT/sam/$base"
  done
  shopt -u nullglob
  if [[ -z "$(ls -A "$ROOT/sam/hf" 2>/dev/null)" ]]; then
    echo "  rmdir sam/hf"
    rmdir "$ROOT/sam/hf" 2>/dev/null || true
  fi
fi

# ── Top-level hf/<model>/ → sam/<model>/ ─────────────────────────────────────
if [[ -d "$ROOT/hf" ]] && [[ ! -L "$ROOT/hf" ]]; then
  shopt -s nullglob
  for d in "$ROOT/hf"/*; do
    [[ -d "$d" ]] || continue
    base=$(basename "$d")
    if [[ -e "$ROOT/sam/$base" ]]; then
      echo "  WARN: skip hf/$base (exists under sam/)"
      continue
    fi
    echo "  mv hf/$base → sam/$base"
    mv "$d" "$ROOT/sam/$base"
  done
  shopt -u nullglob
  if [[ -z "$(ls -A "$ROOT/hf" 2>/dev/null)" ]]; then
    echo "  rmdir hf"
    rmdir "$ROOT/hf" 2>/dev/null || true
  fi
fi

# ── sam/native/*.pt → sam/sam2-hiera-tiny/ if that folder exists (else sam/sam2-native-pt/) ─
if [[ -d "$ROOT/sam/native" ]] && [[ ! -L "$ROOT/sam/native" ]]; then
  tgt="$ROOT/sam/sam2-hiera-tiny"
  if [[ ! -d "$tgt" ]]; then
    tgt="$ROOT/sam/sam2-native-pt"
    mkdir -p "$tgt"
  fi
  shopt -s nullglob
  for f in "$ROOT/sam/native"/*; do
    [[ -f "$f" ]] || continue
    b=$(basename "$f")
    if [[ ! -e "$tgt/$b" ]]; then
      echo "  mv sam/native/$b → $(basename "$tgt")/"
      mv "$f" "$tgt/"
    fi
  done
  shopt -u nullglob
  if [[ -z "$(ls -A "$ROOT/sam/native" 2>/dev/null)" ]]; then
    echo "  rmdir sam/native"
    rmdir "$ROOT/sam/native" 2>/dev/null || true
  fi
fi

# ── hub / huggingface → clip/hub ────────────────────────────────────────────
if [[ -d "$ROOT/hub" ]] && [[ ! -L "$ROOT/hub" ]] && [[ "$ROOT/hub" != "$ROOT/clip/hub" ]]; then
  shopt -s dotglob nullglob
  for x in "$ROOT/hub"/*; do
    [[ -e "$x" ]] || continue
    base=$(basename "$x")
    if [[ ! -e "$ROOT/clip/hub/$base" ]]; then
      echo "  mv hub/$base → clip/hub/"
      mv "$x" "$ROOT/clip/hub/"
    fi
  done
  shopt -u dotglob nullglob
  if [[ -z "$(ls -A "$ROOT/hub" 2>/dev/null)" ]]; then
    echo "  rmdir hub"
    rmdir "$ROOT/hub" 2>/dev/null || true
  fi
fi

if [[ -d "$ROOT/huggingface" ]] && [[ ! -L "$ROOT/huggingface" ]]; then
  shopt -s dotglob nullglob
  for x in "$ROOT/huggingface"/*; do
    [[ -e "$x" ]] || continue
    base=$(basename "$x")
    if [[ ! -e "$ROOT/clip/hub/$base" ]]; then
      echo "  mv huggingface/$base → clip/hub/"
      mv "$x" "$ROOT/clip/hub/"
    fi
  done
  shopt -u dotglob nullglob
  if [[ -z "$(ls -A "$ROOT/huggingface" 2>/dev/null)" ]]; then
    rmdir "$ROOT/huggingface" 2>/dev/null || true
  fi
fi

# ── open_clip at root → clip/open_clip ───────────────────────────────────────
if [[ -d "$ROOT/open_clip" ]] && [[ ! -L "$ROOT/open_clip" ]] && [[ "$ROOT/open_clip" != "$ROOT/clip/open_clip" ]]; then
  shopt -s dotglob nullglob
  for x in "$ROOT/open_clip"/*; do
    [[ -e "$x" ]] || continue
    base=$(basename "$x")
    if [[ ! -e "$ROOT/clip/open_clip/$base" ]]; then
      mv "$x" "$ROOT/clip/open_clip/"
    fi
  done
  shopt -u dotglob nullglob
  if [[ -z "$(ls -A "$ROOT/open_clip" 2>/dev/null)" ]]; then
    rmdir "$ROOT/open_clip" 2>/dev/null || true
  fi
fi

echo ""
echo "=== Result (max depth 3) ==="
if command -v tree &>/dev/null; then
  tree -L 3 -d "$ROOT" 2>/dev/null || ls -la "$ROOT"
else
  find "$ROOT" -maxdepth 3 -type d | head -100
fi

echo ""
echo "Exports for shells / jobs:"
echo "  export OPENCLIP_CACHE=$ROOT/clip/open_clip"
echo "  export HF_HOME=$ROOT/clip/hub"
