#!/usr/bin/env bash
# Fetch third_parties/ dependencies (Co-SLAM, SplaTAM, neural_slam_eval, channel rasterizers).
# habitat_sim is NOT included — install via conda (build_sem.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p third_parties

_ok() {
  local d="$1" marker="$2"
  [[ -f "${d}/${marker}" || -d "${d}/${marker}" ]]
}

echo "[setup_third_parties] AOV-GS root: ${ROOT}"

# Prefer git submodules when .git is present and links are registered.
if [[ -d .git ]] && [[ -f .gitmodules ]]; then
  echo "[setup_third_parties] git submodule update --init --recursive ..."
  git submodule update --init --recursive || true
fi

clone_repo() {
  local dest="$1" url="$2" branch="${3:-}"
  if _ok "$dest" "."; then
    echo "[setup_third_parties] OK  ${dest} (already present)"
    return 0
  fi
  echo "[setup_third_parties] clone ${url} -> ${dest}"
  mkdir -p "$(dirname "$dest")"
  if [[ -n "$branch" ]]; then
    git clone --depth 1 -b "$branch" "$url" "$dest"
  else
    git clone --depth 1 "$url" "$dest"
  fi
}

# Fallback: clone from .gitmodules URLs (works even without registered gitlinks).
clone_repo third_parties/coslam \
  https://github.com/HengyiWang/Co-SLAM.git
clone_repo third_parties/splatam \
  https://github.com/spla-tam/SplaTAM.git
clone_repo third_parties/neural_slam_eval \
  https://github.com/JingwenWang95/neural_slam_eval.git
clone_repo third_parties/channel_rasterization \
  https://github.com/lly00412/semantic-gaussians.git \
  liyan/dev
clone_repo third_parties/sparse_channel_rasterization \
  https://github.com/lly00412/semantic-gaussians.git \
  hairong/sparse_ver

missing=0
for pair in \
  "third_parties/coslam/utils.py" \
  "third_parties/splatam/utils/slam_external.py" \
  "third_parties/neural_slam_eval/eval_recon.py"; do
  if [[ ! -f "$pair" ]]; then
    echo "[setup_third_parties] MISSING ${pair}" >&2
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "[setup_third_parties] FAILED — check network and git, then re-run." >&2
  exit 1
fi

echo "[setup_third_parties] Done. (~200 MB source; habitat_sim via conda only.)"
