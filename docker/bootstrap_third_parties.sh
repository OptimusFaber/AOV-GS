#!/bin/bash
# Symlink or clone third_parties expected by AOV-GS when repo is volume-mounted.
set -euo pipefail

ROOT="${1:-/workspace/AOV-GS}"
DEPS=/opt/aov-gs-deps
mkdir -p "${ROOT}/third_parties"

link_or_clone() {
  local name="$1"
  local src="$2"
  local dst="${ROOT}/third_parties/${name}"
  if [[ -d "${dst}/.git" ]] || [[ -f "${dst}/setup.py" ]] || [[ -d "${dst}" && -n "$(ls -A "${dst}" 2>/dev/null)" ]]; then
    return 0
  fi
  if [[ -d "${src}" ]]; then
    echo "[bootstrap] symlink ${dst} -> ${src}"
    ln -sfn "${src}" "${dst}"
  fi
}

link_or_clone habitat_sim "${DEPS}/habitat_sim"
link_or_clone splatam "${DEPS}/splatam"

# Optional: clone missing submodules from .gitmodules if git available
if [[ -f "${ROOT}/.gitmodules" ]] && command -v git >/dev/null; then
  for pair in coslam splatam neural_slam_eval; do
    dst="${ROOT}/third_parties/${pair}"
    [[ -d "${dst}" && -n "$(ls -A "${dst}" 2>/dev/null)" ]] && continue
    git -C "${ROOT}" submodule update --init "third_parties/${pair}" 2>/dev/null || true
  done
fi
