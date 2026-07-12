#!/usr/bin/env bash
##################################################
# Regenerate ScanNet NVS GT semantics (NYU40 labels via Habitat).
#
# Usage:
#   bash scripts/data/regenerate_scannet_nvs_semantics.sh scene0050_02
#
# Requires: docker/ensure_habitat_egl.sh, scene semantic.ply, NVS poses.
##################################################

set -eo pipefail

SCENE="${1:-scene0050_02}"
GPU="${GPU:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJ_DIR}"

set +u
if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /opt/conda/etc/profile.d/conda.sh
  conda activate aov-gs 2>/dev/null \
    || conda activate active-gs 2>/dev/null \
    || conda activate active-sgm 2>/dev/null \
    || true
fi
set -u

if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -x "${CONDA_PREFIX}/bin/python" ]]; then
  PY="${CONDA_PREFIX}/bin/python"
else
  PY=python3
fi

if [[ -f "${PROJ_DIR}/docker/ensure_habitat_egl.sh" ]]; then
  bash "${PROJ_DIR}/docker/ensure_habitat_egl.sh"
fi

(
  if [[ -f "${PROJ_DIR}/docker/nvidia_habitat_env.sh" ]]; then
    # shellcheck source=/dev/null
    source "${PROJ_DIR}/docker/nvidia_habitat_env.sh"
  fi
  export CUDA_VISIBLE_DEVICES="${GPU}"
  export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
  "${PY}" scripts/data/generate_scannet_semantics_from_poses.py \
    --scenes "${SCENE}" \
    --overwrite
)

echo "Check frame 0:"
"${PY}" - <<PY
import numpy as np
from pathlib import Path
p = Path("data/scannet_sim_nvs/${SCENE}/results_habitat/semantic/semantic000000.npy")
print("unique:", np.unique(np.load(p))[:25])
PY
