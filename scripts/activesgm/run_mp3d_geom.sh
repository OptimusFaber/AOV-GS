#!/bin/bash
# MP3D geometry-only: SplaTAM + active_gs (ActiveOpenSemGeom → results/.../ActiveGeom/).
# Usage: bash scripts/activesgm/run_mp3d_geom.sh [SCENE] [NUM_RUNS] [ENABLE_VIS] [GPU_ID]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "${SCRIPT_DIR}/run_mp3d.sh" "${1:-GdvgFV5R1Z5}" "${2:-1}" ActiveOpenSemGeom "${3:-0}" "${4:-0}"
