#!/bin/bash
# Interactive shell with GPU.
#
#   bash docker/run.sh
#   bash docker/run.sh python src/main/activesgm.py --cfg configs/Replica/office0/ActiveOpenSem.py ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TAG="${TAG:-aov-gs:cuda117}"

docker run --rm -it \
  --gpus all \
  --shm-size=16g \
  -v "${ROOT}:/workspace/AOV-GS" \
  -v aov-gs-data:/workspace/AOV-GS/data \
  -v aov-gs-results:/workspace/AOV-GS/results \
  -v aov-gs-ckpts:/workspace/AOV-GS/ckpts \
  -w /workspace/AOV-GS \
  "${TAG}" \
  "$@"
