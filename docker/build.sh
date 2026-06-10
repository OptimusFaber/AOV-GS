#!/bin/bash
# Build AOV-GS Docker image.
#
#   bash docker/build.sh
#   HTTP_PROXY=... HTTPS_PROXY=... bash docker/build.sh
#   BUILD_ACTIVESEM=0 bash docker/build.sh   # skip ActiveSem channel rasterizers

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TAG="${TAG:-aov-gs:cuda117}"

BUILD_ARGS=()
[[ -n "${HTTP_PROXY:-}" ]] && BUILD_ARGS+=(--build-arg "HTTP_PROXY=${HTTP_PROXY}")
[[ -n "${HTTPS_PROXY:-}" ]] && BUILD_ARGS+=(--build-arg "HTTPS_PROXY=${HTTPS_PROXY}")
BUILD_ARGS+=(--build-arg "BUILD_ACTIVESEM=${BUILD_ACTIVESEM:-1}")

echo "Building ${TAG} from ${ROOT}"
docker build -f "${ROOT}/docker/Dockerfile" -t "${TAG}" "${BUILD_ARGS[@]}" "${ROOT}"

echo "Done: ${TAG}"
echo "Run: bash docker/run.sh"
