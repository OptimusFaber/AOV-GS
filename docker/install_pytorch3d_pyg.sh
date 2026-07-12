#!/bin/bash
# Install pytorch3d + torch-scatter/sparse inside aov-gs env.
# Uses local docker/wheels/*.whl if present, else tries direct URLs.
# If pytorch3d wheel is blocked (e.g. 403), falls back to source build.
set -euo pipefail

WHEEL_DIR="${1:-/tmp/wheels}"
mkdir -p "${WHEEL_DIR}"

PY3="${WHEEL_DIR}/pytorch3d-0.7.4-cp38-cp38-linux_x86_64.whl"
SCAT="${WHEEL_DIR}/torch_scatter-2.1.1+pt113cu117-cp38-cp38-linux_x86_64.whl"
SPRS="${WHEEL_DIR}/torch_sparse-0.6.17+pt113cu117-cp38-cp38-linux_x86_64.whl"

fetch() {
  local url="$1"
  local out="$2"
  if [[ -s "${out}" ]]; then
    echo "[wheels] already have ${out}"
    return 0
  fi
  echo "[wheels] downloading ${url}"
  wget -c --tries=8 --timeout=180 --waitretry=10 -O "${out}" "${url}"
}

try_fetch() {
  local url="$1"
  local out="$2"
  if fetch "${url}" "${out}"; then
    return 0
  fi
  echo "[wheels] wget failed, retry with curl: ${url}"
  curl -fL -A "Mozilla/5.0" -C - --retry 6 --retry-delay 5 --connect-timeout 30 \
    "${url}" -o "${out}"
}

pip install --no-cache-dir fvcore iopath

PY3_READY=0
if [[ -s "${PY3}" ]]; then
  PY3_READY=1
elif try_fetch "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/pytorch3d-0.7.4-cp38-cp38-linux_x86_64.whl" "${PY3}"; then
  PY3_READY=1
fi

SCAT_READY=0
if [[ -s "${SCAT}" ]]; then
  SCAT_READY=1
elif try_fetch "https://data.pyg.org/whl/torch-1.13.1+cu117/torch_scatter-2.1.1+pt113cu117-cp38-cp38-linux_x86_64.whl" "${SCAT}"; then
  SCAT_READY=1
fi

SPRS_READY=0
if [[ -s "${SPRS}" ]]; then
  SPRS_READY=1
elif try_fetch "https://data.pyg.org/whl/torch-1.13.1+cu117/torch_sparse-0.6.17+pt113cu117-cp38-cp38-linux_x86_64.whl" "${SPRS}"; then
  SPRS_READY=1
fi

if [[ "${PY3_READY}" -eq 1 ]]; then
  pip install --no-index "${PY3}"
else
  echo "[pytorch3d] wheel unavailable (likely CDN 403). Fallback: source archive v0.7.4"
  PY3_SRC="${WHEEL_DIR}/pytorch3d-v0.7.4.tar.gz"
  if [[ ! -s "${PY3_SRC}" ]]; then
    try_fetch "https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.7.4.tar.gz" "${PY3_SRC}" \
      || try_fetch "https://codeload.github.com/facebookresearch/pytorch3d/tar.gz/refs/tags/v0.7.4" "${PY3_SRC}"
  fi
  pip install --no-cache-dir "${PY3_SRC}"
fi

if [[ "${SCAT_READY}" -eq 1 && "${SPRS_READY}" -eq 1 ]]; then
  pip install --no-index "${SCAT}" "${SPRS}"
else
  echo "[pyg] wheel(s) unavailable. Fallback via PyG index"
  pip install --no-cache-dir torch-scatter==2.1.1 torch-sparse==0.6.17 \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
fi

python -c "import pytorch3d, torch_scatter, torch_sparse; print('pytorch3d', pytorch3d.__version__, 'OK')"
