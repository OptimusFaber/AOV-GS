#!/bin/bash
# Self-healing EGL for Habitat in containers without host access.
# Called from entrypoint and Habitat / SLAM scripts.
#
#   bash docker/ensure_habitat_egl.sh
#   bash docker/ensure_habitat_egl.sh --probe-only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROBE_ONLY=0
[[ "${1:-}" == "--probe-only" ]] && PROBE_ONLY=1

_source_baked_env() {
  for _f in /usr/local/bin/nvidia_habitat_env.sh "${SCRIPT_DIR}/nvidia_habitat_env.sh"; do
    if [[ -f "${_f}" ]]; then
      # shellcheck disable=SC1090
      source "${_f}"
      return 0
    fi
  done
  echo "ERROR: nvidia_habitat_env.sh not found" >&2
  return 1
}

_source_mesa_fixup() {
  for _f in /usr/local/bin/habitat_mesa_fixup.sh "${SCRIPT_DIR}/habitat_mesa_fixup.sh"; do
    if [[ -f "${_f}" ]]; then
      # shellcheck disable=SC1090
      source "${_f}"
      return 0
    fi
  done
}

_nvidia_egl_present() {
  _habitat_find_nvidia_egl >/dev/null 2>&1
}

_hide_conda_egl_if_nvidia() {
  if _nvidia_egl_present; then
    _habitat_hide_conda_mesa_egl 2>/dev/null || true
  else
    _habitat_restore_conda_mesa_egl 2>/dev/null || true
    echo "[ensure_habitat_egl] NVIDIA EGL missing — keeping conda/system Mesa EGL"
  fi
}

# Install userspace libnvidia-gl matching driver version (nvidia-smi) — no host access.
_install_nvidia_gl_userspace() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ensure_habitat_egl] nvidia-smi unavailable — skipping apt libnvidia-gl" >&2
    return 1
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "[ensure_habitat_egl] nvidia-smi sees no GPU" >&2
    return 1
  fi

  local ver major
  ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  major="${ver%%.*}"
  [[ -n "${major}" ]] || return 1

  echo "[ensure_habitat_egl] apt: libnvidia-gl for driver ${ver} (major ${major})"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq

  local pkg ok=0
  for pkg in \
    "libnvidia-gl-${major}" \
    "libnvidia-gl-${major}-server" \
    "libnvidia-egl-gbm1" \
    "libnvidia-egl-core-${major}"; do
    if apt-cache show "${pkg}" >/dev/null 2>&1; then
      apt-get install -y -qq "${pkg}" && { echo "  installed ${pkg}"; ok=1; }
    fi
  done

  if [[ "${ok}" -eq 0 ]]; then
    for pkg in libnvidia-gl-550 libnvidia-gl-535 libnvidia-gl-525; do
      if apt-cache show "${pkg}" >/dev/null 2>&1; then
        apt-get install -y -qq "${pkg}" && { echo "  installed fallback ${pkg}"; ok=1; break; }
      fi
    done
  fi

  [[ "${ok}" -eq 1 ]]
}

_egl_python_probe() {
  python - <<'PY'
import ctypes
import ctypes.util
import os
import sys

lib = ctypes.util.find_library("EGL")
if not lib:
    print("EGL_PROBE: no libEGL in loader path")
    sys.exit(1)

egl = ctypes.CDLL(lib)
EGL_DEFAULT_DISPLAY = 0
egl.eglGetDisplay.argtypes = [ctypes.c_void_p]
egl.eglGetDisplay.restype = ctypes.c_void_p
egl.eglInitialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
egl.eglInitialize.restype = ctypes.c_int

dpy = egl.eglGetDisplay(EGL_DEFAULT_DISPLAY)
if not dpy:
    print("EGL_PROBE: eglGetDisplay -> NULL (EGL_BAD_PARAMETER / no display)")
    sys.exit(2)

maj = ctypes.c_int()
min_ = ctypes.c_int()
if not egl.eglInitialize(dpy, ctypes.byref(maj), ctypes.byref(min_)):
    print("EGL_PROBE: eglInitialize failed")
    sys.exit(3)

print(f"EGL_PROBE: OK EGL {maj.value}.{min_.value} backend={os.environ.get('HABITAT_EGL_BACKEND','?')}")
PY
}

_main() {
  if command -v conda >/dev/null 2>&1; then
    set +u
    # shellcheck disable=SC1091
    source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate aov-gs 2>/dev/null || true
    set -u
  fi

  _source_baked_env
  _source_mesa_fixup
  _habitat_restore_conda_gl 2>/dev/null || true
  _habitat_ensure_egl_headers 2>/dev/null || true

  if ! _nvidia_egl_present; then
    echo "[ensure_habitat_egl] libEGL_nvidia.so.0 not found — installing userspace packages"
    _install_nvidia_gl_userspace || true
    _source_baked_env
  fi

  _hide_conda_egl_if_nvidia

  if _egl_python_probe; then
    echo "[ensure_habitat_egl] done"
    return 0
  fi

  echo "[ensure_habitat_egl] NVIDIA EGL did not come up — trying Mesa surfaceless" >&2
  export HABITAT_EGL_BACKEND=mesa
  unset __EGL_VENDOR_LIBRARY_FILENAMES
  export EGL_PLATFORM=surfaceless
  _habitat_restore_conda_mesa_egl 2>/dev/null || true

  if _egl_python_probe; then
    echo "[ensure_habitat_egl] Mesa EGL OK (slower, but should work)"
    return 0
  fi

  echo "[ensure_habitat_egl] FAIL — diagnostics:" >&2
  echo "  NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}" >&2
  echo "  NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-<unset>}" >&2
  echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}" >&2
  ls -la /usr/local/nvidia/lib64/libEGL* 2>/dev/null >&2 || true
  ls -la /usr/lib/x86_64-linux-gnu/libEGL* 2>/dev/null | head -5 >&2 || true
  echo "  Check: nvidia-smi, NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display" >&2
  return 1
}

if [[ "${PROBE_ONLY}" -eq 1 ]]; then
  _source_baked_env
  _egl_python_probe
else
  _main
fi
