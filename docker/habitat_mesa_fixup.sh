#!/bin/bash
# Headless Habitat uses EGL (not GLX). Hide only conda Mesa EGL stubs — keep libGL/libOpenGL.
_habitat_hide_conda_mesa_egl() {
  local p="${CONDA_PREFIX:-/opt/conda/envs/aov-gs}/lib"
  for _lib in libEGL.so libEGL.so.1 libGLESv2.so libGLESv2.so.2; do
    if [[ -f "${p}/${_lib}" && ! -f "${p}/${_lib}.mesa.bak" ]]; then
      mv "${p}/${_lib}" "${p}/${_lib}.mesa.bak"
    fi
  done
}

_habitat_restore_conda_mesa_egl() {
  local p="${CONDA_PREFIX:-/opt/conda/envs/aov-gs}/lib"
  for _lib in libEGL.so libEGL.so.1 libGLESv2.so libGLESv2.so.2; do
    if [[ -f "${p}/${_lib}.mesa.bak" ]]; then
      mv "${p}/${_lib}.mesa.bak" "${p}/${_lib}"
    fi
  done
}

_habitat_restore_conda_gl() {
  local p="${CONDA_PREFIX:-/opt/conda/envs/aov-gs}/lib"
  for _lib in libGL.so libGL.so.1 libOpenGL.so libOpenGL.so.0; do
    if [[ -f "${p}/${_lib}.mesa.bak" ]]; then
      mv "${p}/${_lib}.mesa.bak" "${p}/${_lib}"
    fi
  done
}

# Conda EGL headers can miss KHR/khrplatform.h -> Magnum OpenGLFunctionLoader.cpp fails.
_habitat_ensure_egl_headers() {
  local inc="${CONDA_PREFIX:-/opt/conda/envs/aov-gs}/include"
  if [[ -f "${inc}/KHR/khrplatform.h" || -f /usr/include/KHR/khrplatform.h ]]; then
    if [[ ! -f "${inc}/KHR/khrplatform.h" && -f /usr/include/KHR/khrplatform.h ]]; then
      mkdir -p "${inc}/KHR"
      ln -sf /usr/include/KHR/khrplatform.h "${inc}/KHR/khrplatform.h"
      echo "=== linked ${inc}/KHR/khrplatform.h -> /usr/include/KHR/khrplatform.h ==="
    fi
    return 0
  fi
  echo "WARNING: KHR/khrplatform.h not found (install libegl-devel or apt libegl1-mesa-dev)" >&2
  return 1
}
