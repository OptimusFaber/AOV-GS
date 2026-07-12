# NVIDIA headless EGL for Habitat-Sim (sourced from entrypoint / habitat scripts).
# Ref: https://github.com/facebookresearch/habitat-sim/issues/2424

_habitat_find_nvidia_egl() {
  local f
  for f in \
    /usr/local/nvidia/lib64/libEGL_nvidia.so.0 \
    /usr/local/nvidia/lib/libEGL_nvidia.so.0 \
    /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 \
    /lib/x86_64-linux-gnu/libEGL_nvidia.so.0; do
    [[ -f "${f}" ]] && { echo "${f}"; return 0; }
  done
  f="$(find /usr /lib /usr/local/nvidia -name 'libEGL_nvidia.so.0' 2>/dev/null | head -1)"
  [[ -n "${f}" ]] && { echo "${f}"; return 0; }
  return 1
}

_habitat_prepend_nvidia_lib_paths() {
  local d paths=()
  for d in /usr/local/nvidia/lib64 /usr/local/nvidia/lib \
           /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu; do
    [[ -d "$d" ]] && paths+=("$d")
  done
  if ((${#paths[@]} > 0)); then
    local IFS=:
    export LD_LIBRARY_PATH="${paths[*]}:${LD_LIBRARY_PATH:-}"
  fi
}

_habitat_prepend_nvidia_lib_paths

unset LD_PRELOAD

if [[ "${NVIDIA_VISIBLE_DEVICES:-}" == "void" ]]; then
  unset NVIDIA_VISIBLE_DEVICES
fi
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
# all — so nvidia-container-toolkit mounts EGL/GL (if it reads image ENV)
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
export __GLX_VENDOR_LIBRARY_NAME=nvidia
unset MAGNUM_DISABLE_GL_VERSION_CHECK
export HABITAT_USE_MESH_PLY="${HABITAT_USE_MESH_PLY:-0}"
unset DISPLAY

if _habitat_find_nvidia_egl >/dev/null; then
  export HABITAT_EGL_BACKEND=nvidia
  export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/etc/glvnd/egl_vendor.d/10_nvidia.json}"
  unset EGL_PLATFORM
else
  export HABITAT_EGL_BACKEND=mesa
  unset __EGL_VENDOR_LIBRARY_FILENAMES
  export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
  export MESA_GL_VERSION_OVERRIDE="${MESA_GL_VERSION_OVERRIDE:-4.5}"
  export MESA_GLSL_VERSION_OVERRIDE="${MESA_GLSL_VERSION_OVERRIDE:-450}"
fi
