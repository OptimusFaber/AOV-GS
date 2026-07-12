#!/usr/bin/env bash
# Resolve AOV-GS conda env name / python with legacy fallbacks.
#
# Preferred: aov-gs
# Legacy:    active-gs, active-sgm (older installs)
#
# Usage:
#   source docker/resolve_conda_env.sh
#   # sets: AOVGS_ENV_NAME, AOVGS_PYTHON
#   aovgs_conda_activate   # optional

# shellcheck disable=SC2034
AOVGS_ENV_CANDIDATES=(aov-gs active-gs active-sgm)

_aovgs_env_prefix() {
  local e="$1"
  if [[ -n "${HOME:-}" && -d "${HOME}/anaconda3/envs/${e}" ]]; then
    echo "${HOME}/anaconda3/envs/${e}"
    return 0
  fi
  if [[ -n "${HOME:-}" && -d "${HOME}/miniconda3/envs/${e}" ]]; then
    echo "${HOME}/miniconda3/envs/${e}"
    return 0
  fi
  if [[ -d "/opt/conda/envs/${e}" ]]; then
    echo "/opt/conda/envs/${e}"
    return 0
  fi
  if [[ -n "${CONDA_DIR:-}" && -d "${CONDA_DIR}/envs/${e}" ]]; then
    echo "${CONDA_DIR}/envs/${e}"
    return 0
  fi
  return 1
}

AOVGS_ENV_NAME="${AOVGS_ENV_NAME:-}"
AOVGS_PYTHON="${AOVGS_PYTHON:-}"

if [[ -z "${AOVGS_ENV_NAME}" ]]; then
  for _e in "${AOVGS_ENV_CANDIDATES[@]}"; do
    if _aovgs_env_prefix "${_e}" >/dev/null; then
      AOVGS_ENV_NAME="${_e}"
      break
    fi
  done
fi
AOVGS_ENV_NAME="${AOVGS_ENV_NAME:-aov-gs}"

if [[ -z "${AOVGS_PYTHON}" ]]; then
  if _pref="$(_aovgs_env_prefix "${AOVGS_ENV_NAME}")" && [[ -x "${_pref}/bin/python" ]]; then
    AOVGS_PYTHON="${_pref}/bin/python"
  else
    for _e in "${AOVGS_ENV_CANDIDATES[@]}"; do
      if _pref="$(_aovgs_env_prefix "${_e}")" && [[ -x "${_pref}/bin/python" ]]; then
        AOVGS_ENV_NAME="${_e}"
        AOVGS_PYTHON="${_pref}/bin/python"
        break
      fi
    done
  fi
fi
AOVGS_PYTHON="${AOVGS_PYTHON:-python3}"

aovgs_conda_activate() {
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${AOVGS_ENV_NAME}" 2>/dev/null \
      || conda activate aov-gs 2>/dev/null \
      || conda activate active-gs 2>/dev/null \
      || conda activate active-sgm 2>/dev/null \
      || true
  fi
}
