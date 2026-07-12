#!/bin/bash
# Activate the AOV-GS conda env in the current shell (legacy names still work):
#   source docker/activate_env.sh
#   conda -V && which pip

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/resolve_conda_env.sh"

export PATH="/opt/conda/envs/${AOVGS_ENV_NAME}/bin:/opt/conda/bin:/usr/local/cuda/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"

if [[ ! -f /opt/conda/etc/profile.d/conda.sh ]]; then
    # Host install: still allow conda activate via resolve helper
    aovgs_conda_activate
    return 0 2>/dev/null || exit 0
fi

set +u
# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh
conda activate "${AOVGS_ENV_NAME}" 2>/dev/null \
  || conda activate aov-gs 2>/dev/null \
  || conda activate active-gs 2>/dev/null \
  || conda activate active-sgm 2>/dev/null \
  || true
set -u

if [[ -f /.dockerenv ]]; then
    for _sh in \
        /workspace/AOV-GS/docker/nvidia_habitat_env.sh \
        /workspace/AOV-GS-V2/docker/nvidia_habitat_env.sh \
        /usr/local/bin/nvidia_habitat_env.sh; do
        if [[ -f "${_sh}" ]]; then
            # shellcheck disable=SC1090
            source "${_sh}"
            break
        fi
    done
fi
