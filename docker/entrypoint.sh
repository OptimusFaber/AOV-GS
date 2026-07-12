#!/bin/bash
set -eo pipefail

export PATH="/opt/conda/envs/aov-gs/bin:/opt/conda/bin:/usr/local/cuda/bin:${PATH}"

# conda activate.d hooks (binutils) use optional toolchain vars like ADDR2LINE
set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate aov-gs 2>/dev/null || conda activate active-gs 2>/dev/null || conda activate active-sgm 2>/dev/null || true
set -u

# Interactive bash sessions (docker exec / attach) also see conda
cat >/etc/profile.d/aov-gs-conda.sh <<'PROFILE'
export PATH="/opt/conda/envs/aov-gs/bin:/opt/conda/bin:/usr/local/cuda/bin:${PATH}"
if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  set +u
  source /opt/conda/etc/profile.d/conda.sh
  conda activate aov-gs 2>/dev/null || conda activate active-gs 2>/dev/null || conda activate active-sgm 2>/dev/null || true
  set -u
fi
PROFILE
chmod 644 /etc/profile.d/aov-gs-conda.sh

# NVIDIA/Mesa EGL for Habitat — baked ensure (independent of repo bind-mount on server).
if [[ -f /usr/local/bin/ensure_habitat_egl.sh ]]; then
    bash /usr/local/bin/ensure_habitat_egl.sh || echo "WARNING: ensure_habitat_egl failed (see above)" >&2
else
    # shellcheck disable=SC1091
    source /usr/local/bin/nvidia_habitat_env.sh
    source /usr/local/bin/habitat_mesa_fixup.sh
    _habitat_restore_conda_gl 2>/dev/null || true
    _habitat_hide_conda_mesa_egl 2>/dev/null || true
    _habitat_ensure_egl_headers 2>/dev/null || true
fi

if [[ -d /workspace/AOV-GS ]]; then
  cd /workspace/AOV-GS
  /usr/local/bin/bootstrap_third_parties.sh
  export PYTHONPATH="/workspace/AOV-GS:/workspace/AOV-GS/third_parties/splatam:${PYTHONPATH:-}"
fi

exec "$@"
