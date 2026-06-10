#!/bin/bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate active-sgm

# If repo is mounted, link baked third_parties when missing
if [[ -d /workspace/AOV-GS ]]; then
  cd /workspace/AOV-GS
  /usr/local/bin/bootstrap_third_parties.sh
  export PYTHONPATH="/workspace/AOV-GS:/workspace/AOV-GS/third_parties/splatam:${PYTHONPATH:-}"
fi

exec "$@"
