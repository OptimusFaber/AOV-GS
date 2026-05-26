#!/bin/bash
# Wrap a command: record wall time, peak VRAM, append to ablation_metrics.json
#
# Usage:
#   bash scripts/ablation/run_with_metrics.sh STAGE_NAME RESULT_DIR COMMAND...
#
# Example:
#   bash scripts/ablation/run_with_metrics.sh slam results/Replica/office0/ActiveSem/run_0 \
#       python src/main/activesgm.py --cfg ...

set -euo pipefail

STAGE="${1:?stage name}"
RESULT_DIR="${2:?result dir}"
shift 2

mkdir -p "$RESULT_DIR"
VRAM_LOG="${RESULT_DIR}/vram_${STAGE}.log"
METRICS="${RESULT_DIR}/ablation_metrics.json"

START_EPOCH=$(date +%s)
VRAM_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 >"$VRAM_LOG" 2>/dev/null &
  VRAM_PID=$!
fi

set +e
"$@"
EXIT_CODE=$?
set -e

END_EPOCH=$(date +%s)
WALL_S=$((END_EPOCH - START_EPOCH))

if [[ -n "${VRAM_PID}" ]]; then
  kill "$VRAM_PID" 2>/dev/null || true
  wait "$VRAM_PID" 2>/dev/null || true
fi

VRAM_PEAK=""
if [[ -f "$VRAM_LOG" ]]; then
  VRAM_PEAK=$(awk 'NF{m=0; for(i=1;i<=NF;i++) if($i+0>m) m=$i+0; if(m>0) print m}' "$VRAM_LOG" | sort -n | tail -1)
fi

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
python3 - "$METRICS" "$STAGE" "$WALL_S" "${VRAM_PEAK:-}" "$PROJ_DIR" "$RESULT_DIR" <<'PY'
import json, sys
from pathlib import Path

metrics_path, stage, wall_s, vram_peak, proj, result_dir = sys.argv[1:7]
p = Path(metrics_path)
data = {}
if p.exists():
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = {}

stages = data.setdefault("stages", {})
entry = {"wall_s": int(wall_s)}
if vram_peak:
    entry["vram_peak_mb"] = float(vram_peak)
stages[stage] = entry
data["stages"] = stages

# gaussian count after slam stages
if stage.startswith("slam"):
    run = Path(result_dir)
    for rel in ("splatam/final/params.npz", "splatam/final/params0.npz"):
        ck = run / rel
        if ck.exists():
            try:
                import numpy as np
                z = np.load(str(ck))
                data["num_gaussians"] = int(z["means3D"].shape[0])
            except Exception:
                pass
            break

total = sum(int(v.get("wall_s", 0)) for v in stages.values())
data["total_wall_s"] = total
p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY

exit "$EXIT_CODE"
