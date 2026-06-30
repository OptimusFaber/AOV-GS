#!/bin/bash
# Pre-download OneFormer weights for MP3D ActiveSem (avoids slow/partial downloads mid-run).
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "Downloading OneFormer MP3D finetune (~900MB)..."
python - <<'PY'
from transformers import AutoProcessor, AutoModelForUniversalSegmentation
AutoProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
AutoModelForUniversalSegmentation.from_pretrained("lly00412/oneformer-mp3d-finetune")
print("OneFormer MP3D models cached.")
PY

echo "Done. Re-run ActiveSem when ready."
