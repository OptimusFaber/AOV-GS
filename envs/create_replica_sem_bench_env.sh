#!/usr/bin/env bash
# Create a conda environment for replica_sem_benchmark (SAM1 + CLIP + HF SAM2/SAM3).
#
# Installs PyTorch (CUDA or CPU), then transformers>=4.56 and deps from
#     requirements_replica_sem_bench.txt (open_clip, segment_anything from git, …).
#
# EN: installs PyTorch first, then pip packages for eval_clip_sam_systematic.py.
#
# Usage:
#   cd New-Proj
#   bash envs/create_replica_sem_bench_env.sh
#
# Optional environment variables:
#   REPLICA_SEM_BENCH_ENV_NAME   conda env name (default: replica-sem-bench)
#   PYTHON_VER                   Python version (default: 3.10)
#   TORCH_CUDA                   PyTorch wheel flavour:
#                                  cu124  — CUDA 12.4 (default)
#                                  cu121  — CUDA 12.1
#                                  cu118  — CUDA 11.8
#                                  cpu    — CPU-only
#
# Examples:
#   TORCH_CUDA=cu121 bash envs/create_replica_sem_bench_env.sh
#   REPLICA_SEM_BENCH_ENV_NAME=my-bench PYTHON_VER=3.11 bash envs/create_replica_sem_bench_env.sh

# Do not use `set -u`: conda hooks from the *currently active* env (e.g. active-clip’s
# libxml2 deactivate script) reference unset variables and abort under nounset.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEW_PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REQ="${SCRIPT_DIR}/requirements_replica_sem_bench.txt"

ENV_NAME="${REPLICA_SEM_BENCH_ENV_NAME:-replica-sem-bench}"
PYTHON_VER="${PYTHON_VER:-3.10}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"

if [[ ! -f "$REQ" ]]; then
  echo "Missing: $REQ" >&2
  exit 1
fi

if ! command -v conda &>/dev/null; then
  echo "conda not found in PATH. Install Miniconda/Anaconda or init conda for this shell." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -qE "^[[:space:]]*${ENV_NAME}[[:space:]]"; then
  echo "Conda env '$ENV_NAME' already exists. Remove it first, e.g.:"
  echo "  conda env remove -n $ENV_NAME -y"
  exit 1
fi

echo "Creating conda env: $ENV_NAME (python=$PYTHON_VER)"
conda create -y -n "$ENV_NAME" "python=${PYTHON_VER}"

conda activate "$ENV_NAME"

echo "Installing PyTorch + torchvision (TORCH_CUDA=$TORCH_CUDA) ..."
if [[ "$TORCH_CUDA" == "cpu" ]]; then
  pip install --upgrade pip
  pip install torch torchvision
else
  pip install --upgrade pip
  pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
fi

echo "Installing benchmark requirements from requirements_replica_sem_bench.txt ..."
pip install -r "$REQ"

# ``pip install -r`` may pull a different ``torch`` than the CUDA wheel we installed first,
# leaving ``torchvision`` on an old build → "torchvision X requires torch==Y" conflicts.
echo "Re-aligning torch + torchvision (same CUDA index as above) ..."
if [[ "$TORCH_CUDA" == "cpu" ]]; then
  pip install --upgrade torch torchvision
else
  pip install --upgrade torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
fi

echo ""
echo "Note: pip warnings naming mlflow / paddlex / jupyter / botocore / … mean those"
echo "      packages are already in this Python env but are NOT required for the benchmark."
echo "      For a quiet install, use an env created only by this script (no extra pip)."

echo ""
echo "Smoke imports (versions):"
python - <<'PY'
import importlib
import sys

def ver(mod: str) -> str:
    importlib.import_module(mod)
    v = getattr(sys.modules[mod], "__version__", None)
    if v is not None:
        return str(v)
    try:
        from importlib.metadata import PackageNotFoundError, version

        for dist_name in (mod, mod.replace("_", "-")):
            try:
                return version(dist_name)
            except PackageNotFoundError:
                continue
    except Exception:
        pass
    return "ok"

mods = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("segment_anything", "segment_anything"),
    ("open_clip", "open_clip"),
    ("cv2", "cv2"),
    ("pandas", "pandas"),
]
for label, name in mods:
    try:
        print(f"  {label}: {ver(name)}")
    except Exception as e:
        print(f"  {label}: FAILED ({e})", file=sys.stderr)
        sys.exit(1)
PY

echo ""
echo "Done. Activate with:"
echo "  conda activate $ENV_NAME"
echo "If \`transformers.pipeline\` / HF SAM fails with a bogus import error, user"
echo "  site-packages may be breaking boto3/accelerate — run with:"
echo "  export PYTHONNOUSERSITE=1"
echo "Then from New-Proj:"
echo "  export HF_HOME=/mnt/data/model-ckpts/clip/huggingface_hub"
echo "  export OPENCLIP_CACHE=/mnt/data/model-ckpts/clip/open_clip"
echo "  python replica_sem_benchmark/eval_clip_sam_systematic.py --help"
