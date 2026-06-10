#!/bin/bash
# Сборка active-sgm для HPC-узлов без sudo (cds2 и аналоги).
#
# Проблема: conda-линкер (x86_64-conda-linux-gnu-c++) при сборке tinycudann
# передаёт -L$CONDA_PREFIX/lib, но libcuda лежит только в lib/stubs → "cannot find -lcuda".
# Решение: symlink libcuda.so в $CONDA_PREFIX/lib + явные LDFLAGS/LIBRARY_PATH.
#
# Использование (из корня AOV-GS):
#   bash scripts/installation/conda_env/build_sem_cds2.sh
#   bash scripts/installation/conda_env/build_sem_cds2.sh --env-only   # env уже есть
#   bash scripts/installation/conda_env/build_sem_cds2.sh --fix-tcnn    # только CUDA extensions
#
# Перед запуском на GPU Ampere/Ada (опционально):
#   export TCNN_CUDA_ARCHITECTURES="75;80;86"   # или 89 для Ada
#   export TORCH_CUDA_ARCH_LIST="8.0;8.6"

# Без -u: conda deactivate.d (gcc_linux-64) падает на unbound _CONDA_PYTHON_SYSCONFIGDATA_NAME_USED
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"

ENV_NAME="${ENV_NAME:-active-sgm}"
SKIP_CREATE=0
FIX_TCNN_ONLY=0
SKIP_HABITAT=0

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
  echo ""
  echo "Флаги:"
  echo "  --env-only     не создавать conda env (должен существовать ${ENV_NAME})"
  echo "  --fix-tcnn     только tinycudann + diff-gaussian (окружение уже частично собрано)"
  echo "  --skip-habitat пропустить сборку habitat-sim"
  echo "  -h, --help     эта справка"
}

for arg in "$@"; do
  case "$arg" in
    --env-only) SKIP_CREATE=1 ;;
    --fix-tcnn) FIX_TCNN_ONLY=1; SKIP_CREATE=1 ;;
    --skip-habitat) SKIP_HABITAT=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Неизвестный аргумент: $arg"; usage; exit 1 ;;
  esac
done

eval "$(conda shell.bash hook)"

# Линковка libcuda для conda gcc (cds2 fix).
setup_cds2_cuda_linking() {
  local stubs="${CONDA_PREFIX}/lib/stubs"
  local conda_lib="${CONDA_PREFIX}/lib"
  local system_cuda="/usr/lib/x86_64-linux-gnu"

  if [[ ! -f "${stubs}/libcuda.so" ]]; then
    echo "ОШИБКА: ${stubs}/libcuda.so не найден."
    echo "  conda install -c nvidia cuda-driver-dev=11.7"
    exit 1
  fi

  ln -sf "${stubs}/libcuda.so" "${conda_lib}/libcuda.so"
  echo "✓ ${conda_lib}/libcuda.so -> stubs/libcuda.so"

  export LDFLAGS="-L${stubs} -L${conda_lib} ${LDFLAGS:-}"
  export LIBRARY_PATH="${stubs}:${conda_lib}:${LIBRARY_PATH:-}"

  if [[ -e "${system_cuda}/libcuda.so" || -e "${system_cuda}/libcuda.so.1" ]]; then
    export LDFLAGS="-L${system_cuda} ${LDFLAGS}"
    export LIBRARY_PATH="${system_cuda}:${LIBRARY_PATH}"
    echo "✓ добавлен системный путь: ${system_cuda}"
  fi
}

export_conda_cuda_build_env() {
  export CUDA_HOME="${CONDA_PREFIX}"
  export CUDA_PATH="${CONDA_PREFIX}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
  export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
  # По умолчанию включаем Volta (70) + Turing/Ampere. Ada (89) задавайте явно через env var.
  export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-70;75;80;86}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6}"
  export MAX_JOBS="${MAX_JOBS:-4}"
  setup_cds2_cuda_linking
}

install_cuda_extensions() {
  echo "=== Установка tiny-cuda-nn и diff-gaussian-rasterization (cds2 linking) ==="
  export_conda_cuda_build_env

  (
    if [[ -d "${CONDA_PREFIX}/compiler_compat" ]]; then
      mv "${CONDA_PREFIX}/compiler_compat" "${CONDA_PREFIX}/.__compiler_compat_bak__"
    fi
    _restore_compiler_compat() {
      if [[ -d "${CONDA_PREFIX}/.__compiler_compat_bak__" ]]; then
        mv "${CONDA_PREFIX}/.__compiler_compat_bak__" "${CONDA_PREFIX}/compiler_compat"
      fi
    }
    trap _restore_compiler_compat EXIT

    setup_cds2_cuda_linking

    local -a pip_flags=(-v --no-build-isolation)
    if [[ "${FIX_TCNN_ONLY:-0}" -eq 1 ]]; then
      pip_flags+=(--force-reinstall --no-cache-dir)
    fi

    pip install "${pip_flags[@]}" \
      git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

    pip install "${pip_flags[@]}" \
      git+https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git@cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110
  )
}

verify_install() {
  echo ""
  echo "=== ПРОВЕРКА УСТАНОВКИ ==="
  python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
  python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
  python -c "import torch; print(f'✓ CUDA version: {torch.version.cuda}')"
  python -c "import pytorch3d; print(f'✓ PyTorch3D: {pytorch3d.__version__}')" 2>/dev/null || true
  python -c "import torch_scatter; print('✓ torch-scatter: OK')" 2>/dev/null || true
  python -c "import torch_sparse; print('✓ torch-sparse: OK')" 2>/dev/null || true
  python -c "import habitat_sim; print('✓ habitat-sim: OK')" 2>/dev/null || true
  python -c "import tinycudann as tcnn; print('✓ tiny-cuda-nn: OK', tcnn.__file__)"
  python -c "import diff_gaussian_rasterization; print('✓ diff-gaussian-rasterization: OK')"
  python -c "import segment_anything; print('✓ SAM: OK')" 2>/dev/null || true
  python -c "import open_clip; print('✓ open_clip: OK')" 2>/dev/null || true
}

# --- только починка CUDA extensions ---
if [[ "${FIX_TCNN_ONLY}" -eq 1 ]]; then
  conda activate "${ENV_NAME}"
  if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
    echo "Ошибка: не удалось активировать ${ENV_NAME}"
    exit 1
  fi
  echo "=== Режим --fix-tcnn: пересборка CUDA extensions в ${ENV_NAME} ==="
  install_cuda_extensions
  verify_install
  echo ""
  echo "=== Готово (--fix-tcnn) ==="
  exit 0
fi

# --- полная установка ---
echo "=== Сборка active-sgm (cds2 / HPC, без sudo) ==="
echo "Корень: ${ROOT}"

if [[ "${SKIP_CREATE}" -eq 0 ]]; then
  conda create -n "${ENV_NAME}" python=3.8 cmake=3.14 -y
fi

conda activate "${ENV_NAME}"
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
  echo "Ошибка: не удалось активировать ${ENV_NAME}"
  exit 1
fi

echo "✓ Conda: ${CONDA_DEFAULT_ENV}"
which python
python --version

echo "=== PyTorch CUDA 11.7 ==="
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y

echo "=== pytorch3d ==="
conda install pytorch3d=0.7.4 -c pytorch3d -c conda-forge -y

echo "=== torch-scatter / torch-sparse ==="
pip install torch-scatter==2.1.1 torch-sparse==0.6.17 \
  -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

echo "=== CUDA toolkit + CCCL ==="
conda install -c nvidia -c conda-forge \
  cuda-nvcc=11.7 \
  cuda-cudart-dev=11.7 \
  cuda-libraries-dev=11.7 \
  cuda-cccl=11.7 \
  -y

echo "=== Компиляторы ==="
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 -y

export_conda_cuda_build_env

echo "=== libxcrypt + cuda-driver-dev ==="
conda install -c conda-forge libxcrypt -y
# Пин 11.7: на cds2 иногда подтягивается cuda-driver-dev_linux-64 13.x
conda install -c nvidia cuda-driver-dev=11.7.99 -y

setup_cds2_cuda_linking

echo "=== Проверка CCCL ==="
for path in cuda/std thrust cub; do
  if [[ ! -d "${CONDA_PREFIX}/include/${path}" ]]; then
    echo "ОШИБКА: ${CONDA_PREFIX}/include/${path} не найден"
    exit 1
  fi
  echo "✓ ${path}"
done

if [[ "${SKIP_HABITAT}" -eq 0 ]]; then
  echo "=== habitat-sim ==="
  conda install -c conda-forge git ninja \
    libgl-devel libegl-devel libglvnd \
    xorg-libx11 xorg-libxext xorg-libxfixes \
    -y
  export CMAKE_PREFIX_PATH="${CONDA_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

  HABITAT_SRC="${ROOT}/third_parties/habitat_sim"
  mkdir -p "${ROOT}/third_parties"
  if [[ -d "${HABITAT_SRC}/.git" ]] || [[ -f "${HABITAT_SRC}/setup.py" ]] || \
     { [[ -d "${HABITAT_SRC}" ]] && [[ -n "$(ls -A "${HABITAT_SRC}" 2>/dev/null)" ]]; }; then
    echo "Найден ${HABITAT_SRC}: clone пропущен."
  else
    git clone --recursive https://github.com/Huangying-Zhan/habitat-sim.git "${HABITAT_SRC}"
  fi

  (
    cd "${HABITAT_SRC}"
    rm -rf build
    pip install -r requirements.txt
    python setup.py install --headless --bullet
  )
else
  echo "=== habitat-sim пропущен (--skip-habitat) ==="
fi

install_cuda_extensions

echo "=== Остальные pip-зависимости ==="
#
# Важно: окружение на Python 3.8. Новые версии transformers/tokenizers уже требуют Python>=3.9,
# а tokenizers начинает собираться из sdist и падает на backend deps.
# Поэтому пиним совместимые версии (последние стабильные под py3.8).
pip install opencv-python-headless open3d tensorboardX mmengine trimesh plyfile \
  wandb pytorch-msssim lpips torchmetrics kornia safetensors filelock \
  "huggingface-hub<0.21" regex natsort PyMCubes==0.1.4 imgviz psutil hf-xet fsspec

# HF stack (py3.8 compatible)
pip install \
  "transformers==4.30.2" \
  "tokenizers==0.13.3" \
  "accelerate==0.20.3"

echo "=== SAM + CLIP ==="
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install open_clip_torch timm pillow

verify_install

echo ""
echo "=== УСПЕШНО (build_sem_cds2.sh) ==="
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Если упало только на tinycudann в старом env:"
echo "  bash scripts/installation/conda_env/build_sem_cds2.sh --fix-tcnn"
