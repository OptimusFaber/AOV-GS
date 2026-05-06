#!/bin/bash
set -e  # Exit on error

# Корень AOV-GS (не зависит от текущего cwd при запуске bash .../build_sem.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"

# Initialize conda for bash script
eval "$(conda shell.bash hook)"

echo "=== Создание ПОЛНОГО conda окружения с CUDA 11.7 + CCCL ==="
echo "Будут установлены: tiny-cuda-nn + diff-gaussian-rasterization + все остальное"

### create conda environment ###
conda create -n active-sgm python=3.8 cmake=3.14 -y

### activate conda environment ###
conda activate active-sgm

# Verify conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "active-sgm" ]; then
    echo "Error: Failed to activate conda environment 'active-sgm'"
    exit 1
fi

echo "✓ Conda окружение активировано: $CONDA_DEFAULT_ENV"

# Verify we're using the correct python from conda
which python
python --version

echo "=== Установка PyTorch с CUDA 11.7 ==="
# PyTorch with CUDA 11.7
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y

echo "=== Установка pytorch3d ==="
# pytorch3d
conda install pytorch3d=0.7.4 -c pytorch3d -c conda-forge -y

echo "=== Установка torch-scatter и torch-sparse ==="
# torch-scatter and torch-sparse
pip install torch-scatter==2.1.1 torch-sparse==0.6.17 \
  -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

echo "=== Установка CUDA toolkit с CCCL ==="
# CUDA toolkit с CUDA C++ Standard Library (CCCL) - ЭТО КЛЮЧЕВОЕ ОТЛИЧИЕ!
conda install -c nvidia -c conda-forge \
  cuda-nvcc=11.7 \
  cuda-cudart-dev=11.7 \
  cuda-libraries-dev=11.7 \
  cuda-cccl=11.7 \
  -y

echo "=== Установка компиляторов ==="
# GCC/G++ компиляторы
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 -y

# Настройка переменных окружения
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++

# CUDA архитектуры для tiny-cuda-nn / torch extensions (CUDA 11.7 макс. ~sm_86).
# На GPU Ada (sm_89) PyTorch всё равно сообщает 89 — без явного списка сборка часто ломается.
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-75;80;86}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6}"
export MAX_JOBS="${MAX_JOBS:-4}"

echo "=== Установка дополнительных библиотек ==="
# libxcrypt для совместимости
conda install -c conda-forge libxcrypt -y

# CUDA driver dev
conda install -c nvidia cuda-driver-dev=11.7 -y

# Настройка линковки
export LDFLAGS="-L$CONDA_PREFIX/lib/stubs $LDFLAGS"
export LIBRARY_PATH="$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH"

echo "=== Проверка наличия CUDA C++ Standard Library ==="
# Проверяем, что cuda-cccl установлен
if [ -d "$CONDA_PREFIX/include/cuda/std" ]; then
    echo "✓ CUDA C++ Standard Library найдена в $CONDA_PREFIX/include/cuda/std"
    ls -la $CONDA_PREFIX/include/cuda/std/ | head -10
else
    echo "✗ ОШИБКА: cuda/std/ не найдена!"
    exit 1
fi

if [ -d "$CONDA_PREFIX/include/thrust" ]; then
    echo "✓ Thrust найден в $CONDA_PREFIX/include/thrust"
else
    echo "✗ ОШИБКА: thrust/ не найден!"
    exit 1
fi

if [ -d "$CONDA_PREFIX/include/cub" ]; then
    echo "✓ CUB найден в $CONDA_PREFIX/include/cub"
else
    echo "✗ ОШИБКА: cub/ не найден!"
    exit 1
fi

echo "=== Сборка habitat-sim из исходников (как в ActiveSGM) ==="
# ВНИМАНИЕ: не используем conda-пакеты aihabitat `headless` / `habitat-sim-mutex`,
# т.к. они часто приводят к крашу шейдеров PTexMeshShader (GL link failed).
#
# OpenGL/GLX: Magnum вызывает find_package(OpenGL). В чистом conda без системных
# библиотек CMake падает. Ставим нужное через conda-forge.
conda install -c conda-forge git ninja \
  libgl-devel libegl-devel libglvnd \
  xorg-libx11 xorg-libxext xorg-libxfixes \
  -y
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

HABITAT_SRC="${ROOT}/third_parties/habitat_sim"
mkdir -p "${ROOT}/third_parties"
if [ ! -d "${HABITAT_SRC}/.git" ]; then
    git clone --recursive https://github.com/Huangying-Zhan/habitat-sim.git "${HABITAT_SRC}"
else
    echo "Найден ${HABITAT_SRC}: clone пропущен. При проблемах: git pull && git submodule update --init --recursive"
fi

(
    cd "${HABITAT_SRC}"
    rm -rf build
    pip install -r requirements.txt
    python setup.py install --headless --bullet
)

echo "=== Установка tiny-cuda-nn и diff-gaussian-rasterization (CUDA extensions) ==="
# conda кладёт Python distutils ld в compiler_compat с -Wl,--sysroot=/ → на части хостов
# падает линковка (/lib64/libm.so.6). Временно убираем compiler_compat из PREFIX на время pip.
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

  export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-75;80;86}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6}"
  export MAX_JOBS="${MAX_JOBS:-4}"

  pip install -v --no-build-isolation \
    git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

  pip install -v --no-build-isolation \
    git+https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git@cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110
)

echo "=== Установка дополнительных зависимостей ==="
#
# Важно для AOV-GS:
# - SAM: segment-anything
# - CLIP: open_clip_torch
#
pip install opencv-python-headless
pip install open3d
pip install tensorboardX
pip install mmengine
pip install trimesh
pip install plyfile
pip install wandb 
pip install pytorch-msssim 
pip install lpips 
pip install torchmetrics 
pip install kornia
pip install transformers 
pip install accelerate 
pip install safetensors 
pip install filelock
pip install huggingface-hub 
pip install regex 
pip install tokenizers 
pip install natsort 
pip install PyMCubes==0.1.4
pip install imgviz
pip install psutil 
pip install hf-xet
pip install fsspec 

echo "=== Установка SAM + CLIP (AOV-GS semantic features) ==="
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install open_clip_torch
pip install timm
pip install pillow

echo ""
echo "=== ПРОВЕРКА УСТАНОВКИ ==="
echo "Проверяем что всё работает..."

python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'✓ CUDA version: {torch.version.cuda}')"
python -c "import pytorch3d; print(f'✓ PyTorch3D: {pytorch3d.__version__}')"
python -c "import torch_scatter; print('✓ torch-scatter: OK')"
python -c "import torch_sparse; print('✓ torch-sparse: OK')"
python -c "import habitat_sim; print('✓ habitat-sim: OK')"
python -c "import tinycudann; print('✓ tiny-cuda-nn: OK')"
python -c "import diff_gaussian_rasterization; print('✓ diff-gaussian-rasterization: OK')"
python -c "import segment_anything; print('✓ SAM (segment-anything): OK')"
python -c "import open_clip; print('✓ open_clip_torch: OK')"

echo ""
echo "=== УСПЕШНО! Окружение active-sgm готово ==="
echo ""
echo "Для активации в будущем используйте:"
echo "  conda activate active-sgm"
echo ""
echo "Установленные компоненты:"
echo "  ✓ Python 3.8"
echo "  ✓ PyTorch 1.13.1 + CUDA 11.7"
echo "  ✓ pytorch3d 0.7.4"
echo "  ✓ habitat-sim 0.2.3"
echo "  ✓ torch-scatter & torch-sparse"
echo "  ✓ CUDA C++ Standard Library (CCCL) 11.7"
echo "  ✓ tiny-cuda-nn"
echo "  ✓ diff-gaussian-rasterization-w-depth"
echo "  ✓ SAM (segment-anything)"
echo "  ✓ CLIP (open_clip_torch)"
echo ""
echo "ВСЁ В ОДНОМ ОКРУЖЕНИИ БЕЗ СИСТЕМНОЙ CUDA!"

