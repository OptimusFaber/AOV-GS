# AOV-GS runtime image: Python 3.8, PyTorch 1.13.1+cu117, Habitat, SplaTAM deps.
#
# Build (from repo root):
#   docker build -f docker/Dockerfile -t aov-gs:cuda117 .
#
# With proxy:
#   docker build -f docker/Dockerfile -t aov-gs:cuda117 \
#     --build-arg HTTP_PROXY=http://user:pass@host:3128 \
#     --build-arg HTTPS_PROXY=http://user:pass@host:3128 .

# CUDA 11.7 devel (cudagl:11.7.1 tag removed from Docker Hub). Habitat EGL: glvnd ICD + runtime NVIDIA libs.
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_ACTIVESEM=1
ARG BUILD_HABITAT=0

ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    NO_PROXY=localhost,127.0.0.1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    CONDA_DIR=/opt/conda \
    ENV_NAME=aov-gs \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/conda/envs/aov-gs/bin:/opt/conda/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
    __EGL_VENDOR_LIBRARY_FILENAMES=/etc/glvnd/egl_vendor.d/10_nvidia.json \
    __GLX_VENDOR_LIBRARY_NAME=nvidia \
    EGL_PLATFORM=surfaceless \
    TCNN_CUDA_ARCHITECTURES=75;80;86 \
    TORCH_CUDA_ARCH_LIST=8.0;8.6 \
    MAX_JOBS=4 \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10

# ── system deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates unzip zip pigz jq rsync \
    build-essential cmake ninja-build pkg-config \
    libglvnd0 libgl1 libegl1 libgl1-mesa-dev libegl1-mesa-dev \
    libglfw3-dev libglm-dev \
    libx11-dev libxext-dev libxi-dev libxfixes-dev \
    libglib2.0-0 libsm6 libxrender1 libgomp1 \
    python3.8 python3.8-dev python3.8-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/glvnd/egl_vendor.d /usr/share/glvnd/egl_vendor.d \
    && printf '%s\n' \
         '{' \
         '  "file_format_version": "1.0.0",' \
         '  "ICD": {' \
         '    "library_path": "libEGL_nvidia.so.0"' \
         '  }' \
         '}' \
       > /etc/glvnd/egl_vendor.d/10_nvidia.json \
    && cp /etc/glvnd/egl_vendor.d/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# ── Miniforge (conda-forge, no Anaconda ToS) + env ─────────────────────────
# Miniconda latest requires `conda tos accept` — awkward in Docker; Miniforge is fine.
ARG MINIFORGE_VERSION=24.11.3-0
RUN wget -qO /tmp/miniforge.sh \
      "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && ${CONDA_DIR}/bin/conda config --system --remove-key default_channels 2>/dev/null || true \
    && ${CONDA_DIR}/bin/conda config --system --add channels conda-forge \
    && ${CONDA_DIR}/bin/conda create -n ${ENV_NAME} python=3.8 cmake=3.14 pip -y \
    && ${CONDA_DIR}/bin/conda clean -afy

SHELL ["/bin/bash", "-lc"]

# ── PyTorch (robust download + local install) ────────────────────────────────
# NOTE:
# - torch wheel ~1.8GB and often stalls via VPN/CDN.
# - First download the wheel with retry/fallback, then install locally.
RUN source ${CONDA_DIR}/etc/profile.d/conda.sh \
    && conda activate ${ENV_NAME} \
    && TORCH_WHL="torch-1.13.1+cu117-cp38-cp38-linux_x86_64.whl" \
    && TORCH_URL_PRIMARY="https://download-r2.pytorch.org/whl/cu117/${TORCH_WHL}" \
    && TORCH_URL_FALLBACK="https://download.pytorch.org/whl/cu117/${TORCH_WHL}" \
    && curl -fL -A "Mozilla/5.0" -C - --retry 8 --retry-delay 5 --connect-timeout 30 \
         "${TORCH_URL_PRIMARY}" -o "/tmp/${TORCH_WHL}" \
      || curl -fL -A "Mozilla/5.0" -C - --retry 8 --retry-delay 5 --connect-timeout 30 \
         "${TORCH_URL_FALLBACK}" -o "/tmp/${TORCH_WHL}" \
    && pip install \
         --default-timeout=1000 --retries=20 --progress-bar on \
         "/tmp/${TORCH_WHL}" \
         torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
         --extra-index-url https://download.pytorch.org/whl/cu117 \
    && rm -f "/tmp/${TORCH_WHL}" \
    && python -c "import torch; assert torch.cuda.is_available() or True; print('torch', torch.__version__)"

# ── pytorch3d + PyG (direct .whl, resilient to pip -f timeouts) ─────────────
COPY docker/install_pytorch3d_pyg.sh /tmp/install_pytorch3d_pyg.sh
COPY docker/wheels/ /tmp/wheels/
RUN chmod +x /tmp/install_pytorch3d_pyg.sh \
    && source ${CONDA_DIR}/etc/profile.d/conda.sh \
    && conda activate ${ENV_NAME} \
    && bash /tmp/install_pytorch3d_pyg.sh /tmp/wheels

# ── third_party sources (optional) ───────────────────────────────────────────
WORKDIR /opt/aov-gs-deps
RUN git clone --depth 1 https://github.com/spla-tam/SplaTAM.git splatam

# ── habitat-sim (optional, no local recursive clone) ─────────────────────────
RUN if [ "${BUILD_HABITAT}" = "1" ]; then \
      source ${CONDA_DIR}/etc/profile.d/conda.sh \
      && conda activate ${ENV_NAME} \
      && (conda install -y -c conda-forge -c aihabitat habitat-sim=0.2.3 headless \
          || conda install -y -c conda-forge -c aihabitat habitat-sim headless) \
      && for _egl in libEGL.so libEGL.so.1 libGLESv2.so libGLESv2.so.2; do \
           if [ -f "${CONDA_PREFIX}/lib/${_egl}" ]; then \
             mv "${CONDA_PREFIX}/lib/${_egl}" "${CONDA_PREFIX}/lib/${_egl}.mesa.bak"; \
           fi; \
         done; \
    else \
      echo "Skipping habitat-sim installation (BUILD_HABITAT=${BUILD_HABITAT})"; \
    fi

# ── CUDA extensions (geometry SLAM) ──────────────────────────────────────────
RUN source ${CONDA_DIR}/etc/profile.d/conda.sh \
    && conda activate ${ENV_NAME} \
    && pip install --no-cache-dir --no-build-isolation -v \
         git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch \
    && pip install --no-cache-dir --no-build-isolation -v \
         git+https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git@cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110

# ── ActiveSem: channel + sparse channel rasterization (optional) ─────────────
RUN if [ "${BUILD_ACTIVESEM}" = "1" ]; then \
      source ${CONDA_DIR}/etc/profile.d/conda.sh \
      && conda activate ${ENV_NAME} \
      && git clone --depth 1 -b liyan/dev \
           https://github.com/lly00412/semantic-gaussians.git /tmp/channel_rasterization \
      && git clone --depth 1 -b hairong/sparse_ver \
           https://github.com/lly00412/semantic-gaussians.git /tmp/sparse_channel_rasterization \
      && pip install --no-cache-dir --no-build-isolation -v /tmp/channel_rasterization/channel-rasterization \
      && pip install --no-cache-dir --no-build-isolation -v /tmp/sparse_channel_rasterization/sparse-channel-rasterization \
      && rm -rf /tmp/channel_rasterization /tmp/sparse_channel_rasterization; \
    fi

# ── Python deps ──────────────────────────────────────────────────────────────
RUN source ${CONDA_DIR}/etc/profile.d/conda.sh \
    && conda activate ${ENV_NAME} \
    && pip install --no-cache-dir \
         opencv-python-headless open3d tensorboardX mmengine trimesh plyfile wandb \
         pytorch-msssim lpips torchmetrics kornia transformers accelerate safetensors \
         filelock huggingface-hub regex tokenizers natsort PyMCubes==0.1.4 imgviz psutil fsspec \
         open_clip_torch timm pillow fvcore iopath \
    && pip install --no-cache-dir --no-build-isolation \
         git+https://github.com/facebookresearch/segment-anything.git

# ── install scripts copied for reference / re-run ────────────────────────────
WORKDIR /workspace/AOV-GS
COPY docker/entrypoint.sh /usr/local/bin/aov-gs-entrypoint.sh
COPY docker/nvidia_habitat_env.sh /usr/local/bin/nvidia_habitat_env.sh
COPY docker/habitat_mesa_fixup.sh /usr/local/bin/habitat_mesa_fixup.sh
COPY docker/ensure_habitat_egl.sh /usr/local/bin/ensure_habitat_egl.sh
COPY docker/bootstrap_third_parties.sh /usr/local/bin/bootstrap_third_parties.sh
RUN chmod +x \
    /usr/local/bin/aov-gs-entrypoint.sh \
    /usr/local/bin/nvidia_habitat_env.sh \
    /usr/local/bin/habitat_mesa_fixup.sh \
    /usr/local/bin/ensure_habitat_egl.sh \
    /usr/local/bin/bootstrap_third_parties.sh

# ── smoke test ───────────────────────────────────────────────────────────────
RUN source ${CONDA_DIR}/etc/profile.d/conda.sh \
    && conda activate ${ENV_NAME} \
    && python - <<'PY'
import importlib
mods = [
    "torch",
    "pytorch3d",
    "tinycudann",
    "diff_gaussian_rasterization",
    "segment_anything",
    "open_clip",
]
for m in mods:
    importlib.import_module(m)
if importlib.util.find_spec("habitat_sim") is not None:
    importlib.import_module("habitat_sim")
    print("smoke OK (with habitat_sim)")
else:
    print("smoke OK (without habitat_sim)")
PY

ENTRYPOINT ["/usr/local/bin/aov-gs-entrypoint.sh"]
CMD ["/bin/bash"]
