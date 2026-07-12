# Docker for AOV-GS

Image: **CUDA 11.7**, Python 3.8, PyTorch 1.13.1+cu117, Habitat EGL, tiny-cuda-nn, diff-gaussian, SAM/CLIP.  
Conda env inside the image: **`aov-gs`** (Miniforge).

See also: [root README](../README.md), [conda setup](../scripts/installation/README.md), [pipeline](../scripts/aov-gs/README.md).

## Requirements

- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Host driver with **OpenGL/EGL** (not compute-only)
- ~15–25 GB for the image; build takes **1–2 h** with network on the build machine

## Build

```bash
cd AOV-GS

# optional: cache pytorch3d / PyG wheels if Meta CDN is slow
bash docker/wheels/download_wheels.sh

bash docker/build.sh

# proxy
HTTP_PROXY=http://USER:PASS@HOST:3128 HTTPS_PROXY=http://USER:PASS@HOST:3128 bash docker/build.sh

# OpenSem only (skip ActiveSem channel rasterizers)
BUILD_ACTIVESEM=0 bash docker/build.sh
```

Or: `docker compose -f docker/docker-compose.yml build`

## Run

```bash
bash docker/run.sh

# one-shot command
bash docker/run.sh python src/main/activesgm.py \
  --cfg configs/Replica/office0/ActiveOpenSem.py --seed 0 --enable_vis 0
```

Inside the container the `aov-gs` conda env is already active. Repo is mounted at `/workspace/AOV-GS`; volumes persist `data/`, `results/`, `ckpts/`.

## Layout (what stays in `docker/`)

| File | Role |
|------|------|
| `Dockerfile`, `build.sh`, `run.sh`, `docker-compose.yml` | build & launch |
| `entrypoint.sh` | conda + Habitat EGL + `third_parties` bootstrap |
| `ensure_habitat_egl.sh`, `nvidia_habitat_env.sh`, `habitat_mesa_fixup.sh` | headless Habitat EGL |
| `bootstrap_third_parties.sh` | symlink deps into the mounted repo |
| `install_pytorch3d_pyg.sh`, `wheels/` | build-time wheels |
| `activate_env.sh`, `resolve_conda_env.sh` | manual `source` of conda inside/outside the image |

## Offline image transfer

```bash
# build machine
docker save aov-gs:cuda117 | gzip > aov-gs-cuda117.tar.gz

# target host
docker load < aov-gs-cuda117.tar.gz
cd AOV-GS && bash docker/run.sh
```
