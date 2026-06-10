# Docker для AOV-GS

См. также: [корневой README](../README.md), [установка conda](../scripts/installation/README.md), [пайплайн](../scripts/aov-gs/README.md).

Образ: **CUDA 11.7**, Python 3.8, PyTorch 1.13.1+cu117, Habitat, tiny-cuda-nn, diff-gaussian, SAM/CLIP.  
Conda: **Miniforge** (conda-forge), без Anaconda ToS. PyTorch через pip.

## Требования

- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- `nvidia-smi` на хосте
- ~15–25 GB для образа, сборка **1–2 ч** (интернет на машине **сборки**)

## Сборка

```bash
cd AOV-GS

# (опционально) wheels для pytorch3d/PyG — если Meta CDN тормозит
bash docker/wheels/download_wheels.sh

# обычная
bash docker/build.sh

# с прокси (на машине где есть доступ)
HTTP_PROXY=http://USER:PASS@HOST:3128 \
HTTPS_PROXY=http://USER:PASS@HOST:3128 \
bash docker/build.sh

# только OpenSem/Hybrid (без ActiveSem channel rasterizers)
BUILD_ACTIVESEM=0 bash docker/build.sh
```

Или compose:

```bash
docker compose -f docker/docker-compose.yml build
```

## Запуск

```bash
bash docker/run.sh

# внутри контейнера уже active-sgm
python -c "import torch; print(torch.cuda.is_available())"

bash scripts/ablation/run_activesgm_benchmark.sh office0
```

Одной командой:

```bash
bash docker/run.sh python src/main/activesgm.py \
  --cfg configs/Replica/office0/ActiveOpenSem.py --seed 0 --enable_vis 0
```

## Данные

Код монтируется с хоста (`./` → `/workspace/AOV-GS`).  
Тома Docker: `data/`, `results/`, `ckpts/` (сохраняются между запусками).

Положи Replica data в `data/` на хосте или примонтируй свой путь в `docker-compose.yml`.

## Перенос образа на cds2 без интернета

На машине **со сборкой**:

```bash
docker save aov-gs:cuda117 | gzip > aov-gs-cuda117.tar.gz
scp aov-gs-cuda117.tar.gz user@cds2:~/
```

На **cds2**:

```bash
docker load < aov-gs-cuda117.tar.gz
cd AOV-GS && bash docker/run.sh
```

Сборка env на cds2 **не нужна** — только Docker + GPU driver.
