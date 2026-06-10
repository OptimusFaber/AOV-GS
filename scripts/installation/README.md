# Установка окружения

## Содержание

1. [Conda (рекомендуется)](#conda-рекомендуется)
2. [HPC без libcuda](#hpc-без-libcuda)
3. [Open-vocabulary пакеты](#open-vocabulary-пакеты)
4. [Docker](#docker)

---

## Conda (рекомендуется)

**Python 3.8**, **CUDA 11.7** (или совместимый драйвер).

```bash
cd /path/to/AOV-GS
bash scripts/installation/conda_env/build_sem.sh
conda activate active-sgm
```

Скрипт устанавливает: PyTorch 1.13.1+cu117, habitat-sim (headless), pytorch3d, tiny-cuda-nn, diff-gaussian-rasterization-w-depth и зависимости SplaTAM.

Проверка:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## HPC без libcuda

Если `tinycudann` падает с `cannot find -lcuda` (типично для cds2 без sudo):

```bash
bash scripts/installation/conda_env/build_sem_cds2.sh
# или только фикс TCNN в уже созданном env:
bash scripts/installation/conda_env/build_sem_cds2.sh --fix-tcnn
```

---

## Open-vocabulary пакеты

После `build_sem.sh`:

```bash
pip install open_clip_torch
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install scikit-learn   # LangSplatV2 (KMeans)
pip install tqdm           # validate_lang_field_traj
```

**SAM checkpoint:** `ckpts/sam_vit_b_01ec64.pth`  
[Segment Anything — model checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)

**Channel rasterization** (семантический рендер с большим числом каналов, ActiveSem):  
[ActiveSGM — build CUDA tool](https://github.com/lly00412/ActiveSGM#build-cuda-tool-for-semantic-rendering).  
Для **`ActiveOpenSem` + языкового поля** обычно **не нужен**.

---

## Docker

См. [docker/README.md](../../docker/README.md) — образ CUDA 11.7, перенос на офлайн-HPC через `docker save/load`.
