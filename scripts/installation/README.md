# Установка окружения

## Содержание

1. [Быстрый старт (один скрипт)](#быстрый-старт-один-скрипт)
2. [`third_parties`](#third_parties)
3. [Conda вручную](#conda-вручную)
4. [HPC без libcuda](#hpc-без-libcuda)
5. [Docker](#docker)

---

## Быстрый старт (один скрипт)

```bash
git clone --recursive https://github.com/OptimusFaber/AOV-GS
cd AOV-GS
bash scripts/installation/quick_start.sh
```

Делает всё для первого запуска: submodules, conda, pip, SAM, Replica.

| Этап | Размер (оценка) |
|------|-----------------|
| conda `active-sgm` | ~12 GB |
| Replica meshes + SLAM/Habitat/NVS | ~7–17 GB |
| SAM + third_parties | ~0.6 GB |
| **Итого** | **≥ 30 GB** свободного места |

```bash
bash scripts/installation/quick_start.sh --help   # --skip-env, --skip-data, --yes, …
```

После завершения: `conda activate active-sgm` → [пайплайн](../aov-gs/README.md).

---

## `third_parties`

```bash
bash scripts/installation/setup_third_parties.sh
```

Или при клоне: `git clone --recursive …`  
Подробнее: [third_parties/README.md](../../third_parties/README.md).

---

## Conda вручную

**Python 3.8**, **CUDA 11.7**.

```bash
bash scripts/installation/conda_env/build_sem.sh
conda activate active-sgm
pip install open_clip_torch scikit-learn tqdm
pip install git+https://github.com/facebookresearch/segment-anything.git
bash scripts/installation/download_sam_ckpt.sh
```

Проверка: `python -c "import torch; print(torch.cuda.is_available())"`

---

## HPC без libcuda

```bash
bash scripts/installation/conda_env/build_sem_cds2.sh
# или: bash scripts/installation/conda_env/build_sem_cds2.sh --fix-tcnn
```

---

## Docker

[docker/README.md](../../docker/README.md)
