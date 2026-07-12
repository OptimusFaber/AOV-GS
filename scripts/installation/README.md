# Environment setup

## Contents

1. [Quick start (one script)](#quick-start-one-script)
2. [`third_parties`](#third_parties)
3. [Conda manually](#conda-manually)
4. [Docker](#docker)

---

## Quick start (one script)

```bash
git clone --recursive https://github.com/OptimusFaber/AOV-GS
cd AOV-GS
bash scripts/installation/quick_start.sh
```

Does everything for a first run: submodules, conda, pip, SAM, Replica.

| Stage | Size (estimate) |
|------|-----------------|
| conda `aov-gs` | ~12 GB |
| Replica meshes + SLAM/Habitat/NVS | ~7–17 GB |
| SAM + third_parties | ~0.6 GB |
| **Total** | **≥ 30 GB** free space |

```bash
bash scripts/installation/quick_start.sh --help   # --skip-env, --skip-data, --yes, …
```

After it finishes: `conda activate aov-gs` → [pipeline](../aov-gs/README.md).

---

## `third_parties`

```bash
bash scripts/installation/setup_third_parties.sh
```

Or at clone time: `git clone --recursive …`  
Details: [third_parties/README.md](../../third_parties/README.md).

---

## Conda manually

**Python 3.8**, **CUDA 11.7**.

```bash
bash scripts/installation/conda_env/build_sem.sh
conda activate aov-gs
pip install open_clip_torch scikit-learn tqdm
pip install git+https://github.com/facebookresearch/segment-anything.git
bash scripts/installation/download_sam_ckpt.sh
```

Check: `python -c "import torch; print(torch.cuda.is_available())"`

---

## Docker

[docker/README.md](../../docker/README.md)
