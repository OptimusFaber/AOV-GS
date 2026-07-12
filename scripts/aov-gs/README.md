# AOV-GS pipeline scripts (`scripts/aov-gs/`)

Open-vocabulary orchestrators: SLAM → (optional AE) → language field.

---

## Contents

1. [Quick run](#quick-run)
2. [Stages 01–03](#stages-0103)
3. [Wrapper pipelines](#wrapper-pipelines)
4. [GPU and CUDA](#gpu-and-cuda)
5. [Batch scripts](#batch-scripts)
6. [Output artifacts](#output-artifacts)

---

## Quick run

```bash
cd /path/to/AOV-GS
conda activate aov-gs

# Full open-vocab (LangSplatV2 + SAM)
bash scripts/aov-gs/pipeline_gs_open_vocab.sh office0

# SLAM + SAM/CLIP only
bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem 0 0 0
```

---

## Stages 01–03

### 01 — SLAM + SAM/CLIP

```bash
bash scripts/aov-gs/01_slam_exploration.sh \
  [SCENE] [EXP] [SEED] [ENABLE_VIS] [DEBUG] [RESULT_RUN] [MASK_COLLECTOR]
```

| Argument | Default | Description |
|----------|---------|----------|
| `SCENE` | `office0` | `office0`–`office4`, `room0`–`room2` |
| `EXP` | `ActiveOpenSemGeom` | `ActiveOpenSem`, `ActiveGeom`, `Passive`, … |
| `SEED` | `0` | random seed |
| `ENABLE_VIS` | `0` | `0` headless, `1` OpenCV windows |
| `DEBUG` | `0` | `1` → JPEG under `keyframes/` |
| `RESULT_RUN` | *(auto)* | `run_0`, `run_1`, … |
| `MASK_COLLECTOR` | `sam` | `sam` or `corrclip` |

Examples:

```bash
# ActiveOpenSem, headless, auto run_N
bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem 0 0 0

# Fixed folder + CorrCLIP
bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem 0 0 0 run_corrclip corrclip

# Equivalent CorrCLIP (explicit script)
bash scripts/aov-gs/01_slam_exploration_with_corr_clip.sh office0 ActiveOpenSem 0 0 0 run_corrclip
```

Direct Python:

```bash
python src/main/activesgm.py \
  --cfg configs/Replica/office0/ActiveOpenSem.py \
  --seed 0 --enable_vis 0 --corrclip 0 \
  --result_dir results/Replica/office0/ActiveOpenSem/run_0
```

### 02 — LangSplatV2: feature validation

```bash
bash scripts/aov-gs/02_validate_features_langsplatv2.sh results/Replica/office0/ActiveOpenSem/run_0
```

### 02 — LangSplat (legacy): autoencoder

```bash
bash scripts/aov-gs/02_train_clip_autoencoder.sh \
  results/Replica/office0/ActiveOpenSem/run_0 [LATENT_DIM] [EPOCHS] [DEVICE]
# LATENT_DIM=64, EPOCHS=100, DEVICE=cuda:0
```

### 03 — language field

**LangSplatV2** (recommended):

```bash
bash scripts/aov-gs/03_train_gaussian_lang_field_langsplatv2.sh \
  RESULT_DIR K LEVEL NUM_ITERS DEVICE L TOPK RENDER_CKPT TRAIN_DOWNSCALE

# Example:
bash scripts/aov-gs/03_train_gaussian_lang_field_langsplatv2.sh \
  results/Replica/office0/ActiveOpenSem/run_0 64 s 30000 cuda:0 1 4 auto 1.0
```

All SAM levels (`s`, `m`, `l`):

```bash
bash scripts/aov-gs/03_train_gaussian_lang_field_langsplatv2_all_levels.sh RESULT_DIR K NUM_ITERS
```

**LangSplat (legacy):**

```bash
bash scripts/aov-gs/03_train_gaussian_lang_field.sh \
  RESULT_DIR LATENT_DIM LEVEL NUM_ITERS DEVICE RENDER_CKPT
```

---

## Wrapper pipelines

| Script | Config | Description |
|--------|--------|----------|
| `pipeline_gs_open_vocab.sh` | `ActiveOpenSem` | unified: SAM/CorrCLIP + LangSplat(V2) |
| `pipeline_gs_no_segmenter.sh` | `ActiveGS` | geometry only |
| `pipeline_gs_oneformer.sh` | `ActiveSem` | OneFormer (closed-set) |
| `pipeline_gs_langsplatv2.sh` | `ActiveOpenSem` | 01 → 02 validate → 03 V2 |
| `pipeline_gs_langsplat.sh` | `ActiveOpenSem` | 01 → 02 AE → 03 legacy |

`pipeline_gs_open_vocab.sh`:

```bash
bash scripts/aov-gs/pipeline_gs_open_vocab.sh \
  [SCENE] [SEED] [ENABLE_VIS] [DEBUG] [MASK_COLLECTOR] [LANG_MODE] [RESULT_RUN] [train args…]

# CorrCLIP + LangSplatV2
bash scripts/aov-gs/pipeline_gs_open_vocab.sh office0 0 0 0 corrclip langsplatv2 run_corrclip
```

---

## GPU and CUDA

### Summary: what to set where

| Component | Where to set | Parameter | Default |
|-----------|------------|----------|---------|
| **Visible GPUs** (shell) | env | `CUDA_VISIBLE_DEVICES` / `GPU` | `01_*.sh`: `GPU=0,1` |
| **SplaTAM** | config | `primary_device` in `configs/Replica/replica_splatam_s.py` | `cuda:0` |
| **SAM + CLIP** | config / env | `sam_clip.device` or `SAM_CLIP_DEVICE` | `cuda:0` |
| **OneFormer** | config | `semantic_device` in `ActiveSem.py` | `cuda:0` |
| **Lang field train** | shell argument | `DEVICE` in `03_*.sh` | `cuda:0` |
| **Python scripts** | CLI | `--device` | `cuda:0` |
| **NVS eval** | config | `primary_device` (+ `CUDA_VISIBLE_DEVICES`) | `cuda:0` |

### One GPU

```bash
GPU=0 bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem
```

By default SAM+CLIP runs on **`cuda:0`** (same as SplaTAM).

### Two GPUs: SLAM on 0, SAM/CLIP on 1

`01_slam_exploration.sh` by default: `GPU=0,1` → both cards are visible to the process.

In the config:

```python
# configs/Replica/replica_splatam_s.py
primary_device = "cuda:0"

# configs/Replica/<scene>/ActiveOpenSem_base.py
sam_clip = dict(device = "cuda:1", ...)   # only with two GPUs
```

Or without editing the config: `SAM_CLIP_DEVICE=cuda:1`.

Run:

```bash
GPU=0,1 bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem
```

### Training the language field on physical GPU 1

`diff_gaussian_rasterization` only sees **logical `cuda:0`**. The `03_*.sh` shell scripts remap via `_gpu_helpers.sh`:

```bash
# Physical GPU 1 → pass cuda:1 as the DEVICE argument
bash scripts/aov-gs/03_train_gaussian_lang_field_langsplatv2.sh \
  results/.../run_0 64 s 30000 cuda:1 1 4 auto 1.0
# inside: CUDA_VISIBLE_DEVICES=1, python --device cuda:0
```

Manually:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_language_field.py ... --device cuda:0
```

**Do not** run `train_language_field.py --device cuda:1` without `CUDA_VISIBLE_DEVICES`.

### Python scripts with `--device`

| Script | Purpose |
|--------|------------|
| `validate_lang_field_traj.py` | mIoU on traj |
| `compute_miou_p_traj.py` | mIoU_p (SAM pseudo) |
| `query_language_field.py` | text query |
| `render_view_from_pose.py` | single NVS frame |
| `render_query_from_pose.py` | query + render |
| `train_language_field.py` | training (see remapping above) |

Example on GPU 1 without remapping (validation, not the rasterizer):

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/validate_lang_field_traj.py \
  --scene office0 --result_dir results/.../run_0 \
  --traj_txt data/replica_sim_nvs/office0/traj.txt \
  --device cuda:0
```

### VRAM / `--render_checkpoint`

| Value | Behavior |
|----------|-----------|
| `off` | faster, more VRAM |
| `on` | gradient checkpoint, less VRAM |
| `auto` | heuristic (default in `03_*.sh`) |

On OOM at stage 3: set `RENDER_CKPT=on` or reduce `TRAIN_DOWNSCALE` (e.g. `0.5`).

---

## Batch scripts

| Script | Env | Purpose |
|--------|-----|------------|
| `run_lang_field_batch_replica.sh` | `GPUS="0 1"` | train lang field on all scenes |
| `run_lang_field_validate_batch.sh` | `GPUS`, `VALIDATE_SLOTS_PER_GPU=4` | batch mIoU |
| `run_multiseed_geom_open_sem_all.sh` | `GPUS`, `GEOM_PER_GPU`, `OPEN_SEM_PER_GPU` | multiseed SLAM |
| `run_active_geom_all_scenes.sh` | `GPU` | ActiveGeom across scenes |
| `run_lang_field_full_grid.sh` | — | AE grid (legacy) |
| `run_lang_field_full_grid_langsplatv2.sh` | — | V2 grid |

---

## Output artifacts

After stage 01 (`ActiveOpenSem`):

```
results/Replica/<scene>/ActiveOpenSem/run_N/
├── splatam/final/params0.npz      # or params.npz
├── keyframe_poses.json
├── language_features/             # *_f.npy, *_s.npy
└── main_cfg.json
```

After stage 03 (LangSplatV2):

```
lang_field_sk64_l1/lang_field.pt
```
