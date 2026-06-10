# Скрипты пайплайна AOV-GS (`scripts/aov-gs/`)

Оркестраторы open-vocabulary: SLAM → (опционально AE) → языковое поле.

---

## Содержание

1. [Быстрый запуск](#быстрый-запуск)
2. [Этапы 01–03](#этапы-013)
3. [Пайплайны-обёртки](#пайплайны-обёртки)
4. [GPU и CUDA](#gpu-и-cuda)
5. [Batch-скрипты](#batch-скрипты)
6. [Выходные артефакты](#выходные-артефакты)

---

## Быстрый запуск

```bash
cd /path/to/AOV-GS
conda activate active-sgm

# Полный open-vocab (LangSplatV2 + SAM)
bash scripts/aov-gs/pipeline_gs_open_vocab.sh office0

# Только SLAM + SAM/CLIP
bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem 0 0 0
```

---

## Этапы 01–03

### 01 — SLAM + SAM/CLIP

```bash
bash scripts/aov-gs/01_slam_exploration.sh \
  [SCENE] [EXP] [SEED] [ENABLE_VIS] [DEBUG] [RESULT_RUN] [MASK_COLLECTOR]
```

| Аргумент | Default | Описание |
|----------|---------|----------|
| `SCENE` | `office0` | `office0`–`office4`, `room0`–`room2` |
| `EXP` | `ActiveOpenSemGeom` | `ActiveOpenSem`, `ActiveGeom`, `Passive`, … |
| `SEED` | `0` | random seed |
| `ENABLE_VIS` | `0` | `0` headless, `1` OpenCV-окна |
| `DEBUG` | `0` | `1` → JPEG в `keyframes/` |
| `RESULT_RUN` | *(auto)* | `run_0`, `run_1`, … |
| `MASK_COLLECTOR` | `sam` | `sam` или `corrclip` |

Примеры:

```bash
# ActiveOpenSem, headless, авто run_N
bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem 0 0 0

# Фиксированная папка + CorrCLIP
bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem 0 0 0 run_corrclip corrclip

# Эквивалент CorrCLIP (явный скрипт)
bash scripts/aov-gs/01_slam_exploration_with_corr_clip.sh office0 ActiveOpenSem 0 0 0 run_corrclip
```

Прямой Python:

```bash
python src/main/activesgm.py \
  --cfg configs/Replica/office0/ActiveOpenSem.py \
  --seed 0 --enable_vis 0 --corrclip 0 \
  --result_dir results/Replica/office0/ActiveOpenSem/run_0
```

### 02 — LangSplatV2: проверка фич

```bash
bash scripts/aov-gs/02_validate_features_langsplatv2.sh results/Replica/office0/ActiveOpenSem/run_0
```

### 02 — LangSplat (legacy): автоэнкодер

```bash
bash scripts/aov-gs/02_train_clip_autoencoder.sh \
  results/Replica/office0/ActiveOpenSem/run_0 [LATENT_DIM] [EPOCHS] [DEVICE]
# LATENT_DIM=64, EPOCHS=100, DEVICE=cuda:0
```

### 03 — языковое поле

**LangSplatV2** (рекомендуется):

```bash
bash scripts/aov-gs/03_train_gaussian_lang_field_langsplatv2.sh \
  RESULT_DIR K LEVEL NUM_ITERS DEVICE L TOPK RENDER_CKPT TRAIN_DOWNSCALE

# Пример:
bash scripts/aov-gs/03_train_gaussian_lang_field_langsplatv2.sh \
  results/Replica/office0/ActiveOpenSem/run_0 64 s 30000 cuda:0 1 4 auto 1.0
```

Все уровни SAM (`s`, `m`, `l`):

```bash
bash scripts/aov-gs/03_train_gaussian_lang_field_langsplatv2_all_levels.sh RESULT_DIR K NUM_ITERS
```

**LangSplat (legacy):**

```bash
bash scripts/aov-gs/03_train_gaussian_lang_field.sh \
  RESULT_DIR LATENT_DIM LEVEL NUM_ITERS DEVICE RENDER_CKPT
```

---

## Пайплайны-обёртки

| Скрипт | Конфиг | Описание |
|--------|--------|----------|
| `pipeline_gs_open_vocab.sh` | `ActiveOpenSem` | unified: SAM/CorrCLIP + LangSplat(V2) |
| `pipeline_gs_no_segmenter.sh` | `ActiveGS` | только геометрия |
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

## GPU и CUDA

### Сводка: что куда писать

| Компонент | Где задать | Параметр | Default |
|-----------|------------|----------|---------|
| **Видимые GPU** (shell) | env | `CUDA_VISIBLE_DEVICES` / `GPU` | `01_*.sh`: `GPU=0,1` |
| **SplaTAM** | конфиг | `primary_device` в `configs/Replica/replica_splatam_s.py` | `cuda:0` |
| **SAM + CLIP** | конфиг | `sam_clip.device` в `ActiveOpenSem_base.py` | `cuda:1` |
| **OneFormer** | конфиг | `semantic_device` в `ActiveSem.py` | `cuda:0` |
| **Lang field train** | аргумент shell | `DEVICE` в `03_*.sh` | `cuda:0` |
| **Python-скрипты** | CLI | `--device` | `cuda:0` |
| **NVS eval** | конфиг | `primary_device` (+ `CUDA_VISIBLE_DEVICES`) | `cuda:0` |

### Одна GPU

```bash
GPU=0 bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem
```

В конфиге `ActiveOpenSem_base.py` для одной карты поставьте `sam_clip.device = "cuda:0"` (или оставьте `cuda:1` — упадёт, если GPU одна).

### Две GPU: SLAM на 0, SAM/CLIP на 1

`01_slam_exploration.sh` по умолчанию: `GPU=0,1` → обе карты видны процессу.

В конфиге (уже так для `office0`):

```python
# configs/Replica/replica_splatam_s.py
primary_device = "cuda:0"

# configs/Replica/office0/ActiveOpenSem_base.py
sam_clip = dict(device = "cuda:1", ...)
```

Запуск:

```bash
GPU=0,1 bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem
```

### Обучение language field на физической GPU 1

`diff_gaussian_rasterization` видит только **logical `cuda:0`**. Shell-скрипты `03_*.sh` делают remapping через `_gpu_helpers.sh`:

```bash
# Физическая GPU 1 → передать cuda:1 в аргумент DEVICE
bash scripts/aov-gs/03_train_gaussian_lang_field_langsplatv2.sh \
  results/.../run_0 64 s 30000 cuda:1 1 4 auto 1.0
# внутри: CUDA_VISIBLE_DEVICES=1, python --device cuda:0
```

Вручную:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_language_field.py ... --device cuda:0
```

**Не** запускайте `train_language_field.py --device cuda:1` без `CUDA_VISIBLE_DEVICES`.

### Python-скрипты с `--device`

| Скрипт | Назначение |
|--------|------------|
| `validate_lang_field_traj.py` | mIoU на traj |
| `compute_miou_p_traj.py` | mIoU_p (SAM pseudo) |
| `query_language_field.py` | текстовый запрос |
| `render_view_from_pose.py` | один кадр NVS |
| `render_query_from_pose.py` | запрос + рендер |
| `train_language_field.py` | обучение (см. remapping выше) |

Пример на GPU 1 без remapping (валидация, не rasterizer):

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/validate_lang_field_traj.py \
  --scene office0 --result_dir results/.../run_0 \
  --traj_txt data/replica_sim_nvs/office0/traj.txt \
  --device cuda:0
```

### VRAM / `--render_checkpoint`

| Значение | Поведение |
|----------|-----------|
| `off` | быстрее, больше VRAM |
| `on` | gradient checkpoint, меньше VRAM |
| `auto` | эвристика (default в `03_*.sh`) |

При OOM на этапе 3: `RENDER_CKPT=on` или уменьшить `TRAIN_DOWNSCALE` (например `0.5`).

---

## Batch-скрипты

| Скрипт | Env | Назначение |
|--------|-----|------------|
| `run_lang_field_batch_replica.sh` | `GPUS="0 1"` | обучение lang field на всех сценах |
| `run_lang_field_validate_batch.sh` | `GPUS`, `VALIDATE_SLOTS_PER_GPU=4` | batch mIoU |
| `run_multiseed_geom_open_sem_all.sh` | `GPUS`, `GEOM_PER_GPU`, `OPEN_SEM_PER_GPU` | multiseed SLAM |
| `run_active_geom_all_scenes.sh` | `GPU` | ActiveGeom по сценам |
| `run_lang_field_full_grid.sh` | — | сетка AE (legacy) |
| `run_lang_field_full_grid_langsplatv2.sh` | — | сетка V2 |

---

## Выходные артефакты

После этапа 01 (`ActiveOpenSem`):

```
results/Replica/<scene>/ActiveOpenSem/run_N/
├── splatam/final/params0.npz      # или params.npz
├── keyframe_poses.json
├── language_features/             # *_f.npy, *_s.npy
└── main_cfg.json
```

После этапа 03 (LangSplatV2):

```
lang_field_sk64_l1/lang_field.pt
```
