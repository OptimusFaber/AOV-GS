# AOV-GS — Active Open-Vocabulary 3D Gaussian Splatting

**Краткое имя для репозитория:** `aov-gs` или `OAVGS` (Open / Active Vocabulary + Gaussian Splatting).

Ветка на базе [ActiveSGM](https://github.com/lly00412/ActiveSGM): активное исследование сцены (**SplaTAM**, 3D Gaussian Splatting) и **открытое языковое поле**. На каждом этапе можно выбрать:

- **Сбор масок:** обычный SAM+CLIP (`sam`) или **CorrCLIP**-постобработка (`corrclip`);
- **Языковое поле:** **LangSplatV2** (codebook, рекомендуется) или **LangSplat** (legacy + автоэнкодер).

Поддерживаются **пять пайплайнов** (см. `scripts/aov-gs/pipeline_gs_*.sh`): геометрия, OneFormer, unified open-vocab, LangSplat и LangSplatV2.

---

## Содержание

1. [Клонирование и `third_parties`](#клонирование-и-third_parties)
2. [Окружение](#окружение)
3. [Данные Replica + Habitat](#данные-replica--habitat)
4. [Чекпойнты SAM / CLIP](#чекпойнты-sam--clip)
5. [Настройка путей в конфигах](#настройка-путей-в-конфигах)
6. [Выбор GPU](#выбор-gpu)
7. [Пайплайны Gaussian Splatting](#пайплайны-gaussian-splatting)
8. [Выбор mask collector и lang mode](#выбор-mask-collector-и-lang-mode)
9. [Инференс и отладка (новые скрипты)](#инференс-и-отладка-новые-скрипты)
10. [Валидация mIoU на traj](#валидация-miou-на-traj)
11. [Результаты](#результаты)
12. [VRAM и `--render_checkpoint`](#vram-и---render_checkpoint)

---

## Клонирование и `third_parties`

Каталог **`third_parties/`** в **`.gitignore`** (там же лежит очень большой `habitat_sim` и вложенные git-репозитории). После `git clone` его нужно получить отдельно.

**Вариант A — как в upstream ActiveSGM:** склонировать [ActiveSGM](https://github.com/lly00412/ActiveSGM) с `--recursive` и скопировать каталог `third_parties/` в корень этого проекта.

**Вариант B:** в корне лежит **`.gitmodules`** со списком URL (SplaTAM, Co-SLAM, neural_slam_eval, semantic-gaussians на двух ветках). Пока в индексе нет gitlink’ов submodule, проще всего **вручную** клонировать каждый репозиторий в указанный подкаталог `third_parties/...` с нужной веткой (как в ActiveSGM), либо один раз склонировать ActiveSGM с `--recursive` и скопировать оттуда весь `third_parties/`. **Habitat** для рантайма обычно ставят через **conda** (`build_sem.sh`), а не из `third_parties/habitat_sim`.

Без заполненного `third_parties/` не заработают импорты вроде `third_parties/splatam` (см. `src/slam/splatam/splatam.py`).

---

## Окружение

Рекомендации как у ActiveSGM: **Python 3.8**, **CUDA 11.7** (или близкий стек под ваш драйвер). Полная сборка conda:

```bash
cd /path/to/aov-gs   # корень репозитория
bash scripts/installation/conda_env/build_sem.sh
conda activate active-sgm
```

Скрипт ставит PyTorch 1.13.1 + CUDA 11.7, **habitat-sim** (headless), **pytorch3d**, **tiny-cuda-nn**, **diff-gaussian-rasterization-w-depth**, CCCL и прочие зависимости SplaTAM.

На HPC без sudo (например **cds2**), если `tinycudann` падает с `cannot find -lcuda`, используйте `build_sem_cds2.sh` (тот же стек + фикс линковки `libcuda`) или только дочинку: `bash scripts/installation/conda_env/build_sem_cds2.sh --fix-tcnn`.

### Open-vocabulary (этап 1–3): SAM + CLIP

После `build_sem.sh` установите пакеты для извлечения масок и CLIP:

```bash
pip install open_clip_torch
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Для **LangSplatV2** (инициализация codebook через KMeans) добавьте:

```bash
pip install scikit-learn
```

Положите вес SAM (см. следующий раздел) и при необходимости задайте кэш OpenCLIP (`OPENCLIP_CACHE` / загрузки по сети).

### Дополнительные CUDA-расширения (полный ActiveSGM)

Если запускаете **семантический** рендер с большим числом каналов (не минимальный open-vocab пайплайн), см. официальную инструкцию ActiveSGM: [dense / sparse channel rasterization](https://github.com/lly00412/ActiveSGM#build-cuda-tool-for-semantic-rendering). Для конфигов **`ActiveOpenSem`** + языкового поля через `diff_gaussian_rasterization` это часто **не требуется**.

---

## Данные Replica + Habitat

Пайплайн ориентирован на **Replica** в симуляторе **Habitat**, по шагам как в [разделе Data preparation в ActiveSGM](https://github.com/lly00412/ActiveSGM/blob/main/README.md#data-preparation).

### Скачивание Replica

- Датасет: [Replica (Facebook Research)](https://github.com/facebookresearch/Replica-Dataset).

```bash
bash scripts/data/replica_download.sh data/replica_v1
bash scripts/data/replica_update.sh data/replica_v1
```

### Три варианта данных (как в ActiveSGM)

1. **ReplicaSLAM** — траектории для инициализации / пассивного режима.  
2. **ReplicaSLAM-Habitat** — те же траектории, RGB-D из Habitat (согласовано с симулятором).  
3. **ReplicaNVS** — для оценки novel view synthesis (если используете соответствующие скрипты).

```bash
bash scripts/data/replica_slam_download.sh
bash scripts/data/generate_replica_habitat.sh all
bash scripts/data/generate_replica_nvs.sh all
```

Нужны ассеты Replica, данные после `generate_replica_habitat` и согласованные с `configs/Replica/<scene>/habitat.py` симлинки / структура под `data/Replica/<scene>/`. Подробности — в ActiveSGM и [ActiveGAMER](https://github.com/oppo-us-research/ActiveGAMER).

---

## Чекпойнты SAM / CLIP

В конфиге `ActiveOpenSem` задаётся путь к SAM, например:

```python
sam_ckpt_path = "ckpts/sam_vit_b_01ec64.pth"
```

Создайте `ckpts/` в корне репозитория и положите туда `sam_vit_b_01ec64.pth` ([официальная ссылка Segment Anything](https://github.com/facebookresearch/segment-anything#model-checkpoints)).

Веса **OpenCLIP** подтягиваются при первом запуске или через кэш; для офлайн-сетапа см. переменные в разделе бенчмарка ниже.

---

## Настройка путей в конфигах

1. Откройте `configs/Replica/<scene>/habitat.py` и выставьте **`scene_id`** на ваш `mesh_semantic.ply` / stage config в дереве Replica.  
2. Убедитесь, что `data/Replica/<scene>/` содержит нужные `results` / `results_habitat` и `traj.txt` (см. `replica.py` в датасете).  
3. **`ActiveOpenSem.py`** добавлен для **всех** сцен Replica с `habitat.py` в этом репозитории: **`office0` … `office4`**, **`room0` … `room2`**. В каждом файле заданы `general.scene` и **`bbox_bound`**, согласованные с соответствующим `ActiveSem.py` той же сцены.
4. Если RGB-кадры из `generate_replica_nvs.sh` выглядят «пересвеченными»/сильно искаженными, проверьте `data/replica_v1/<scene>/habitat/replicaSDK_stage.stage_config.json`:  
   - `render_asset` должен указывать на **текстурный меш** `../mesh.ply`;  
   - `semantic_asset` должен указывать на `mesh_semantic.ply`.  
   Типичный симптом в логе при ошибке: `with render asset : .../mesh_semantic.ply`. В этом случае симулятор рендерит semantic mesh как RGB.

Выбор GPU — в [следующем разделе](#выбор-gpu).

---

## Выбор GPU

**По умолчанию всё на `cuda:0`.** Можно держать SLAM и сегментатор на одной карте или разнести по двум.

### Сводка: что где задаётся

| Компонент | Параметр | Файл / способ | Default |
|-----------|----------|---------------|---------|
| **SLAM** (SplaTAM / SemSplaTAM) | `primary_device` | `configs/Replica/replica_splatam_s.py` или `slam.override` | `cuda:0` |
| **OneFormer** (ActiveSem) | `semantic_device` | `configs/Replica/<scene>/ActiveSem.py` | `cuda:0` |
| **SAM + CLIP** (ActiveOpenSem / Hybrid) | `sam_clip.device` | `configs/Replica/<scene>/ActiveOpenSem.py` | `cuda:0` |
| **Language field** (offline train) | `DEVICE` (5-й аргумент) | `03_train_gaussian_lang_field_v2.sh` | `cuda:0` |
| **Traj mIoU (lang field)** | `--device` | `validate_lang_field_traj.py` | `cuda:0` |
| **Traj mIoU_p (SAM+CLIP pseudo)** | `--device` | `compute_miou_p_traj.py` | `cuda:0` |
| **Инференс / рендер / query** | `--device` | `query_language_field.py`, `render_*_from_pose.py`, … | `cuda:0` |
| **Ablation / pipeline** | `GPU` | `CUDA_VISIBLE_DEVICES` в shell | `0` |

### Одна GPU (рекомендуется для отладки)

```bash
GPU=0 bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem
```

В конфигах по умолчанию: `primary_device=cuda:0`, `sam_clip.device=cuda:0`, `semantic_device=cuda:0`.

### Две GPU: SLAM + сегментатор раздельно

```bash
GPU=0,1 \
SLAM_DEVICE=cuda:0 \
SEM_DEVICE=cuda:1 \
bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem
```

| Алгоритм | SLAM | Сегментатор | Ключ в конфиге |
|----------|------|-------------|----------------|
| **ActiveSem** | `cuda:0` | OneFormer → `cuda:1` | `semantic_device` |
| **ActiveOpenSem / Hybrid** | `cuda:0` | SAM+CLIP → `cuda:1` | `sam_clip.device` |

Override **без правки конфига** (CLI или env):

```bash
python src/main/activesgm.py \
  --cfg configs/Replica/office0/ActiveOpenSem.py \
  --slam_device cuda:0 \
  --seg_device cuda:1 \
  --result_dir results/Replica/office0/ActiveOpenSem/run_0
```

Эквивалент через env: `ACTIVESGM_SLAM_DEVICE`, `ACTIVESGM_SEM_DEVICE`.

### Language field на другой GPU

Rasterizer (`diff_gaussian_rasterization`) работает только как **logical `cuda:0`**. Для физической GPU 1 передайте `cuda:1` — скрипт сам сделает remapping:

```bash
bash scripts/aov-gs/03_train_gaussian_lang_field_v2.sh \
  results/Replica/office0/ActiveOpenSem/run_0 \
  64 m 12000 cuda:1 1 4 auto 1.0
# → CUDA_VISIBLE_DEVICES=1, в Python --device cuda:0
```

Или явно:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/aov-gs/03_train_gaussian_lang_field_v2.sh ... cuda:0 ...
```

**Не запускайте** `python scripts/train_language_field.py --device cuda:1` напрямую без `CUDA_VISIBLE_DEVICES`.

### `--device` в Python-скриптах

Все GPU-скрипты в `scripts/` принимают **`--device`** (по умолчанию `cuda:0`). Полный список:

| Скрипт | `--device` |
|--------|------------|
| `validate_lang_field_traj.py` | traj mIoU (LangSplatV2) |
| `compute_miou_p_traj.py` | mIoU_p (SAM+CLIP pseudo, без lang field) |
| `train_language_field.py` | обучение lang field (см. remapping выше) |
| `query_language_field.py` | open-vocab query / heatmap |
| `render_view_from_pose.py`, `render_query_from_pose.py` | один кадр |
| `render_keyframe_poses.py`, `pose_visibility_from_traj.py` | poses / visibility |
| `train_language_autoencoder.py`, `eval_language_autoencoder.py` | AE |
| `eval_lang_field_segmentation.py` | legacy (один level) |
| `eval_clip_sam_miou.py`, `sam_query_match.py`, `demo_sam_clip_text_query.py` | 2D SAM+CLIP |
| `aov-gs/eval_run_lang_fields.py`, `eval_run_autoencoders.py` | batch eval |

Пример — валидация на физической GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/validate_lang_field_traj.py \
  --scene office0 \
  --result_dir results/Replica/office0/ActiveOpenSem/run_0 \
  --traj_txt data/replica_sim_nvs/office0/traj.txt \
  --device cuda:0
```

Или без remapping: `--device cuda:1` (для скриптов **без** rasterizer remapping — валидация, query, SAM+CLIP).

`run_nvs_validation.py` и `eval_semsplatam_per_class_miou.py` берут GPU из **`primary_device`** / **`semantic_device`** в конфиге (или `CUDA_VISIBLE_DEVICES`).

### Ablation

```bash
GPU=0      bash scripts/ablation/run_activesgm_benchmark.sh
GPU=0,1    SLAM_DEVICE=cuda:0 SEM_DEVICE=cuda:1 bash scripts/ablation/run_scene_lang_pipeline.sh office0
```

---

## Пайплайны Gaussian Splatting

Все оркестраторы лежат в **`scripts/aov-gs/`**. Перед запуском: `cd` в корень репозитория, conda-окружение активировано, данные Replica подготовлены.

| № | Скрипт | Конфиг | Что получаете |
|---|--------|--------|----------------|
| 0 | **`pipeline_gs_open_vocab.sh`** | `ActiveOpenSem` | **Unified:** выбор `MASK_COLLECTOR` + `LANG_MODE` в одной команде |
| 1 | `pipeline_gs_no_segmenter.sh` | `ActiveGS` | Только SplaTAM, без `language_features/` |
| 2 | `pipeline_gs_oneformer.sh` | `ActiveSem` | OneFormer, без open-vocab SAM+CLIP |
| 3 | `pipeline_gs_langsplat.sh` | `ActiveOpenSem` | SAM+CLIP → AE → **LangSplat (legacy)** |
| 4 | `pipeline_gs_langsplatv2.sh` | `ActiveOpenSem` | SAM+CLIP → **LangSplatV2** (без AE) |

### Unified open-vocab (рекомендуется)

```bash
# LangSplatV2 + plain SAM (defaults)
bash scripts/aov-gs/pipeline_gs_open_vocab.sh office0

# LangSplatV2 + CorrCLIP, отдельная папка результатов
bash scripts/aov-gs/pipeline_gs_open_vocab.sh office0 0 0 0 corrclip langsplatv2 run_corrclip

# Legacy LangSplat + AE
bash scripts/aov-gs/pipeline_gs_open_vocab.sh office0 0 0 0 sam langsplat run_ls 64 100 s 30000 cuda:0 auto
```

Аргументы: `[SCENE] [SEED] [ENABLE_VIS] [DEBUG] [MASK_COLLECTOR] [LANG_MODE] [RESULT_RUN] [train args…]`

### Этап 1 — аргументы `01_slam_exploration.sh`

`[SCENE] [EXP] [SEED] [ENABLE_VIS] [DEBUG] [RESULT_RUN] [MASK_COLLECTOR]`

**GPU (env):** `GPU` (default `0`), `SLAM_DEVICE`, `SEM_DEVICE` — см. [Выбор GPU](#выбор-gpu).

| `MASK_COLLECTOR` | CLI | Поведение |
|------------------|-----|-----------|
| `sam` (default) | `--corrclip 0` | Обычный SAM: все маски без merge/suppression |
| `corrclip` | `--corrclip 1` | CorrCLIP-стиль: merge близких масок + inter-class suppression (пороги в `[sam_clip]` конфига) |

Удобные обёртки: `01_slam_exploration_with_corr_clip.sh` ≡ `MASK_COLLECTOR=corrclip`.

Прямой Python:

```bash
python src/main/activesgm.py \
  --cfg configs/Replica/office0/ActiveOpenSem.py \
  --result_dir results/Replica/office0/ActiveOpenSem/run_0 \
  --corrclip 0   # sam
# --corrclip 1   # corrclip (из конфига)
```

### 1) Без сегментатора

```bash
cd /path/to/AOV-GS
bash scripts/aov-gs/pipeline_gs_no_segmenter.sh office0 0 0 0
```

Результаты: `results/Replica/office0/ActiveGS/run_0/`. Дальнейших шагов для языкового поля нет.

### 2) С OneFormer

Требуется файл `configs/Replica/<scene>/ActiveSem.py` (для `office0` он уже есть). Веса OneFormer задаются в конфиге (`oneformer_checkpoint`, и т.д.) — нужен доступ к HuggingFace или локальные кэши.

```bash
bash scripts/aov-gs/pipeline_gs_oneformer.sh office0 0 0 0
```

Результаты: `results/Replica/office0/ActiveSem/run_0/`. Это **не** open‑vocab CLIP+SAM пайплайн; для языкового поля по тексту используйте режимы 3–4.

### 3) CLIP + SAM + LangSplat (с автоэнкодером)

Одной командой (после неё уже выполнены этапы 01 → 02 → 03):

```bash
bash scripts/aov-gs/pipeline_gs_langsplat.sh
# или с явными параметрами:
# bash scripts/aov-gs/pipeline_gs_langsplat.sh office0 0 0 0 64 100 cuda:0 s 30000 auto
```

Выход языкового поля: `results/Replica/office0/ActiveOpenSem/run_0/lang_field_s64/lang_field.pt` (при уровне `s` и `LATENT_DIM=64`).

**Пошагово вручную:**

| Этап | Выход |
|------|--------|
| 1 | `splatam/final/params*.npz`, `keyframe_poses.json`, `language_features/*_{s,f}.npy` |
| 2 | `language_features_dim{N}/`, `ckpt/<scene>/best_ckpt.pth` |
| 3 | `lang_field_<level>{N}/lang_field.pt` |

```bash
bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem 0 0 0
bash scripts/aov-gs/02_train_clip_autoencoder.sh results/Replica/office0/ActiveOpenSem/run_0 64 100 cuda:0
bash scripts/aov-gs/03_train_gaussian_lang_field.sh results/Replica/office0/ActiveOpenSem/run_0 64 s 30000 cuda:0 auto
```

Сетка по размерностям AE: `bash scripts/aov-gs/run_lang_field_full_grid.sh results/Replica/office0/ActiveOpenSem/run_0` (нужны папки `language_features_dim3` … `language_features_dim64`).

### 4) CLIP + SAM + LangSplatV2

```bash
bash scripts/aov-gs/pipeline_gs_langsplatv2.sh
# полный пример:
# bash scripts/aov-gs/pipeline_gs_langsplatv2.sh office0 0 0 0 64 s 30000 cuda:0 1 4 auto 1.0
```

Шаг **02** для V2 — это `02_validate_features_langsplatv2.sh` (проверка `language_features/`, без обучения AE). Шаг **03** — `03_train_gaussian_lang_field_langsplatv2.sh` (аргументы: `RESULT_DIR`, `K`, `LEVEL`, `NUM_ITERS`, `DEVICE`, `L`, `TOPK`, `RENDER_CHECKPOINT`, `TRAIN_DOWNSCALE`).

Все уровни SAM подряд: `bash scripts/aov-gs/03_train_gaussian_lang_field_langsplatv2_all_levels.sh results/.../run_0`.

Сетка V2: `bash scripts/aov-gs/run_lang_field_full_grid_langsplatv2.sh results/Replica/office0/ActiveOpenSem/run_0` (переменные `GRID_K_VALUES`, `GRID_L_VALUES`, см. заголовок скрипта).

**Прямой вызов Python (V2):**

```bash
python scripts/train_language_field.py \
  --checkpoint    results/Replica/office0/ActiveOpenSem/run_0/splatam/final/params0.npz \
  --poses         results/Replica/office0/ActiveOpenSem/run_0/keyframe_poses.json \
  --features_dir  results/Replica/office0/ActiveOpenSem/run_0/language_features \
  --level         s \
  --output_dir    results/Replica/office0/ActiveOpenSem/run_0/lang_field_sk64_l1 \
  --codebook_size 64 --vq_layer_num 1 --topk 4 \
  --num_iters     30000 --device cuda:0 --render_checkpoint auto
```

Старый режим: `--lang_mode langsplat` (или `--legacy`) + `language_features_dim*` после AE.

---

## Выбор mask collector и lang mode

### Обучение (`train_language_field.py`)

| Флаг | Значения | Описание |
|------|----------|----------|
| `--lang_mode` | `langsplatv2` (default), `langsplat` | V2: codebook + `language_features/`; legacy: AE + `language_features_dim*` |
| `--legacy` | flag | Алиас `--lang_mode langsplat` |

```bash
# LangSplatV2
python scripts/train_language_field.py \
  --checkpoint .../params0.npz --poses .../keyframe_poses.json \
  --features_dir .../language_features --level s \
  --output_dir .../lang_field_sk64_l1 \
  --lang_mode langsplatv2 --codebook_size 64 --vq_layer_num 1 --topk 4

# LangSplat (legacy)
python scripts/train_language_field.py \
  ... --features_dir .../language_features_dim64 \
  --lang_mode langsplat --latent_dim 64
```

### Инференс (`query_language_field.py`)

| Флаг | Значения | Описание |
|------|----------|----------|
| `--lang_mode` | `auto` (default), `langsplatv2`, `langsplat` | `auto` — по полю `format` в `lang_field.pt` |
| `--ae_ckpt` | path | **Обязателен** для `langsplat`; для V2 не нужен |
| `--semantic_mask_mode` | `clip_langsplat`, `sam` | Бинаризация маски: LangSplatV2 heatmap vs SAM argmax |

---

## Инференс и отладка (новые скрипты)

| Скрипт | Назначение |
|--------|------------|
| `render_view_from_pose.py` | RGB из одной позы `traj.txt` |
| `render_query_from_pose.py` | RGB \| heatmap \| sem mask по текстовому запросу (один PNG) |
| `validate_lang_field_traj.py` | **Основная** traj mIoU для LangSplatV2 (pair mIoU, s/m/l, tqdm) |
| `compute_miou_p_traj.py` | mIoU_p: SAM+CLIP pseudo на rendered RGB (без lang field) |
| `lang_field_eval_utils.py` | Общие утилиты (poses, levels, mIoU writer, semsplatam metrics) |
| `lang_pipeline_utils.py` | Парсинг `mask_collector` / `lang_mode` |
| `run_nvs_validation.py` | NVS метрики (PSNR, SSIM, LPIPS) |
| `visualize_language_seg_maps.py` | Визуализация `*_s.npy` при обучении |

### Рендер вида

```bash
python scripts/render_view_from_pose.py \
  --checkpoint results/.../splatam/final/params.npz \
  --traj data/replica_sim_nvs/office0/traj.txt \
  --frame 42 --align_gs_train_frame --scene office0 \
  --out view_f42.png
```

![RGB-рендер office0](examples/view_f42.png)

### Рендер по текстовому запросу

Выход — **2×2**: Render | Heatmap (+шкала) / GT | Prediction. GT берётся из `results_habitat/semantic/` (нужен `--scene` или `--info_semantic`).

```bash
python scripts/render_query_from_pose.py \
  --checkpoint results/.../splatam/final/params.npz \
  --result_dir results/Replica/office0/ActiveOpenSem/run_0 \
  --traj data/replica_sim_nvs/office0/traj.txt \
  --frame 42 --align_gs_train_frame --scene office0 \
  --text "a sofa" --levels all --semantic_mask_thresh 0.50 \
  --out query_sofa_f42.png
```

![Запрос «a sofa», office0 — Render | Heatmap / GT | Prediction](examples/query_sofa_f01.png)

### Open-vocab запрос (полный пайплайн)

```bash
python scripts/query_language_field.py \
  --checkpoint .../params.npz \
  --lang_field .../lang_field_sk64_l1/lang_field.pt \
  --text "a sofa" \
  --lang_mode auto \
  --out results/query_sofa
```

---

## Валидация mIoU на traj

### Основной скрипт: `validate_lang_field_traj.py`

Для **CLIP + SAM + LangSplatV2** — pair mIoU по всей traj (кадр × класс), пирамида s/m/l, прогресс в терминале (tqdm; нужен `pip install tqdm`).

По умолчанию включён пресет **`val_results_sml_levels`** (s/m/l, thresh 0.55, `align_gs_train_frame`). Отключить: `--eval_preset none`.

```bash
python scripts/validate_lang_field_traj.py \
  --scene office0 \
  --result_dir results/Replica/office0/ActiveOpenSem/run_0 \
  --traj_txt data/replica_sim_nvs/office0/traj.txt \
  --device cuda:0 \
  --out_dir results/Replica/office0/ActiveOpenSem/run_0/lang_field_traj_eval
```

В stderr появится progress bar:

```
[lang-field-traj] pairs=274 frames=24 classes=23 ...
lang-field-traj:  45%|████▌     | 123/274 [02:15<02:45, 1.09pair/s]
```

| Флаг | Описание |
|------|----------|
| `--device` | GPU для рендера и CLIP (default `cuda:0`) |
| `--eval_preset` | `val_results_sml_levels` (default) или `none` |
| `--levels` | `all` или подмножество: `s`, `m`, `l`, `s,m` |
| `--semantic_mask_thresh` | Порог бинаризации heatmap (default 0.55 с пресетом) |
| `--semsplatam_metrics` | Дополнительно: frame mIoU (`miou_g`, `miou_p`, …) → `semantic_result.txt` |
| `--class_name_replace_hyphen_with " "` | Запросы без дефисов (`desk organizer`) |
| `--negative_from_other_classes` | Негативы = остальные классы сцены |

**Выход:** mIoU в stdout; `metrics.json`, `pairs.csv`, `miou_summary.txt`, `miou_per_class.csv`.

### Дополнительно: `compute_miou_p_traj.py`

Только **mIoU_p** — SAM+CLIP pseudo-labels на **отрендеренном RGB** SplaTAM (без language field). Не заменяет `validate_lang_field_traj.py`.

```bash
python scripts/compute_miou_p_traj.py \
  --scene office0 \
  --result_dir results/Replica/office0/ActiveOpenSem/run_0 \
  --traj_txt data/replica_sim_nvs/office0/traj.txt \
  --device cuda:0 \
  --out_dir results/Replica/office0/ActiveOpenSem/run_0/miou_p_traj_eval
```

Прогресс: `miou_p: 60%|██████ | 14/24 [01:02<00:41, 4.2frame/s]`.

### Что **не** использовать для traj LangSplatV2

| Скрипт | Почему |
|--------|--------|
| `eval_lang_field_segmentation.py` | устаревший, один level |
| `eval_clip_sam_miou.py` | 2D baseline без 3DGS |
| `eval_run_lang_fields.py` | несколько текстовых запросов, не полный бенч |

---

## Результаты

_(раздел будет дополнен)_

---

## VRAM и `--render_checkpoint`

### VRAM (ориентиры)

- **01 + SAM/CLIP:** SplaTAM ~8–14 GB + SAM ~2–4 GB + CLIP ~1–2 GB.  
- **LangSplat, этап 2 (AE):** обычно ~2–6 GB.  
- **LangSplat, этап 3:** растёт с числом гауссианов и `latent_dim`; при OOM `RENDER_CHECKPOINT=on` или `auto`.  
- **LangSplatV2, этап 3:** растёт с \(L \times K\); см. `--train_downscale` в `03_train_gaussian_lang_field_langsplatv2.sh`.

### Режим `--render_checkpoint`

| Значение | Поведение |
|----------|-----------|
| `off` | Один граф на все 3‑канальные проходы; быстрее, больше VRAM. |
| `on` | Gradient checkpoint на каждый проход; меньше VRAM. |
| `auto` | Для legacy — checkpoint при `latent_dim==64`; для V2 — эвристика в `LangSplatam`. |

### Запросы к языковому полю (режимы 3–4)

```bash
bash scripts/aov-gs/run_query_lang_fields_office0.sh \
  results/Replica/office0/ActiveOpenSem/run_0
```

`scripts/query_language_field.py` сам определяет формат чекпойнта (`legacy` vs LangSplatV2).

### Чеклисты

**LangSplat (legacy):** одинаковый `latent_dim` в этапах 2 и 3; папки `language_features_dim{N}`; `--level` согласован с `*_s.npy`.

**LangSplatV2:** этап 3 читает сырые **`language_features/`** (512‑D), не `language_features_dim*`; в `splatam/final/` допустимы `params0.npz` или `params.npz`.

---

## Прочие скрипты

- **`scripts/data/`** — загрузка и генерация Replica / MP3D.  
- **`scripts/evaluation/`** — оценки 3D / семантики / NVS.  
- **`scripts/aov-gs/`** — шаги 01–03, гриды, **`pipeline_gs_open_vocab.sh`**, **`_gpu_helpers.sh`**.  
- **`scripts/ablation/`** — бенчмарки ActiveSem / ActiveOpenSem, `run_scene_lang_pipeline.sh`.
- **`scripts/aov-gs/run_active_geom_all_scenes.sh`** — batch ActiveGeom на всех сценах Replica.
- **`scripts/aov-gs/run_active_open_sem_corrclip_in_planner_all_scenes.sh`** — batch ActiveOpenSem + CorrCLIP in planner.  
- **`compute_miou_p_traj.py`** — mIoU_p (SAM+CLIP pseudo на rendered RGB).  
- **`eval_lang_field_segmentation.py`** — старая оценка (один level); для traj используйте **`validate_lang_field_traj.py`**.  
- **`eval_clip_sam_miou.py`** — baseline 2D SAM+CLIP без 3DGS.

---

## Лицензии и заимствования

Код опирается на SplaTAM, ActiveSGM, Habitat, diff-gaussian-rasterization и др.; лицензии — в соответствующих каталогах `third_parties/` (после копирования).

### Цитирование (ActiveSGM)

```bibtex
@inproceedings{chen2025understanding,
  title={Understanding while Exploring: Semantics-driven Active Mapping},
  author={Chen, Liyan and Zhan, Huangying and Yin, Hairong and Xu, Yi and Mordohai, Philippos},
  booktitle={NeurIPS},
  year={2025}
}
```

При публикации своей работы на базе этого репозитория добавьте отдельную запись на **AOV-GS / open-vocabulary language field**, если требуется вашим вузом.
