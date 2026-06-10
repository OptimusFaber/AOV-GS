# AOV-GS — Active Open-Vocabulary 3D Gaussian Splatting

Ветка на базе [ActiveSGM](https://github.com/lly00412/ActiveSGM): активное исследование (**SplaTAM**) и **открытое языковое поле** (SAM+CLIP → LangSplatV2 или legacy LangSplat).

---

## Содержание

1. [Быстрый старт](#быстрый-старт)
2. [Документация по разделам](#документация-по-разделам)
3. [`third_parties/` (обязательно)](#third_parties-обязательно)
4. [Основные команды](#основные-команды)
5. [GPU и CUDA](#gpu-и-cuda)
6. [NVS и валидация](#nvs-и-валидация)
7. [Структура репозитория](#структура-репозитория)
8. [Лицензии](#лицензии)

---

## Быстрый старт

```bash
git clone <repo-url> AOV-GS && cd AOV-GS

# 1. third_parties (не в git) — см. ниже
# 2. окружение
bash scripts/installation/conda_env/build_sem.sh
conda activate active-sgm
pip install open_clip_torch scikit-learn tqdm
pip install git+https://github.com/facebookresearch/segment-anything.git

# 3. данные Replica — см. scripts/data/README.md
bash scripts/data/replica_download.sh data/replica_v1
bash scripts/data/replica_update.sh data/replica_v1
bash scripts/data/replica_slam_download.sh
bash scripts/data/generate_replica_habitat.sh all
bash scripts/data/generate_replica_nvs.sh all

# 4. SAM checkpoint → ckpts/sam_vit_b_01ec64.pth

# 5. полный пайплайн
bash scripts/aov-gs/pipeline_gs_open_vocab.sh office0
```

---

## Документация по разделам

| Тема | Файл |
|------|------|
| **Пайплайн 01–03**, batch-скрипты, **GPU подробно** | [scripts/aov-gs/README.md](scripts/aov-gs/README.md) |
| **Conda, HPC, Docker** | [scripts/installation/README.md](scripts/installation/README.md) |
| **Данные Replica / NVS** | [scripts/data/README.md](scripts/data/README.md) |
| **Docker-образ** | [docker/README.md](docker/README.md) |

---

## `third_parties/` (обязательно)

Каталог в **`.gitignore`** — после `git clone` его нет. Без него падают импорты `third_parties.coslam`, `third_parties.splatam` и NVS.

Нужны: `coslam`, `splatam`, `neural_slam_eval`, `channel_rasterization`, `sparse_channel_rasterization`.

```bash
# Вариант A: ActiveSGM с --recursive → скопировать third_parties/
# Вариант B: соседний клон с уже собранными deps
rsync -a ../AOV-GS-V2/third_parties/ third_parties/
# Вариант C: вручную по .gitmodules
```

Проверка:

```bash
test -f third_parties/coslam/utils.py && \
test -f third_parties/splatam/utils/slam_external.py && echo OK
```

---

## Основные команды

Подробные аргументы — в [scripts/aov-gs/README.md](scripts/aov-gs/README.md).

### Полный open-vocab (рекомендуется)

```bash
bash scripts/aov-gs/pipeline_gs_open_vocab.sh office0
# CorrCLIP + фиксированный run:
# bash scripts/aov-gs/pipeline_gs_open_vocab.sh office0 0 0 0 corrclip langsplatv2 run_corrclip
```

| `MASK_COLLECTOR` | Описание |
|------------------|----------|
| `sam` | обычный SAM+CLIP |
| `corrclip` | merge масок + inter-class suppression |

| `LANG_MODE` | Описание |
|-------------|----------|
| `langsplatv2` | codebook, без AE (default) |
| `langsplat` | legacy + автоэнкодер |

### Пошагово

```bash
# 1 — SLAM + SAM/CLIP
bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem 0 0 0

# 2 — LangSplatV2: проверка фич
bash scripts/aov-gs/02_validate_features_langsplatv2.sh results/Replica/office0/ActiveOpenSem/run_0

# 3 — языковое поле
bash scripts/aov-gs/03_train_gaussian_lang_field_langsplatv2.sh \
  results/Replica/office0/ActiveOpenSem/run_0 64 s 30000 cuda:0 1 4 auto 1.0
```

### Другие режимы

| Скрипт | Режим |
|--------|-------|
| `pipeline_gs_no_segmenter.sh` | только геометрия (`ActiveGS`) |
| `pipeline_gs_oneformer.sh` | OneFormer closed-set (`ActiveSem`) |

Результаты: `results/Replica/<scene>/<EXP>/run_N/`.

---

## GPU и CUDA

Краткая шпаргалка. **Полная таблица и примеры** — [scripts/aov-gs/README.md § GPU](scripts/aov-gs/README.md#gpu-и-cuda).

| Что | Куда писать | Default |
|-----|-------------|---------|
| Видимые GPU в shell | `GPU` или `CUDA_VISIBLE_DEVICES` | `01_*.sh`: `GPU=0,1` |
| SplaTAM | `primary_device` в `configs/Replica/replica_splatam_s.py` | `cuda:0` |
| SAM + CLIP | `sam_clip.device` в `configs/Replica/<scene>/ActiveOpenSem_base.py` | `cuda:1` |
| OneFormer | `semantic_device` в `ActiveSem.py` | `cuda:0` |
| Обучение lang field | 5-й аргумент `DEVICE` в `03_*.sh` | `cuda:0` |
| Python-скрипты | `--device` | `cuda:0` |

**Две карты** (SLAM на 0, SAM на 1) — конфиг уже настроен для `office0`; запуск:

```bash
GPU=0,1 bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem
```

**Lang field на физической GPU 1** — rasterizer требует logical `cuda:0`:

```bash
bash scripts/aov-gs/03_train_gaussian_lang_field_langsplatv2.sh \
  results/.../run_0 64 s 30000 cuda:1 1 4 auto 1.0
# скрипт сам выставит CUDA_VISIBLE_DEVICES=1
```

**Одна GPU:** `GPU=0` и в конфиге `sam_clip.device = "cuda:0"`.

---

## NVS и валидация

**Novel view synthesis** (PSNR / SSIM / LPIPS на `data/replica_sim_nvs/`):

```bash
python scripts/run_nvs_validation.py \
  --cfg configs/Replica/office0/ActiveOpenSem.py \
  --result_dir results/Replica/office0/ActiveOpenSem/run_0 \
  --stage eval_exploration_stage_1
```

**mIoU на traj** (LangSplatV2):

```bash
python scripts/validate_lang_field_traj.py \
  --scene office0 \
  --result_dir results/Replica/office0/ActiveOpenSem/run_0 \
  --traj_txt data/replica_sim_nvs/office0/traj.txt \
  --device cuda:0
```

**Текстовый запрос:**

```bash
python scripts/query_language_field.py \
  --checkpoint results/.../splatam/final/params.npz \
  --lang_field results/.../lang_field_sk64_l1/lang_field.pt \
  --text "a sofa" --lang_mode auto --out results/query_sofa
```

Другие скрипты: `render_view_from_pose.py`, `render_query_from_pose.py`, `compute_miou_p_traj.py` — флаги в `--help`.

---

## Структура репозитория

| Путь | Назначение |
|------|------------|
| `src/main/activesgm.py` | активное исследование + SAM/CLIP |
| `src/slam/` | SplaTAM, LangSplatam |
| `scripts/aov-gs/` | пайплайн 01–03 |
| `scripts/train_language_field.py` | обучение языкового поля |
| `configs/Replica/<scene>/` | конфиги сцен |
| `data/replica_sim_nvs/` | NVS GT |
| `src/utils/display_utils.py` | headless (`ENABLE_VIS=0`) |

---

## Лицензии

Код использует SplaTAM, ActiveSGM, Habitat и др.; лицензии — в `third_parties/`.

```bibtex
@inproceedings{chen2025understanding,
  title={Understanding while Exploring: Semantics-driven Active Mapping},
  author={Chen, Liyan and Zhan, Huangying and Yin, Hairong and Xu, Yi and Mordohai, Philippos},
  booktitle={NeurIPS},
  year={2025}
}
```
