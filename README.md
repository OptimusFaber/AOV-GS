# AOV-GS — Active Open-Vocabulary 3D Gaussian Splatting

**Exploring while Grounding: Open-Vocabulary Active Mapping with 3D Gaussian Splatting**

Самостоятельный проект: активное исследование сцены (**SplaTAM**), сбор SAM+CLIP-признаков и **открытое языковое поле** (LangSplatV2 / legacy LangSplat).

Репозиторий: [github.com/OptimusFaber/AOV-GS](https://github.com/OptimusFaber/AOV-GS)

---

## Содержание

1. [Быстрый старт](#быстрый-старт)
2. [Документация](#документация)
3. [`third_parties/`](#third_parties)
4. [Основные команды](#основные-команды)
5. [GPU и CUDA](#gpu-и-cuda)
6. [NVS и валидация](#nvs-и-валидация)
7. [Структура репозитория](#структура-репозитория)
8. [Лицензии](#лицензии)

---

## Быстрый старт

```bash
git clone --recursive https://github.com/OptimusFaber/AOV-GS
cd AOV-GS

bash scripts/installation/quick_start.sh
```

Скрипт `quick_start.sh` по шагам:

1. `third_parties/` — git submodules / авто-клон (~200 MB)
2. conda-окружение `active-sgm` (~12 GB)
3. pip: open_clip, segment-anything, scikit-learn
4. чекпойнт SAM → `ckpts/sam_vit_b_01ec64.pth` (~400 MB)
5. данные Replica + Habitat + NVS

**Место на диске:** планируйте **≥ 30 GB** свободного (conda ~12 GB, Replica и производные данные ~7–17 GB, остальное ~1 GB).

Опции: `--skip-env`, `--skip-data`, `--skip-sam`, `--yes` (без подтверждения). Подробности: `bash scripts/installation/quick_start.sh --help`.

Если клонировали **без** `--recursive`:

```bash
bash scripts/installation/setup_third_parties.sh
```

После установки:

```bash
conda activate active-sgm
# дальше — scripts/aov-gs/README.md
```

---

## Документация

| Тема | Файл |
|------|------|
| Пайплайн 01–03, batch, **GPU** | [scripts/aov-gs/README.md](scripts/aov-gs/README.md) |
| Conda, HPC, quick start | [scripts/installation/README.md](scripts/installation/README.md) |
| Данные Replica / NVS | [scripts/data/README.md](scripts/data/README.md) |
| `third_parties/` | [third_parties/README.md](third_parties/README.md) |
| Docker | [docker/README.md](docker/README.md) |

---

## `third_parties/`

Зависимости (~200 MB): Co-SLAM, SplaTAM, neural_slam_eval, channel rasterizers.

**Не нужно** вручную копировать из других репозиториев:

```bash
git clone --recursive https://github.com/OptimusFaber/AOV-GS
# или после обычного clone:
bash scripts/installation/setup_third_parties.sh
```

**`habitat_sim`** (~5 GB) в git не хранится — ставится через conda (`build_sem.sh`).

Проверка:

```bash
test -f third_parties/coslam/utils.py && \
test -f third_parties/splatam/utils/slam_external.py && echo OK
```

---

## Основные команды

См. [scripts/aov-gs/README.md](scripts/aov-gs/README.md).

```bash
# Полный open-vocab
bash scripts/aov-gs/pipeline_gs_open_vocab.sh office0

# Пошагово
bash scripts/aov-gs/01_slam_exploration.sh office0 ActiveOpenSem 0 0 0
bash scripts/aov-gs/02_validate_features_langsplatv2.sh results/Replica/office0/ActiveOpenSem/run_0
bash scripts/aov-gs/03_train_gaussian_lang_field_langsplatv2.sh \
  results/Replica/office0/ActiveOpenSem/run_0 64 s 30000 cuda:0 1 4 auto 1.0
```

Результаты: `results/Replica/<scene>/<EXP>/run_N/`.

---

## GPU и CUDA

Краткая шпаргалка. Подробно: [scripts/aov-gs/README.md § GPU](scripts/aov-gs/README.md#gpu-и-cuda).

| Что | Куда писать | Default |
|-----|-------------|---------|
| Видимые GPU | `GPU` / `CUDA_VISIBLE_DEVICES` | `01_*.sh`: `GPU=0,1` |
| SplaTAM | `primary_device` в `replica_splatam_s.py` | `cuda:0` |
| SAM + CLIP | `sam_clip.device` в `ActiveOpenSem_base.py` | **`cuda:0`** |
| Lang field train | `DEVICE` в `03_*.sh` | `cuda:0` |
| Python-скрипты | `--device` | `cuda:0` |

**Две GPU** (SLAM на 0, SAM на 1): в конфиге `sam_clip.device = "cuda:1"`, запуск `GPU=0,1 bash scripts/aov-gs/01_slam_exploration.sh ...`

**Lang field на GPU 1:** `cuda:1` в аргументе `03_*.sh` (auto-remap через `_gpu_helpers.sh`).

Переопределение SAM без правки конфига: `SAM_CLIP_DEVICE=cuda:1`.

---

## NVS и валидация

```bash
python scripts/run_nvs_validation.py \
  --cfg configs/Replica/office0/ActiveOpenSem.py \
  --result_dir results/Replica/office0/ActiveOpenSem/run_0 \
  --stage eval_exploration_stage_1

python scripts/validate_lang_field_traj.py \
  --scene office0 \
  --result_dir results/Replica/office0/ActiveOpenSem/run_0 \
  --traj_txt data/replica_sim_nvs/office0/traj.txt \
  --device cuda:0
```

---

## Структура репозитория

| Путь | Назначение |
|------|------------|
| `src/main/activesgm.py` | SLAM + SAM/CLIP |
| `scripts/aov-gs/` | пайплайн 01–03 |
| `scripts/installation/quick_start.sh` | установка «в один заход» |
| `third_parties/` | vendored deps (submodules) |
| `configs/Replica/<scene>/` | конфиги сцен |

---

## Лицензии

MIT — см. [LICENSE](LICENSE).  
Сторонние компоненты (SplaTAM, Co-SLAM, Habitat и др.) — лицензии в соответствующих репозиториях / `third_parties/`.
