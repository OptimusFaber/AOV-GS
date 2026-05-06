# AOV-GS — Active Open-Vocabulary 3D Gaussian Splatting

**Краткое имя для репозитория:** `aov-gs` или `OAVGS` (Open / Active Vocabulary + Gaussian Splatting).

Ветка на базе [ActiveSGM](https://github.com/lly00412/ActiveSGM): активное исследование сцены (**SplaTAM**, 3D Gaussian Splatting) и **открытое языковое поле** (SAM + CLIP → автоэнкодер → признаки на гауссианах), в духе LangSplat. Основной конфиг — **ActiveOpenSem** без OneFormer.

---

## Содержание

1. [Клонирование и `third_parties`](#клонирование-и-third_parties)
2. [Окружение](#окружение)
3. [Данные Replica + Habitat](#данные-replica--habitat)
4. [Чекпойнты SAM / CLIP](#чекпойнты-sam--clip)
5. [Настройка путей в конфигах](#настройка-путей-в-конфигах)
6. [Пайплайн: 3 этапа](#пайплайн-3-этапа)
7. [Публикация на GitHub](#публикация-на-github)

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

### Open-vocabulary (этап 1–3): SAM + CLIP

После `build_sem.sh` установите пакеты для извлечения масок и CLIP:

```bash
pip install open_clip_torch
pip install git+https://github.com/facebookresearch/segment-anything.git
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
3. При **одной GPU** в `configs/Replica/<scene>/ActiveOpenSem.py` поставьте `sam_clip.device = "cuda:0"` (иначе падение при отсутствии `cuda:1`).

---

## Пайплайн: 3 этапа

**Суть:** (1) активное картирование SplaTAM + параллельно SAM/CLIP → `language_features/`; (2) автоэнкодер CLIP 512→D; (3) обучение языкового поля на замороженных гауссианах.

| Этап | Выход |
|------|--------|
| 1 | `splatam/final/params*.npz`, `keyframe_poses.json`, `language_features/*_{s,f}.npy` |
| 2 | `language_features_dim{N}/`, `ckpt/<scene>/best_ckpt.pth` |
| 3 | `lang_field_<level>{N}/lang_field.pt` |

### Этап 1

```bash
cd /path/to/aov-gs
bash scripts/activesgm/01_slam_exploration.sh office0
# отладка (keyframes на диск), без окон:  office0 ActiveOpenSem 0 0 1
# окна OpenCV (RGB-D) + отладка:         office0 ActiveOpenSem 0 1 1
```

Аргументы: `[SCENE] [EXP] [SEED] [ENABLE_VIS] [DEBUG]`. Четвёртый (`ENABLE_VIS`) должен быть **1**, иначе в логе будет `Visualize : 0` и окон не будет — это не баг, а настройка (`--enable_vis` → `visualizer.vis_rgbd`).

### Этап 2

```bash
bash scripts/activesgm/02_train_clip_autoencoder.sh \
  results/Replica/office0/ActiveOpenSem/run_0 \
  64 \
  100 \
  cuda:0
```

### Этап 3

```bash
bash scripts/activesgm/03_train_gaussian_lang_field.sh \
  results/Replica/office0/ActiveOpenSem/run_0 \
  64 \
  s \
  30000 \
  cuda:0 \
  auto
```

Прямой вызов Python и **сетка экспериментов** — см. сжатое описание ниже.

### VRAM (ориентиры)

- **Этап 1:** SplaTAM ~8–14 GB + SAM ~2–4 GB + CLIP ~1–2 GB; удобны 2 GPU или одна 16+ GB.  
- **Этап 2:** обычно легче (~2–6 GB).  
- **Этап 3:** растёт с числом гауссианов, D и числом проходов растра; для D=64 используйте `--render_checkpoint on` или `auto` при OOM.

### Сетка `run_lang_field_full_grid.sh`

```bash
bash scripts/activesgm/run_lang_field_full_grid.sh \
  results/Replica/office0/ActiveOpenSem/run_0
```

Перед запуском должны существовать `language_features_dim3` … `language_features_dim64`. Логи: `lang_field_grid_logs/`, кривые loss: `lang_field_{L}{D}/loss_{D}{L}.txt`.

### Режим `--render_checkpoint` (D > 3)

| Значение | Поведение |
|----------|-----------|
| `off` | Один граф на все проходы; быстрее, больше VRAM. |
| `on` | Checkpoint на каждый 3-канальный проход; меньше VRAM. |
| `auto` | Checkpoint при `latent_dim == 64`, иначе как `off`. |

### Запросы к полю

```bash
bash scripts/activesgm/run_query_lang_fields_office0.sh \
  results/Replica/office0/ActiveOpenSem/run_0
```

Также: `scripts/query_language_field.py`, `scripts/demo_sam_clip_text_query.py`.

### Чеклист согласованности

1. `latent_dim` на этапах 2 и 3 совпадает с именем `language_features_dim{N}`.  
2. Чекпойнт карты: в `final/` может быть `params0.npz` или `params.npz` — скрипты учитывают оба.  
3. `--level` (s/m/l) согласован с данными в `_s.npy`.

---

## Публикация на GitHub

Репозиторий изначально может быть без `.git`. Пример:

```bash
cd /path/to/aov-gs
git init
git branch -M main
git add .
git commit -m "Initial commit: AOV-GS open-vocabulary Gaussian splatting pipeline"
```

Создайте пустой репозиторий на GitHub (имя, например, **`aov-gs`**; в описании можно указать **OAVGS**):

```bash
git remote add origin https://github.com/<USER>/aov-gs.git
git push -u origin main
```

Из-за `.gitignore` на `third_parties/` в самом GitHub-репозитории **нет** тяжёлых зависимостей — только **`.gitmodules`** как шпаргалка по URL. Новым пользователям: см. раздел [Клонирование и `third_parties`](#клонирование-и-third_parties).

---

## Прочие скрипты

- **`scripts/data/`** — загрузка и генерация Replica / MP3D.  
- **`scripts/evaluation/`** — оценки 3D / семантики / NVS в духе ActiveSGM.  
- **`scripts/activesgm/`** — минимальный пайплайн AOV-GS (этап 1–3), грид/запросы для языкового поля.

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
