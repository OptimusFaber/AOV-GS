# `third_parties/`

Зависимости Python-кода AOV-GS (Co-SLAM, SplaTAM, neural_slam_eval, channel rasterizers).

## Как получить (один раз)

**Рекомендуется** — клонировать репозиторий с submodules:

```bash
git clone --recursive https://github.com/OptimusFaber/AOV-GS
cd AOV-GS
```

Если уже склонировали без `--recursive`:

```bash
bash scripts/installation/setup_third_parties.sh
# или: git submodule update --init --recursive
```

Скрипт подтягивает репозитории из [`.gitmodules`](../.gitmodules) (~200 MB исходников).

## Что внутри

| Каталог | Назначение |
|---------|------------|
| `coslam` | Co-SLAM utils, colormap |
| `splatam` | SplaTAM datasets / slam helpers |
| `neural_slam_eval` | метрики реконструкции / NVS |
| `channel_rasterization` | dense semantic rasterizer (ActiveSem) |
| `sparse_channel_rasterization` | sparse semantic rasterizer |

## Чего здесь нет

**`habitat_sim/`** (~5 GB исходников + сборка) — **не** хранится в git.  
Рантайм Habitat ставится через conda: `bash scripts/installation/conda_env/build_sem.sh`.

## Почему не один большой коммит в git

- `habitat_sim` слишком тяжёлый для GitHub.
- Остальные пакеты (~200 MB) подключаются как **git submodules** — один `clone --recursive` или `setup_third_parties.sh`, без ручного копирования из других проектов.

Локальные артефакты сборки (`build/`, `*.so`, `__pycache__`) в git не попадают (см. `.gitignore`).
