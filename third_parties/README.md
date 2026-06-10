# `third_parties/`

Зависимости Python-кода AOV-GS. **В git не хранятся** (как в [ActiveSGM](https://github.com/lly00412/ActiveSGM)) — подтягиваются после клона.

## Установка

```bash
git clone --recursive https://github.com/OptimusFaber/AOV-GS
cd AOV-GS
```

Если клонировали без `--recursive`:

```bash
bash scripts/installation/setup_third_parties.sh
```

Или вручную: `git submodule update --init --recursive` (см. [`.gitmodules`](../.gitmodules)).

## Каталоги

| Путь | Назначение | Нужен для |
|------|------------|-----------|
| `coslam/` | Co-SLAM utils | визуализация, RRT |
| `splatam/` | SplaTAM helpers | SLAM, NVS |
| `neural_slam_eval/` | mesh / recon metrics | eval_recon |
| `channel_rasterization/` | dense semantic rasterizer | **ActiveSem** (опционально) |
| `sparse_channel_rasterization/` | sparse semantic rasterizer | **ActiveSem** (опционально) |

Для **`ActiveOpenSem`** + языкового поля достаточно `coslam`, `splatam`, `neural_slam_eval` (~150 MB).

## `habitat_sim/`

Не входит в `third_parties/` git. Habitat ставится через conda: `bash scripts/installation/conda_env/build_sem.sh`.
