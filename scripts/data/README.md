# Подготовка данных

Replica + Habitat, как в [ActiveSGM — Data preparation](https://github.com/lly00412/ActiveSGM/blob/main/README.md#data-preparation).

## Replica (основной пайплайн)

```bash
cd /path/to/AOV-GS

# 1. Скачать mesh
bash scripts/data/replica_download.sh data/replica_v1
bash scripts/data/replica_update.sh data/replica_v1

# 2. ReplicaSLAM траектории
bash scripts/data/replica_slam_download.sh

# 3. RGB-D из Habitat (для обучения)
bash scripts/data/generate_replica_habitat.sh all
# или одна сцена:
# bash scripts/data/generate_replica_habitat.sh office0

# 4. NVS-траектории (для оценки novel views)
bash scripts/data/generate_replica_nvs.sh all
# → data/replica_sim_nvs/<scene>/traj.txt
# → data/replica_sim_nvs/<scene>/results_habitat/
```

## Конфиги

- `configs/Replica/<scene>/habitat.py` — **`scene_id`** на ваш `mesh_semantic.ply`
- `data/Replica/<scene>/` — `traj.txt`, `results/` или `results_habitat/`
- NVS GT: `data/replica_sim_nvs/<scene>/`

## Прочее

| Скрипт | Назначение |
|--------|------------|
| `generate_mp3d_*.sh` | MP3D (не Replica) |
| `finetune_replica_oneformerv2.sh` | OneFormer для ActiveSem |
| `generate_replica_activegamer.sh` | ActiveGAMER trajectories |
