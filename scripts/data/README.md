# Data preparation

Replica + Habitat, as in [ActiveSGM — Data preparation](https://github.com/lly00412/ActiveSGM/blob/main/README.md#data-preparation).

## Replica (main pipeline)

```bash
cd /path/to/AOV-GS

# 1. Download mesh
bash scripts/data/replica_download.sh data/replica_v1
bash scripts/data/replica_update.sh data/replica_v1

# 2. ReplicaSLAM trajectories
bash scripts/data/replica_slam_download.sh

# 3. RGB-D from Habitat (for training)
bash scripts/data/generate_replica_habitat.sh all
# or a single scene:
# bash scripts/data/generate_replica_habitat.sh office0

# 4. NVS trajectories (for novel-view evaluation)
bash scripts/data/generate_replica_nvs.sh all
# → data/replica_sim_nvs/<scene>/traj.txt
# → data/replica_sim_nvs/<scene>/results_habitat/
```

## Configs

- `configs/Replica/<scene>/habitat.py` — set **`scene_id`** to your `mesh_semantic.ply`
- `data/Replica/<scene>/` — `traj.txt`, `results/` or `results_habitat/`
- NVS GT: `data/replica_sim_nvs/<scene>/`

## Other

| Script | Purpose |
|--------|------------|
| `generate_mp3d_*.sh` | MP3D (not Replica) |
| `finetune_replica_oneformerv2.sh` | OneFormer for ActiveSem |
| `generate_replica_activegamer.sh` | ActiveGAMER trajectories |

## ScanNet (3 test scenes)

Raw scans: `/mnt/data/scannet/scans/` (`scene0000_00`, `scene0005_00`, `scene0010_00`).

```bash
cd AOV-GS

# 1. Mesh -> Habitat + configs + passive traj
bash scripts/data/prepare_scannet.sh all

# 2. Smoke test (GPU + habitat-sim)
python scripts/data/test_scannet_habitat.py --scene scene0000_00

# 3. NVS eval trajectories (optional, before mIoU eval)
bash scripts/data/generate_scannet_nvs.sh all

# 4. Active experiment
bash scripts/aov-gs/01_slam_exploration_scannet.sh scene0000_00 ActiveOpenSemGeom
```

Prepared layout: `data/ScanNet/habitat/{scene}/`, `data/ScanNet/{scene}/traj.txt`, `configs/ScanNet/{scene}/`.
