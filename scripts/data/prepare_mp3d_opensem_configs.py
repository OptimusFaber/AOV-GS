#!/usr/bin/env python3
"""Generate ActiveOpenSem configs for MP3D scenes (SplaTAM + SAM/CLIP, no OneFormer)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG_ROOT = ROOT / "configs" / "MP3D"

SCENES = {
    "GdvgFV5R1Z5": {
        "bbox_bound": [[-6.8, 0.6], [-3.8, 3.6], [-0.1, 3.9]],
        "start_c2w": [
            [1, 0, 0, -4],
            [0, 0, -1, 1],
            [0, 1, 0, 1.5],
            [0, 0, 0, 1],
        ],
        "surface_dist_thre": 0.3,
    },
    "gZ6f7yhEvPG": {
        "bbox_bound": [[-4.2, 3.6], [-2.8, 3.0], [-0.1, 5.3]],
        "start_c2w": [
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 2],
            [0, 0, 0, 1],
        ],
        "surface_dist_thre": 0.3,
    },
    "HxpKQynjfin": {
        "bbox_bound": [[-1.1, 4.8], [-8.3, 1.3], [-0.4, 2.8]],
        "start_c2w": [
            [1, 0, 0, 2],
            [0, 0, -1, -3.5],
            [0, 1, 0, 1.5],
            [0, 0, 0, 1],
        ],
        "surface_dist_thre": 0.3,
    },
    "pLe4wQe7qrG": {
        "bbox_bound": [[-2.4, 9.2], [-3.8, 3.9], [-0.2, 10.5]],
        "start_c2w": [
            [1, 0, 0, 5],
            [0, 0, -1, 0],
            [0, 1, 0, 1.75],
            [0, 0, 0, 1],
        ],
        "surface_dist_thre": 0.3,
    },
    "YmJkqBEsHnH": {
        "bbox_bound": [[-16.3, 4.2], [-5.3, 1.2], [-1.0, 5.6]],
        "start_c2w": [
            [0, 0, 1, 0],
            [1, 0, 0, -2],
            [0, 1, 0, 1.75],
            [0, 0, 0, 1],
        ],
        "surface_dist_thre": 0.3,
    },
}


def _fmt_matrix(rows, indent="            "):
    lines = []
    for row in rows:
        inner = ", ".join(str(x) for x in row)
        lines.append(f"{indent}[{inner}],")
    return "\n".join(lines)


def _fmt_bbox(bbox):
    parts = []
    for lo, hi in bbox:
        parts.append(f"[{lo}, {hi}]")
    return f"[{', '.join(parts)}]"


def write_base(scene: str, params: dict) -> None:
    bbox = _fmt_bbox(params["bbox_bound"])
    start_c2w = _fmt_matrix(params["start_c2w"])
    surface = params["surface_dist_thre"]
    text = f'''"""
ActiveOpenSem_base ({scene}) — MP3D Habitat scene, SplaTAM + SAM/CLIP (no OneFormer).

Prerequisites:
  - data/MP3D/ (habitat meshes)
  - data/mp3d_sim_nvs_v2/{scene}/ (optional NVS eval frames)

Run:
  bash scripts/aov-gs/01_slam_exploration_mp3d.sh {scene} ActiveOpenSemGeom
  bash scripts/aov-gs/01_slam_exploration_mp3d.sh {scene} ActiveOpenSem
"""

import os

from mmengine.config import read_base

with read_base():
    from ...default import *

general = dict(
    dataset="MP3D",
    scene="{scene}",
    num_iter=5000,
    device="cuda",
)

dirs = dict(
    data_dir="data/",
    result_dir="results/",
    cfg_dir=os.path.join("configs", general["dataset"], general["scene"]),
)

sim = dict(method="habitat_v2")
if sim["method"] == "habitat_v2":
    sim.update(habitat_cfg=os.path.join(dirs["cfg_dir"], "habitat.py"))

slam = dict(method="splatam")
if slam["method"] == "splatam":
    slam.update(
        room_cfg=f"{{dirs['cfg_dir']}}/../mp3d_splatam_s.py",
        enable_active_planning=True,
        dataset_eval_basedir="data/mp3d_sim_nvs_v2",
        eval_during_training=False,
        eval_during_training_freq=200,
        eval_during_training_max_frames=None,
        bbox_bound={bbox},
        bbox_voxel_size=0.05,
        surface_dist_thre={surface},
        find_free_indices_bs=1000,
        refine_map_iter=60,
        use_global_keyframe=True,
        global_keyframe=dict(
            completeness_thre=0.1,
            color_thre=34,
            depth_thre=0.01,
            quality_method="relative",
            quality_freq=100,
            quality_perc_thre=30,
        ),
        save_keyframes=True,
        save_keyframe_poses=True,
        save_clip_features=True,
        initialize_from_live_frame=True,
        start_c2w=[
{start_c2w}
        ],
        override=dict(
            map_every=5,
            report_global_progress_every=5,
            save_checkpoints=False,
            tracking=dict(use_gt_poses=True),
        ),
    )

sam_clip = dict(
    sam_ckpt_path="ckpts/sam_vit_b_01ec64.pth",
    clip_model="ViT-B-16",
    clip_pretrained="laion2b_s34b_b88k",
    device="cuda:0",
    queue_size=8,
    clip_batch_size=32,
    max_masks_per_frame=120,
    corrclip_mask_merge=True,
    corrclip_merge_sim_thresh=0.86,
    corrclip_merge_dist_px=80.0,
    corrclip_interclass_suppress_alpha=0.15,
    corrclip_interclass_sim_thresh=0.78,
    corrclip_interclass_sigma_px=120.0,
)

planner = dict(
    method="active_gs",
    max_exploration_steps=5000,
    post_refine_steps=200,
    max_refinement_steps=200,
    num_exploration_stage=2,
    gs_z_levels=[[35], [20, 35, 50]],
    num_dir_samples=[5, 15],
    xy_sampling_step=[1.0, 0.5],
    trans_step_size=0.1,
    rot_step_size=10,
    surface_dist_thre=slam["surface_dist_thre"],
    explore_thre=0.005,
    color_ig_thre=34,
    depth_ig_thre=0.01,
    post_refinement_eval_freq=100,
    up_dir=[0, 0, 1],
    use_traj_pose=False,
    SLAMData_dir=os.path.join(
        dirs["data_dir"], "MP3D", "v1/scans", general["scene"],
    ),
    local_planner_method="RRTNaruto",
)

if planner["local_planner_method"] == "RRTNaruto":
    planner.update(
        rrt_step_size=planner["trans_step_size"] / slam["bbox_voxel_size"],
        rrt_step_amplifier=10,
        rrt_maxz=100,
        rrt_max_iter=None,
        rrt_z_levels=None,
        enable_eval=False,
        enable_direct_line=True,
    )

visualizer = dict(
    method="active_gs",
    vis_rgbd=True,
    vis_rgbd_max_depth=10,
)
'''
    (CFG_ROOT / scene / "ActiveOpenSem_base.py").write_text(text, encoding="utf-8")


def write_geom(scene: str) -> None:
    text = f'''"""ActiveOpenSemGeom — SplaTAM + active_gs (geometry-only) on MP3D.

Results: results/MP3D/{scene}/ActiveGeom/run_N/

Run:
    bash scripts/aov-gs/01_slam_exploration_mp3d.sh {scene} ActiveOpenSemGeom
    bash scripts/activesgm/run_mp3d.sh {scene} 1 ActiveOpenSemGeom
"""

import os

from mmengine.config import read_base

with read_base():
    from .ActiveOpenSem_base import *

dirs["result_dir"] = os.path.join("results", general["dataset"], general["scene"], "ActiveGeom")

# Geometry baseline: ActiveGSPlanner (not hybrid v3).
planner["method"] = "active_gs"

visualizer["vis_rgbd"] = False
slam["override"]["viz"] = dict(enter_interactive_post_online=False, visualize_cams=False)
'''
    (CFG_ROOT / scene / "ActiveOpenSemGeom.py").write_text(text, encoding="utf-8")


def write_opensem(scene: str) -> None:
    text = '''"""ActiveOpenSem — SAM+CLIP + hybrid v3 planner on MP3D."""

import os

from mmengine.config import read_base

with read_base():
    from .ActiveOpenSem_base import *

dirs["result_dir"] = os.path.join(
    "results", general["dataset"], general["scene"], "ActiveOpenSem",
)
planner["method"] = "active_gs_hybrid_v3"
planner["seman_thre"] = 0.7
planner["max_revisit_count"] = 3
planner["semantic_exploration"] = dict(
    enabled_stages=[0],
    novelty_aggregation="mean",
    min_bank_masks=1,
    max_masks_per_candidate=40,
    max_semantic_candidates=8,
    log_semantic_scores=True,
    encode_keyframes_if_missing=True,
    post_refinement_semantic=True,
    post_refinement_max_keyframes=None,
)
visualizer["vis_rgbd"] = False
slam["override"]["viz"] = dict(enter_interactive_post_online=False, visualize_cams=False)
'''
    (CFG_ROOT / scene / "ActiveOpenSem.py").write_text(text, encoding="utf-8")


def write_passive(scene: str) -> None:
    text = f'''"""ActiveOpenSemPassive — fixed NVS trajectory (no active planning) on MP3D."""

import os

from mmengine.config import read_base

with read_base():
    from .ActiveOpenSem_base import *

dirs["result_dir"] = os.path.join(
    "results", general["dataset"], general["scene"], "Passive",
)

slam["enable_active_planning"] = False

planner["method"] = "predefined_traj"
planner["use_traj_pose"] = True
planner["SLAMData_dir"] = os.path.join(
    dirs["data_dir"],
    "mp3d_sim_nvs_v2",
    general["scene"],
)

visualizer["vis_rgbd"] = False
slam["override"]["viz"] = dict(
    enter_interactive_post_online=False,
    visualize_cams=False,
)
'''
    (CFG_ROOT / scene / "ActiveOpenSemPassive.py").write_text(text, encoding="utf-8")


def main() -> None:
    for scene, params in SCENES.items():
        scene_dir = CFG_ROOT / scene
        if not (scene_dir / "habitat.py").exists():
            raise FileNotFoundError(f"Missing {scene_dir}/habitat.py")
        write_base(scene, params)
        write_geom(scene)
        write_opensem(scene)
        write_passive(scene)
        print(f"Wrote ActiveOpenSem configs for {scene}")


if __name__ == "__main__":
    main()
