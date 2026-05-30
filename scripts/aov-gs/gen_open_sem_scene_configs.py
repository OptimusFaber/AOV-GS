#!/usr/bin/env python3
"""Generate ActiveOpenSem_base / Geom / ActiveOpenSem configs for Replica scenes."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs" / "Replica"

SCENES: dict[str, dict] = {
    "office1": {
        "bbox": "[[-1.9, 3.1], [-1.7, 2.7], [-1.2, 1.9]]",
        "xy": "[0.5, 0.3]",
    },
    "office2": {
        "bbox": "[[-3.6, 3.2], [-3.0, 5.4], [-1.3, 1.6]]",
        "xy": "[1.0, 0.5]",
    },
    "office3": {
        "bbox": "[[-5.2, 3.6], [-6.1, 3.4], [-1.3, 2.0]]",
        "xy": "[1.0, 0.5]",
    },
    "office4": {
        "bbox": "[[-1.3, 5.4], [-2.4, 4.2], [-1.3, 1.7]]",
        "xy": "[1.0, 0.5]",
    },
    "room0": {
        "bbox": "[[-1.0, 7.0], [-1.2, 3.6], [-1.6, 1.4]]",
        "xy": "[1.0, 0.5]",
    },
    "room1": {
        "bbox": "[[-5.5, 1.3], [-3.1, 2.8], [-1.5, 1.4]]",
        "xy": "[1.0, 0.5]",
    },
    "room2": {
        "bbox": "[[-0.9, 6.1], [-3.3, 1.8], [-3.0, 0.8]]",
        "xy": "[1.0, 0.5]",
    },
}

ACTIVE_OPEN_SEM_BASE = '''\
"""
ActiveOpenSem_base ({scene}) — SplaTAM + active_gs + SAM/CLIP feature extraction.
Shared base for ActiveOpenSem, ActiveGeom, and passive variants.
"""

import os

from mmengine.config import read_base

with read_base():
    from ...default import *

general = dict(
    dataset="Replica",
    scene="{scene}",
    num_iter=2000,
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
        room_cfg=f"{{dirs['cfg_dir']}}/../replica_splatam_s.py",
        enable_active_planning=True,
        dataset_eval_basedir="data/replica_sim_nvs",
        eval_during_training=True,
        eval_during_training_freq=200,
        eval_during_training_max_frames=None,
        bbox_bound={bbox},
        bbox_voxel_size=0.05,
        surface_dist_thre=0.5,
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
        override=dict(
            map_every=5,
            report_global_progress_every=5,
            tracking=dict(use_gt_poses=True),
        ),
    )

sam_clip = dict(
    sam_ckpt_path="ckpts/sam_vit_b_01ec64.pth",
    clip_model="ViT-B-16",
    clip_pretrained="laion2b_s34b_b88k",
    device="cuda:1",
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
    max_exploration_steps=1500,
    post_refine_steps=200,
    max_refinement_steps=200,
    num_exploration_stage=2,
    gs_z_levels=[[35], [20, 50]],
    num_dir_samples=[5, 15],
    xy_sampling_step={xy},
    trans_step_size=0.1,
    rot_step_size=10,
    surface_dist_thre=slam["surface_dist_thre"],
    explore_thre=0.005,
    color_ig_thre=34,
    depth_ig_thre=0.01,
    post_refinement_eval_freq=100,
    up_dir=[0, 0, 1],
    use_traj_pose=True,
    SLAMData_dir=os.path.join(dirs["data_dir"], "Replica", general["scene"]),
    local_planner_method="RRTNaruto",
)

if planner["local_planner_method"] == "RRTNaruto":
    planner.update(
        rrt_step_size=planner["trans_step_size"] / slam["bbox_voxel_size"],
        rrt_step_amplifier=10,
        rrt_maxz=100,
        rrt_max_iter=50000,
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

GEOM = '''\
"""ActiveOpenSemGeom — geometry-only active_gs. Results: .../ActiveGeom/run_N/"""

import os

from mmengine.config import read_base

with read_base():
    from .ActiveOpenSem_base import *

dirs["result_dir"] = os.path.join("results", general["dataset"], general["scene"], "ActiveGeom")
visualizer["vis_rgbd"] = False
slam["override"]["viz"] = dict(enter_interactive_post_online=False, visualize_cams=False)
'''

ACTIVE_OPEN_SEM = '''\
"""ActiveOpenSem — SAM+CLIP semantic exploration + hybrid v3 planner. Results: .../ActiveOpenSem/run_N/"""

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

LEGACY_NAMES = (
    "ActiveOpenSemHybrid.py",
    "ARGUS.py",
    "ARGUS_base.py",
)


def main() -> None:
    for scene, params in SCENES.items():
        d = CFG / scene
        d.mkdir(parents=True, exist_ok=True)
        base = ACTIVE_OPEN_SEM_BASE.format(scene=scene, bbox=params["bbox"], xy=params["xy"])
        (d / "ActiveOpenSem_base.py").write_text(base, encoding="utf-8")
        (d / "ActiveOpenSemGeom.py").write_text(GEOM, encoding="utf-8")
        (d / "ActiveOpenSem.py").write_text(ACTIVE_OPEN_SEM, encoding="utf-8")
        for legacy in LEGACY_NAMES:
            legacy_path = d / legacy
            if legacy_path.exists():
                legacy_path.unlink()
        print(f"written configs for {scene}")


if __name__ == "__main__":
    main()
