"""
ActiveOpenSem_base (office2) — SplaTAM + active_gs + SAM/CLIP feature extraction.
Shared base for ActiveOpenSem, ActiveGeom, and passive variants.
"""

import os

from mmengine.config import read_base

with read_base():
    from ...default import *

general = dict(
    dataset="Replica",
    scene="office2",
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
        room_cfg=f"{dirs['cfg_dir']}/../replica_splatam_s.py",
        enable_active_planning=True,
        dataset_eval_basedir="data/replica_sim_nvs",
        eval_during_training=True,
        eval_during_training_freq=200,
        eval_during_training_max_frames=None,
        bbox_bound=[[-3.6, 3.2], [-3.0, 5.4], [-1.3, 1.6]],
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
    xy_sampling_step=[1.0, 0.5],
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
