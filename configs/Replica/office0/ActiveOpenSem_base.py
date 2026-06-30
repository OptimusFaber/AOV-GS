"""
ActiveOpenSem_base (office0) — direct port of ActiveSGM's
shared SAM+CLIP base: a SplaTAM backbone driven by the geometric
``active_gs`` planner (no OneFormer / no semantic head), augmented with
the AOV-GS parallel SAM+CLIP feature extractor so that per-keyframe
language features are written to disk during exploration.

Pipeline
--------

    ┌─────────────┐      ┌─────────────────┐      ┌─────────────┐
    │  Simulator  │ ---> │  SplaTAM SLAM   │ ---> │  active_gs  │
    └─────────────┘      │  (geometry)     │      │   planner   │
                         └───────┬─────────┘      └─────────────┘
                                 │ keyframes
                                 ▼
                         ┌─────────────────┐
                         │  SAM + CLIP     │
                         │   extractor     │
                         └─────────────────┘

Run with:

    bash scripts/aov-gs/01_slam_exploration.sh office0

After the run finishes, the result directory contains
``language_features/<frame>_f.npy`` (CLIP embeddings) and
``language_features/<frame>_s.npy`` (SAM mask sizes), ready for
**LangSplatV2** language-field training (``scripts/aov-gs/03_train_gaussian_lang_field.sh``).
"""

import os

from mmengine.config import read_base

with read_base():
    from ...default import *


##################################################
### General
##################################################
general = dict(
    dataset  = "Replica",
    scene    = "office0",
    num_iter = 2000,
    device   = "cuda",
)

##################################################
### Directories
##################################################
dirs = dict(
    data_dir   = "data/",
    result_dir = "results/",
    cfg_dir    = os.path.join("configs", general["dataset"], general["scene"]),
)


##################################################
### Simulator
##################################################
sim = dict(
    method = "habitat_v2",
)

if sim["method"] == "habitat_v2":
    sim.update(
        habitat_cfg = os.path.join(dirs["cfg_dir"], "habitat.py"),
    )

##################################################
### SLAM
##################################################
slam = dict(
    method = "splatam",
)

if slam["method"] == "splatam":
    slam.update(
        room_cfg               = f"{dirs['cfg_dir']}/../replica_splatam_s.py",
        enable_active_planning = True,
        dataset_eval_basedir   = "data/replica_sim_nvs",

        ### Validation during training ###
        eval_during_training            = False,
        eval_during_training_freq       = 200,
        eval_during_training_max_frames = None,

        ### bounding box (office0) ###
        bbox_bound      = [[-2.1, 2.5], [-3.2, 2.0], [-1.3, 2.0]],
        bbox_voxel_size = 0.05,

        surface_dist_thre    = 0.5,
        find_free_indices_bs = 1000,

        ### Refinement step ###
        refine_map_iter     = 60,
        use_global_keyframe = True,
        global_keyframe = dict(
            completeness_thre = 0.1,
            color_thre        = 34,   # smaller than this thre, add to global keyframe
            depth_thre        = 0.01, # NOT USED
            quality_method    = "relative",
            quality_freq      = 100,
            quality_perc_thre = 30,
        ),

        ### Save keyframes + poses for offline language-field training ###
        save_keyframes      = True,
        save_keyframe_poses = True,

        ### Parallel SAM+CLIP extractor (see [sam_clip] section below) ###
        save_clip_features = True,

        ### override ###
        override = dict(
            map_every                    = 5,
            report_global_progress_every = 5,
            save_checkpoints             = False,  # only stage/final via print_and_save_result
            tracking = dict(
                use_gt_poses = True,  # Use GT Poses for Tracking
            ),
        ),
    )


##################################################
### SAM + CLIP extractor
# Activated whenever ``slam.save_clip_features`` is True.
# ``sam_ckpt_path`` must point at a local SAM checkpoint (no auto-download).
# ``clip_model`` / ``clip_pretrained`` follow the open_clip convention.
##################################################
sam_clip = dict(
    # SAM backbone weights (downloaded once to ckpts/).
    #   sam_vit_b_01ec64.pth   – fast  (~375 MB)
    #   sam_vit_h_4b8939.pth   – big   (~2.4 GB) but higher quality
    sam_ckpt_path = "ckpts/sam_vit_b_01ec64.pth",

    # CLIP encoder (open_clip format).
    clip_model      = "ViT-B-16",
    clip_pretrained = "laion2b_s34b_b88k",

    # Where to run SAM+CLIP (pick cuda:1 on multi-GPU setups; falls back
    # to cuda:0 if only one GPU is available).
    device = "cuda:1",

    # Throughput / memory knobs.
    queue_size          = 8,    # max keyframes in-flight
    clip_batch_size     = 32,   # masks per CLIP forward
    max_masks_per_frame = 120,  # cap to bound peak GPU memory

    # CorrCLIP-inspired refinements for SAM+CLIP collection:
    # 1) merge semantically similar nearby masks (over-segmentation reduction),
    # 2) suppress inter-class embedding leakage.
    corrclip_mask_merge = True,
    corrclip_merge_sim_thresh = 0.86,
    corrclip_merge_dist_px = 80.0,
    corrclip_interclass_suppress_alpha = 0.15,
    corrclip_interclass_sim_thresh = 0.78,
    corrclip_interclass_sigma_px = 120.0,
)


##################################################
### Planner
##################################################
planner = dict(
    method = "active_gs",  # geometric planner [predefined_traj, active_gs]

    ### active_gs params ###
    max_exploration_steps = 1500,
    post_refine_steps     = 200,
    max_refinement_steps  = 200,
    num_exploration_stage = 2,
    gs_z_levels = [
        [35],
        [20, 50],
    ],
    num_dir_samples = [   # viewing direction sample number
        5,
        15,
    ],
    xy_sampling_step = [  # Unit: meter
        1.0,
        0.5,
    ],

    trans_step_size = 0.1,  # meter
    rot_step_size   = 10,   # degree

    surface_dist_thre = slam["surface_dist_thre"],

    ### Stop Criteria ###
    explore_thre              = 0.005,
    color_ig_thre             = 34,
    depth_ig_thre             = 0.01,
    post_refinement_eval_freq = 100,

    up_dir        = [0, 0, 1],   # up direction for planning pose
    use_traj_pose = True,        # use pre-defined trajectory pose
    SLAMData_dir  = os.path.join(  # SLAM data dir for trajectory poses
        dirs["data_dir"],
        "Replica", general["scene"],
    ),

    ### RRT ###
    local_planner_method = "RRTNaruto",
)

if planner["local_planner_method"] == "RRTNaruto":
    planner.update(
        rrt_step_size      = planner["trans_step_size"] / slam["bbox_voxel_size"],
        rrt_step_amplifier = 10,
        rrt_maxz           = 100,
        rrt_max_iter       = None,
        rrt_z_levels       = None,
        enable_eval        = False,
        enable_direct_line = True,
    )

##################################################
### Visualization
##################################################
visualizer = dict(
    method             = "active_gs",
    vis_rgbd           = True,    # visualize RGB-D
    vis_rgbd_max_depth = 10,
)
