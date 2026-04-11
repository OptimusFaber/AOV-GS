"""
ActiveOpenVocab config – Open-Vocabulary Language Field Pipeline  (office0)

Stage 1: geometric exploration (SplaTAM, no OneFormer)
         + parallel SAM+CLIP feature extraction for every keyframe.

Based on ActiveLang.py (office0), extended with:
  - slam.save_keyframes          = True   (save RGB keyframes to disk)
  - slam.save_keyframe_poses     = True   (save w2c poses to JSON)
  - slam.save_clip_features      = True   (enable SAMCLIPExtractor)
  - sam_clip.*                            (SAMCLIPExtractor settings)

Downstream stages (run separately after exploration):
  python scripts/train_language_autoencoder.py  ...
  python scripts/train_language_field.py        ...
"""

import numpy as np
import os

from mmengine.config import read_base

with read_base():
    from ...default import *


##################################################
### General
##################################################
general = dict(
    dataset = "Replica",
    scene   = "office0",
    num_iter = 2000,
    device  = "cuda"
)

##################################################
### Directories
##################################################
dirs = dict(
    data_dir   = "data/",
    result_dir = "results/",
    cfg_dir    = os.path.join("configs", general["dataset"], general["scene"])
)

##################################################
### Simulator
##################################################
sim = dict(
    method = "habitat_v2"
)

if sim["method"] == "habitat_v2":
    sim.update(
        habitat_cfg = os.path.join(dirs["cfg_dir"], "habitat.py")
    )

##################################################
### SLAM  (pure geometry – no OneFormer)
##################################################
slam = dict(
    method = "splatam",

    room_cfg               = f"{dirs['cfg_dir']}/../replica_splatam_s.py",
    enable_active_planning = True,
    dataset_eval_basedir   = "data/replica_sim_nvs",

    ### bounding box (office0) ###
    bbox_bound      = [[-2.1, 2.5], [-3.2, 2.0], [-1.3, 2.0]],
    bbox_voxel_size = 0.05,

    surface_dist_thre    = 0.5,
    find_free_indices_bs = 1000,

    ### Refinement ###
    refine_map_iter     = 60,
    use_global_keyframe = True,
    global_keyframe = dict(
        completeness_thre = 0.1,
        color_thre        = 34,
        depth_thre        = 0.01,
        quality_method    = "relative",
        quality_freq      = 100,
        quality_perc_thre = 30,
    ),

    ### Save keyframes + poses for offline language-field training ###
    save_keyframes      = True,
    save_keyframe_poses = True,

    ### Activate parallel SAM+CLIP extraction ###
    save_clip_features = True,

    ### SplaTAM override ###
    override = dict(
        map_every                    = 5,
        report_global_progress_every = 5,
        tracking = dict(
            use_gt_poses = True,
        )
    )
)

##################################################
### SAM + CLIP extractor settings
##################################################
sam_clip = dict(
    sam_ckpt_path      = "ckpts/sam_vit_b_01ec64.pth",
    clip_model         = "ViT-B-16",
    clip_pretrained    = "laion2b_s34b_b88k",
    device             = "cuda:1",   # упадёт на cuda:0 если второй GPU нет
    queue_size         = 8,          # уменьшено: меньше кадров "в полёте" → меньше пиковой нагрузки
    clip_batch_size    = 32,         # макс. масок за один forward CLIP
    max_masks_per_frame = 120,       # ограничение числа SAM-масок → предсказуемый пик памяти
)

##################################################
### Planner
##################################################
planner = dict(
    method = "active_gs",

    num_exploration_stage = 2,
    gs_z_levels = [
        [35],
        [20, 50],
    ],
    num_dir_samples  = [5, 15],
    xy_sampling_step = [1.0, 0.5],

    trans_step_size = 0.1,
    rot_step_size   = 10,

    surface_dist_thre = slam["surface_dist_thre"],

    explore_thre  = 0.005,
    color_ig_thre = 34,
    depth_ig_thre = 0.01,

    post_refinement_eval_freq = 100,

    up_dir       = [0, 0, 1],
    use_traj_pose = True,
    SLAMData_dir  = os.path.join(dirs["data_dir"], "Replica", general["scene"]),

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
    vis_rgbd           = False,
    vis_rgbd_max_depth = 10,
)
