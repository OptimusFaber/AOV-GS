"""Passive SplaTAM replay of ActiveSem/ActiveSGM trajectory + SAM/CLIP keyframes."""

import os

from mmengine.config import read_base

with read_base():
    from .ActiveOpenSem_base import *

dirs["result_dir"] = os.path.join(
    "results", general["dataset"], general["scene"], "ActiveSem",
)

slam["enable_active_planning"] = False
slam["save_clip_features"] = True
slam["save_keyframe_poses"] = True
slam["save_keyframes"] = True
slam["override"]["map_every"] = 5
slam["override"]["report_global_progress_every"] = 50
slam["override"]["save_checkpoints"] = False

planner["method"] = "predefined_traj"
planner["use_traj_pose"] = True
planner["SLAMData_dir"] = os.path.join(
    dirs["data_dir"], "replica_activesem_traj", general["scene"],
)
planner["post_refine_steps"] = 0
planner["max_refinement_steps"] = 0

_traj_path = os.path.join(planner["SLAMData_dir"], "traj.txt")
if os.path.isfile(_traj_path):
    with open(_traj_path, encoding="utf-8") as _tf:
        general["num_iter"] = sum(1 for _ln in _tf if _ln.strip())
else:
    general["num_iter"] = 5000

sam_clip["device"] = "cuda:0"
sam_clip["levels"] = ("s", "m", "l")
sam_clip["queue_size"] = 16
sam_clip["clip_batch_size"] = 32
sam_clip["max_masks_per_frame"] = 120

visualizer["vis_rgbd"] = False
slam["override"]["viz"] = dict(
    enter_interactive_post_online=False,
    visualize_cams=False,
)
