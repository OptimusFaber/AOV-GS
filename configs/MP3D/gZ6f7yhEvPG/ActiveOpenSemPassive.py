"""ActiveOpenSemPassive — fixed NVS trajectory (no active planning) on MP3D."""

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
