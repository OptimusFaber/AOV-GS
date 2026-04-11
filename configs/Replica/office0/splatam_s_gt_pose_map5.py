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
    scene = "office0",
    num_iter = 2000,
)

##################################################
### Directories
##################################################
dirs = dict(
    data_dir = "data/",
    result_dir = "results/",
    cfg_dir = os.path.join("configs", general['dataset'], general['scene'])
)


##################################################
### Simulator
##################################################
sim = dict(
    method = "habitat_v2"                                  # simulator method
)

if _base_.sim["method"] == "habitat_v2":
    _base_.sim.update(
        habitat_cfg = os.path.join(dirs['cfg_dir'], "habitat.py")
    )

##################################################
### SLAM
##################################################
slam = dict(
    method="splatam"                                     # SLAM backbone method
)

if slam["method"] == "splatam":
    slam.update(
        room_cfg        = f"{dirs['cfg_dir']}/splatam_s.py",   # Co-SLAM room configuration
        enable_active_planning = False,                             # enable/disable active planning

        ### override ###
        map_every = 5,
    )

##################################################
### Planner
##################################################
planner = dict(
    method= "predefined_traj",                           # planner method
    up_dir = [0, 0, 1], # up direction for planning pose
    use_traj_pose = True,                          # use pre-defined trajectory pose
    SLAMData_dir = os.path.join(                    # SLAM Data directory (for passive mapping or pre-defined trajectory pose)
        dirs["data_dir"], 
        "Replica", general['scene']
        ),
)

##################################################
### Visualization
##################################################
visualizer = dict(
    method = "active_lang",
    vis_rgbd        = True,                             # visualize RGB-D
    vis_rgbd_max_depth = 10

    ### mesh related ###
    # mesh_vis_freq = 500,                                # mesh save frequency
)

