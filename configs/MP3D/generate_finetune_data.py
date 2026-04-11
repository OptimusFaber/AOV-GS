import numpy as np
import os

from mmengine.config import read_base

with read_base():
    from ..default import *



##################################################
### General
##################################################
general = dict(
    dataset="MP3D",
    scene="GdvgFV5R1Z5",
    num_iter=5000,
    device='cuda',
    semantic_dir= "./data/replica_v1/office_0/habitat/",
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

if sim["method"] == "habitat_v2":
    sim.update(
        habitat_cfg = os.path.join(dirs['cfg_dir'], "habitat.py")
    )

#####################
##### slam
####################
slam = dict(
    method="semsplatam"                                    # SLAM backbone method
)

slam.update(
        enable_active_planning = False,
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
        "MP3D", general['scene']
        ),
)

##################################################
### oneformer
##################################################

oneformer = dict(
    checkpoint="shi-labs/oneformer_ade20k_swin_large",
)




