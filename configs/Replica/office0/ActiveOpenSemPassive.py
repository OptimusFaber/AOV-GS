"""
ActiveOpenSemPassive — SplaTAM + фиксированная траектория (без active planning).

Робот следует по data/Replica/office0/traj.txt без выбора NBV / направления.
Результаты: results/Replica/office0/Passive/run_N/

Запуск:

    python src/main/activesgm.py configs/Replica/office0/ActiveOpenSemPassive.py
"""

import os

from mmengine.config import read_base

with read_base():
    from .ActiveOpenSem_base import *


dirs["result_dir"] = os.path.join(
    "results",
    general["dataset"],
    general["scene"],
    "Passive",
)

slam["enable_active_planning"] = False

planner["method"] = "predefined_traj"
planner["use_traj_pose"] = True
planner["SLAMData_dir"] = os.path.join(
    dirs["data_dir"],
    "Replica",
    general["scene"],
)

visualizer["vis_rgbd"] = False

slam["override"]["viz"] = dict(
    enter_interactive_post_online=False,
    visualize_cams=False,
)
