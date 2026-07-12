"""
ActiveOpenSemGeom — SplaTAM + active_gs (geometry only, no semantics in planning).

Results: results/Replica/office0/ActiveGeom/run_N/

Run:

    python src/main/activesgm.py configs/Replica/office0/ActiveOpenSemGeom.py
"""

import os

from mmengine.config import read_base

with read_base():
    from .ActiveOpenSem_base import *


dirs["result_dir"] = os.path.join(
    "results",
    general["dataset"],
    general["scene"],
    "ActiveGeom",
)

visualizer["vis_rgbd"] = False

slam["override"]["viz"] = dict(
    enter_interactive_post_online=False,
    visualize_cams=False,
)
