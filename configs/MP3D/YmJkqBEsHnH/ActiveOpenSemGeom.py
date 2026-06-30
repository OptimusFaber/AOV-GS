"""ActiveOpenSemGeom — SplaTAM + active_gs (geometry-only) on MP3D.

Results: results/MP3D/YmJkqBEsHnH/ActiveGeom/run_N/

Run:
    bash scripts/aov-gs/01_slam_exploration_mp3d.sh YmJkqBEsHnH ActiveOpenSemGeom
    bash scripts/activesgm/run_mp3d_geom.sh YmJkqBEsHnH
"""

import os

from mmengine.config import read_base

with read_base():
    from .ActiveOpenSem_base import *

dirs["result_dir"] = os.path.join("results", general["dataset"], general["scene"], "ActiveGeom")

planner["method"] = "active_gs"

visualizer["vis_rgbd"] = False
slam["override"]["viz"] = dict(enter_interactive_post_online=False, visualize_cams=False)
