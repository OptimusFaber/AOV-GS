"""ActiveOpenSemGeom — geometry-only active_gs. Results: .../ActiveGeom/run_N/"""

import os

from mmengine.config import read_base

with read_base():
    from .ActiveOpenSem import *

dirs["result_dir"] = os.path.join("results", general["dataset"], general["scene"], "ActiveGeom")
visualizer["vis_rgbd"] = False
slam["override"]["viz"] = dict(enter_interactive_post_online=False, visualize_cams=False)
