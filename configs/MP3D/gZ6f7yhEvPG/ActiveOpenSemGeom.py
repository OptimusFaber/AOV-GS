"""ActiveOpenSemGeom — SplaTAM + active_gs (geometry-only) on MP3D.

Results: results/MP3D/gZ6f7yhEvPG/ActiveGeom/run_N/

Run:
    bash scripts/aov-gs/01_slam_exploration_mp3d.sh gZ6f7yhEvPG ActiveOpenSemGeom
    bash scripts/activesgm/run_mp3d_geom.sh gZ6f7yhEvPG
"""

import os

from mmengine.config import read_base

with read_base():
    from .ActiveOpenSem_base import *

dirs["result_dir"] = os.path.join("results", general["dataset"], general["scene"], "ActiveGeom")

# Geometry baseline: ActiveGSPlanner (not hybrid v3).
planner["method"] = "active_gs"
planner["force_post_refinement_at_step"] = 4800
# Geometry uses SAM+CLIP only for embedding-bank collection.
slam["save_clip_features"] = True
slam["save_clip_every_kf"] = 3
slam["save_clip_max_step"] = 4000
sam_clip["queue_size"] = 2
sam_clip["clip_batch_size"] = 16
sam_clip["max_masks_per_frame"] = 48

visualizer["vis_rgbd"] = False
slam["override"]["viz"] = dict(enter_interactive_post_online=False, visualize_cams=False)
