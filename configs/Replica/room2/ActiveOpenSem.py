"""ActiveOpenSem — SAM+CLIP semantic exploration + hybrid v3 planner. Results: .../ActiveOpenSem/run_N/"""

import os

from mmengine.config import read_base

with read_base():
    from .ActiveOpenSem_base import *

dirs["result_dir"] = os.path.join(
    "results", general["dataset"], general["scene"], "ActiveOpenSem",
)
planner["method"] = "active_gs_hybrid_v3"
planner["seman_thre"] = 0.7
planner["max_revisit_count"] = 3
planner["semantic_exploration"] = dict(
    enabled_stages=[0],
    novelty_aggregation="mean",
    min_bank_masks=1,
    max_masks_per_candidate=40,
    max_semantic_candidates=8,
    log_semantic_scores=True,
    encode_keyframes_if_missing=True,
    post_refinement_semantic=True,
    post_refinement_max_keyframes=None,
)
visualizer["vis_rgbd"] = False
slam["override"]["viz"] = dict(enter_interactive_post_online=False, visualize_cams=False)
