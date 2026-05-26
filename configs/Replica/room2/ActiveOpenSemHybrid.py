"""ActiveOpenSemHybrid — hybrid planner stage 0. Results: .../ActiveOpenSemHybrid/run_N/"""

import os

from mmengine.config import read_base

with read_base():
    from .ActiveOpenSem import *

dirs["result_dir"] = os.path.join(
    "results", general["dataset"], general["scene"], "ActiveOpenSemHybrid",
)
planner["method"] = "active_gs_hybrid"
planner["semantic_exploration"] = dict(
    enabled_stages=[0],
    novelty_aggregation="mean",
    min_bank_masks=1,
    max_masks_per_candidate=40,
    max_semantic_candidates=8,
    log_semantic_scores=True,
    encode_keyframes_if_missing=True,
)
visualizer["vis_rgbd"] = False
slam["override"]["viz"] = dict(enter_interactive_post_online=False, visualize_cams=False)
