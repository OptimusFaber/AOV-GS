"""
ActiveOpenSemHybrid — SplaTAM + гибридный планировщик на stage 0.

Stage 0: geometry IG × distance × SAM+CLIP embedding novelty
         (идём туда, где эмбеддинги масок отличаются от уже собранных keyframes).
Stage 1: чистая геометрия (как active_gs).

Запуск:

    python src/main/activesgm.py configs/Replica/office0/ActiveOpenSemHybrid.py
"""

import os

from mmengine.config import read_base

with read_base():
    from .ActiveOpenSem import *


# Отдельная папка результатов (не ActiveOpenSem).
dirs["result_dir"] = os.path.join(
    "results",
    general["dataset"],
    general["scene"],
    "ActiveOpenSemHybrid",
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

# Headless-friendly: no live OpenCV/Qt windows (SSH / server without DISPLAY).
visualizer["vis_rgbd"] = False

slam["override"]["viz"] = dict(
    enter_interactive_post_online=False,
    visualize_cams=False,
)
