"""
ActiveOpenSem quality profile for controlled A/B.

Focus: maximize SAM+CLIP supervision coverage and mask quality.
"""

from mmengine.config import read_base

with read_base():
    from .ActiveOpenSem_base import *


sam_clip.update(
    # Favor coverage: larger queue + longer submit wait.
    queue_size=64,
    submit_timeout_s=4.0,
    # Keep more masks to preserve small objects.
    max_masks_per_frame=220,
    # Lower CLIP batch to reduce OOM risk for larger mask count.
    clip_batch_size=24,
    # Prefer higher-quality SAM if checkpoint exists.
    sam_ckpt_path="ckpts/sam_vit_h_4b8939.pth",
    # Slightly more conservative CorrCLIP suppression.
    corrclip_interclass_suppress_alpha=0.10,
)
