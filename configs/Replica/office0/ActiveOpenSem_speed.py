"""
ActiveOpenSem speed profile for controlled A/B.

Focus: baseline throughput while keeping current behavior explicit.
"""

from mmengine.config import read_base

with read_base():
    from .ActiveOpenSem_base import *


sam_clip.update(
    queue_size=8,
    submit_timeout_s=1.0,
    clip_batch_size=32,
    max_masks_per_frame=120,
    sam_ckpt_path="ckpts/sam_vit_b_01ec64.pth",
    corrclip_mask_merge=True,
    corrclip_interclass_suppress_alpha=0.15,
)
