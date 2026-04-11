import os
from mmengine.config import read_base

with read_base():
    from .ActiveSem import *

##################################################
### Open-vocabulary CLIP index
# Inherits all settings from ActiveSem.py and
# adds the CLIP module for goal-directed navigation.
#
# Run with:
#   bash scripts/activesgm/run_replica.sh SCENE 1 ActiveSem-CLIP 0 0 1
#
# The 6th argument (USE_CLIP=1) is required so that
# cfg_loader does NOT strip the clip section.
##################################################
clip = dict(
    # Device for CLIP inference.
    # "cuda:1" keeps CLIP on the same GPU as OneFormer,
    # leaving cuda:0 free for mapping/planning.
    # Change to "cuda:0" for single-GPU setups.
    device = "cuda:0",

    # open_clip model identifier.
    # "ViT-B-32"  – fast,  512-dim embeddings (~160 MB)
    # "ViT-L-14"  – accurate, 768-dim embeddings (~430 MB)
    model_name = "ViT-B-32",

    # Pre-trained weights tag (open_clip convention).
    pretrained = "openai",

    # Index refresh frequency (planning steps).
    # Only NEW keyframes are encoded → overhead is low.
    update_every = 10,

    # Default top-K results for navigation queries.
    top_k = 5,
)
