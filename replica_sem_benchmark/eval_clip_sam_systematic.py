#!/usr/bin/env python3
"""
Systematic open-vocabulary concept segmentation benchmark.
**Default:** Meta SAM1 (``segment_anything``) × CLIP (``open_clip_torch``) — no
``sam2_video``, works with older ``transformers`` (CLIP/text side does not need
SAM2 Hub).

**Optional:** Hugging Face ``facebook/sam2-hiera-*`` mask-generation — requires
``transformers>=4.56`` (``Sam2VideoModel``). **SAM3** (``model_type: sam3_video``)
needs ``transformers`` **5.x** (e.g. **5.5.0+** on PyPI); 4.57.x cannot load it — see
``_format_sam3_needs_newer_transformers`` in this file.  For SAM3, Hub weights are
stored as ``Sam3VideoModel`` (``tracker_model.*``); this script remaps them into the
flat ``Sam3TrackerModel`` used by ``mask-generation`` so masks are not empty.

Two anti-fragmentation approaches
──────────────────────────────────
Approach 1 — ``threshold_grid`` (plain):
    CLIP embeddings for all cached SAM masks (once): default tight crops with black
    background; optional ``--use-boxes`` = padded bbox crops (background kept). Then for each
    (conf_threshold, mask_threshold) pair subset rows and CLIP-argmax per query.

Approach 2 — ``threshold_grid_dbscan``:
    Reuses the **same** precomputed embeddings as Approach 1. After threshold
    filtering, runs DBSCAN (cosine distance) on mask embeddings, merges masks
    within each cluster (pixel OR), then CLIP-argmax on **cluster** embeddings.
    Sweeps ``--dbscan_eps_grid`` × **three** fixed threshold pairs (see
    ``_THRESH_COMBOS_9``) — not the full 5×5 grid when ``--full_grid`` is on.

Metrics collected per run
──────────────────────────
    miou                    mean binary IoU over (image, class) pairs
    sa_co_f1                hit rate @ IoU=0.5 (simplified F1 for single-query mode)
    avg_masks_per_image     avg number of objects after filtering/clustering
    percent_fragmented      heuristic fragmentation rate (low score OR thin fill)
    avg_clip_score          avg cosine sim of selected mask to text query
    mAP_retrieval           mean AP over ranked mask retrieval per query
    avg_speed_ms            ms/frame: post-precompute (grid / grid+DBSCAN) or full
                            CLIP forward only when precompute was skipped
    vram_gb_per_frame       peak PyTorch **allocated** tensor memory (GB)
    vram_reserved_gb_per_frame  peak **reserved** pool (GB) — usually closer to nvidia-smi

VRAM strategy (16 GiB limit)
──────────────────────────────
SAM and CLIP are never resident simultaneously:
    1. Load SAM → generate & cache masks for all samples → unload SAM.
    2. Load CLIP → embed cached masks (all approaches) → unload CLIP.
    3. Repeat for next (SAM, CLIP) pair.

This keeps peak VRAM ≤ max(SAM_peak, CLIP_peak) ≤ ~7 GB even for
facebook/sam2-hiera-large + ViT-L-14.

Output
──────
    pandas DataFrame → CSV (default: ``replica_sem_benchmark/results/sweep_results.csv``)
    + optional Weights & Biases logging.
    Rich table (or plain pandas fallback) printed to stdout.

``--queries_from_gt`` (sweep mode)
──────────────────────────────────
    Builds **one** query list: every unique semantic class that appears in **any**
    frame of the manifest (union over all ``semantic/*.npy`` in the run). The same
    CLIP text queries and class IDs are evaluated on **every** image; per-class
    metrics skip empty GT on a frame unless ``--include_empty_gt``.

Usage
──────
    # Quick smoke (SAM1: pass --sam1; default local ckpt ckpts/sam_vit_b_01ec64.pth)
    python replica_sem_benchmark/eval_clip_sam_systematic.py \\
        --manifest replica_sem_benchmark/manifest.json \\
        --queries_from_gt \\
        --sam1 \\
        --max_samples 2 \\
        --clip_models ViT-B-32/laion2b_s34b_b79k \\
        --approaches threshold_grid \\
        --out_csv replica_sem_benchmark/results/smoke_test.csv

    # Same with explicit SAM1 path
    python replica_sem_benchmark/eval_clip_sam_systematic.py \\
        ... --sam1 --sam_models local --sam_ckpt /path/to/sam_vit_b_01ec64.pth

    # Preset: all SAM1 + HF SAM2 + SAM3 under --sam_models_root (enable each family)
    #   python ... --sam1 --sam2 --sam3 --sam_models preset:hf_local --sam_models_root /mnt/data/model-ckpts/sam

    # Optional: HF SAM2 (needs pip install -U \"transformers>=4.56\")
    #   --sam_models facebook/sam2-hiera-tiny

    # Full sweep (all CLIP presets × default SAM1; threshold_grid + threshold_grid_dbscan)
    python replica_sem_benchmark/eval_clip_sam_systematic.py \\
        --manifest replica_sem_benchmark/manifest.json \\
        --queries_from_gt \\
        --max_samples 20 \\
        --full_grid \\
        --out_csv replica_sem_benchmark/results/full_sweep.csv \\
        --wandb_project my-project

    # Legacy single-run mode (backward compat with old eval_clip_sam_miou.py)
    python replica_sem_benchmark/eval_clip_sam_systematic.py \\
        --mode legacy \\
        --manifest replica_sem_benchmark/manifest.json \\
        --queries_from_gt \\
        --sam_ckpt ckpts/sam_vit_b_01ec64.pth \\
        --clip_model ViT-B-16 --clip_pretrained laion2b_s34b_b88k

Requirements: open_clip_torch, segment_anything, scikit-learn, pandas, pillow,
torch.  ``transformers`` is only needed if you use ``--sam_models facebook/...``
(HF SAM2).
Optional    : rich, wandb, transformers>=4.56 (HF SAM2 only)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

# ─── Project path setup ───────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent       # replica_sem_benchmark/
_PROJ = _HERE.parent                          # New-Proj/
_DEFAULT_OUT_CSV = _HERE / "results" / "sweep_results.csv"
_SCRIPTS = _PROJ / "scripts"
for _p in (str(_SCRIPTS), str(_PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from clip_model_catalog import CLIP_VRAM_ESTIMATES_GB as _CLIP_VRAM
    from clip_model_catalog import clip_configs_for_eval
except ImportError:
    from replica_sem_benchmark.clip_model_catalog import (
        CLIP_VRAM_ESTIMATES_GB as _CLIP_VRAM,
        clip_configs_for_eval,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Default sweep: SAM1 (segment_anything). No Hugging Face, no sam2_video.
SAM_MODELS_DEFAULT: list[str] = ["local"]

# Optional Hub IDs (only if you install transformers>=4.56). Not used by default.
SAM_HF_MODELS_ALL: list[str] = [
    "facebook/sam2-hiera-tiny",
    "facebook/sam2-hiera-small",
    "facebook/sam2-hiera-base-plus",
    "facebook/sam2-hiera-large",
    "facebook/sam3",
    "facebook/sam3.1",
]

# Rough peak VRAM (GB) when running SAM alone; used for skip-check.
_SAM_VRAM: dict[str, float] = {
    "facebook/sam2-hiera-tiny":      1.5,
    "facebook/sam2-hiera-small":     2.5,
    "facebook/sam2-hiera-base-plus": 3.5,
    "facebook/sam2-hiera-large":     5.0,
    "facebook/sam2.1-hiera-tiny":    1.5,
    "facebook/sam2.1-hiera-small":   2.5,
    "facebook/sam2.1-hiera-base-plus": 3.5,
    "facebook/sam2.1-hiera-large":   5.0,
    "facebook/sam3":                 6.0,
    "facebook/sam3.1":               7.0,
    "local":                         2.5,   # SAM1 ViT-B (segment_anything)
    "sam1":                          2.5,
}


def _sam_vram_estimate(sam_id: str) -> float:
    """VRAM guess for Hub IDs and for local snapshot paths under /mnt/data/...."""
    if sam_id in _SAM_VRAM:
        return _SAM_VRAM[sam_id]
    low = sam_id.lower()
    name = Path(sam_id).name.lower()
    if low.endswith(".pth") or low.endswith(".pt"):
        if "vit_h" in low or "vit-h" in low:
            return 7.0
        if "vit_l" in low or "vit-l" in low:
            return 5.0
        return 2.5  # vit_b default
    blob = f"{low} {name}"
    if "sam3.1" in blob or "sam3-1" in blob:
        return 7.0
    if "sam3" in blob:
        return 6.0
    if "large" in blob:
        return 5.0
    if "base-plus" in blob or "base_plus" in blob:
        return 3.5
    if "small" in blob:
        return 2.5
    if "tiny" in blob:
        return 1.5
    return 6.0


# Default layout (see ``organize_model_ckpts.sh`` / ``download_model_ckpts.sh``)::
#   /mnt/data/model-ckpts/sam/<model_name>/   one folder per model (HF: config.json)
#   /mnt/data/model-ckpts/sam/sam1/*.pth      segment_anything (no config.json)
#   /mnt/data/model-ckpts/clip/open_clip/     OPENCLIP_CACHE (optional)
#   /mnt/data/model-ckpts/clip/huggingface_hub/  HF_HOME (Hub root; weights under ``hub/``)
# Legacy: ``sam/hf/<model>/`` or top-level ``hf/<model>/`` — still detected when scanning.
DEFAULT_MODEL_CKPTS_ROOT = Path("/mnt/data/model-ckpts")
_CANON_SAM1 = DEFAULT_MODEL_CKPTS_ROOT / "sam" / "sam1"


def default_sam_hf_snapshots_root() -> Path:
    """Root for SAM weights: ``sam/`` (each subfolder = one model where applicable)."""
    return DEFAULT_MODEL_CKPTS_ROOT / "sam"


def _resolve_sam_snapshot_scan_dir(root: Path) -> Path:
    """
    Scan either flat ``sam/<model>/`` or legacy bucket ``sam/hf/<model>/`` / ``hf/<model>/``.
    """
    if not root.is_dir():
        return root
    for sub in root.iterdir():
        if not sub.is_dir() or sub.name in ("sam1",):
            continue
        if (sub / "config.json").is_file():
            return root
    for bucket in (root / "hf", DEFAULT_MODEL_CKPTS_ROOT / "hf"):
        if bucket.is_dir() and any(
            p.is_dir() and (p / "config.json").is_file() for p in bucket.iterdir()
        ):
            return bucket
    return root


def expand_hf_local_snapshots(
    root: Path,
    *,
    exclude_substrings: tuple[str, ...] = (),
) -> list[str]:
    """
    Each subfolder with ``config.json`` is one HF model snapshot (SAM2 / SAM3).
    ``exclude_substrings`` — e.g. ``("sam3",)`` to skip SAM3 on small GPUs.
    """
    scan = _resolve_sam_snapshot_scan_dir(root)
    if not scan.is_dir():
        return []
    out: list[str] = []
    for sub in sorted(scan.iterdir()):
        if not sub.is_dir():
            continue
        if not (sub / "config.json").is_file():
            continue
        low = sub.name.lower()
        if any(ex in low for ex in exclude_substrings):
            continue
        out.append(str(sub.resolve()))
    return out


def expand_sam1_checkpoints_under_root(root: Path) -> list[str]:
    """
    Meta SAM1 ``segment_anything`` checkpoints: ``<root>/sam1/*.pth`` (flat).
    Same layout as ``/mnt/data/model-ckpts/sam/sam1/sam_vit_b_01ec64.pth``.
    """
    d = root / "sam1"
    if not d.is_dir():
        return []
    return [str(p.resolve()) for p in sorted(d.glob("*.pth")) if p.is_file()]


def expand_preset_sam_dir_models(
    root: Path,
    *,
    exclude_sam3_hf_folders: bool,
) -> list[str]:
    """
    Full sweep under a SAM ckpt root: SAM1 ``.pth`` files + HF snapshots (config.json).
    ``preset:hf_local*`` expands to this list so one tree can mix SAM1, SAM2, SAM3.
    """
    sam1 = expand_sam1_checkpoints_under_root(root)
    excl = ("sam3",) if exclude_sam3_hf_folders else ()
    hf = expand_hf_local_snapshots(root, exclude_substrings=excl)
    return sam1 + hf

# Default CLIP list: ``clip_model_catalog.clip_configs_for_eval()`` (see that module).

# Reduced threshold set (user-selected).
# Only these (conf_threshold, mask_threshold) pairs are evaluated unless --full_grid is set.
_THRESH_COMBOS_9: list[tuple[float, float]] = [
    (0.9, 0.5),
    (0.9, 0.9),
    (0.5, 0.5),
]
# Previous 9-combo subset (kept for quick restore):
# _THRESH_COMBOS_9 = [
#     (0.5, 0.5), (0.5, 0.7), (0.5, 0.9),
#     (0.7, 0.5), (0.7, 0.7), (0.7, 0.9),
#     (0.9, 0.5), (0.9, 0.7), (0.9, 0.9),
# ]
_CONF_VALS  = [0.5, 0.6, 0.7, 0.8, 0.9]
_MASK_VALS  = [0.5, 0.6, 0.7, 0.8, 0.9]

# Score threshold for SAM mask caching (generate everything, filter later)
_SAM_CACHE_MIN_SCORE = 0.10


# ══════════════════════════════════════════════════════════════════════════════
# RESULT RECORD
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunRecord:
    """One benchmark configuration row."""

    sam_model:                  str   = ""
    clip_model:                 str   = ""
    clip_pretrained:            str   = ""
    threshold:                  float = 0.0
    mask_threshold:             float = 0.0
    post_processing:            str   = "None"
    avg_speed_ms_per_frame:     float = float("nan")
    vram_gb_per_frame:          float = float("nan")  # max_memory_allocated peak
    vram_reserved_gb_per_frame: float = float("nan")  # max_memory_reserved peak (≈ driver/nvidia-smi)
    miou:                       float = float("nan")
    sa_co_f1:                   float = float("nan")
    recall_obj:                 float = float("nan")  # Approach 1: any-mask hit rate vs GT object
    precision_hit:              float = float("nan")  # Approach 2: chosen-mask hit rate vs GT object (optional)
    avg_masks_per_image:        float = float("nan")
    percent_fragmented_masks:   float = float("nan")
    avg_clip_score_selected:    float = float("nan")
    mAP_retrieval:              float = float("nan")
    clip_crop:                  str   = "tight"  # "tight" | "bbox_pad_N" (CLIP input only; metrics still use masks)


# ══════════════════════════════════════════════════════════════════════════════
# SAM2 / SAM3 BACKEND  (HuggingFace transformers pipeline)
# ══════════════════════════════════════════════════════════════════════════════

# Hub checkpoints (facebook/sam2-hiera-*, sam2.1-*, …) declare ``model_type: sam2_video``
# and class ``Sam2VideoModel``. That stack landed in transformers around **4.56**.
_HF_SAM2_MIN_VERSION = (4, 56, 0)


def _transformers_version_tuple() -> tuple[int, int, int]:
    try:
        import transformers as tr
        parts = str(tr.__version__).split(".")[:3]
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    except Exception:
        return (0, 0, 0)


def _hf_sam_supported_by_transformers() -> bool:
    return _transformers_version_tuple() >= _HF_SAM2_MIN_VERSION


def _partition_sam_models_for_transformers(sam_ids: list[str]) -> tuple[list[str], list[str]]:
    """
    HF SAM2/3 (``sam2_video`` / ``Sam2VideoModel``) requires ``transformers>=4.56``.
    SAM1 (``local`` / ``*.pth``) always runs. Returns ``(kept, skipped)``.
    """
    if _hf_sam_supported_by_transformers():
        return sam_ids, []
    skipped = [s for s in sam_ids if not _is_local_sam1(s)]
    kept = [s for s in sam_ids if _is_local_sam1(s)]
    return kept, skipped


def _transformers_supports_sam3_video() -> bool:
    """
    SAM3 checkpoints use ``model_type: sam3_video``. This is newer than SAM2's
    ``sam2_video`` and may be absent even in ``transformers>=4.56``.
    """
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        return "sam3_video" in CONFIG_MAPPING
    except Exception:
        return False


def _looks_like_hf_sam3(model_id: str) -> bool:
    """True if path/Hub id points to a SAM3 snapshot (not SAM2)."""
    s = model_id.replace("\\", "/").lower()
    if "facebook/sam3" in s or "/sam3/" in s:
        return True
    if s.rstrip("/").endswith("/sam3") or s.rstrip("/").endswith("/sam3.1"):
        return True
    p = Path(model_id)
    if p.is_dir() and (p / "config.json").is_file():
        try:
            data = json.loads((p / "config.json").read_text(encoding="utf-8"))
            return str(data.get("model_type", "")).lower() == "sam3_video"
        except Exception:
            pass
    return False


def _classify_sam_model_family(sam_id: str) -> str:
    """
    Rough family for sweep filtering: ``sam1`` (Meta .pth / local), ``sam3`` (HF sam3_video),
    else ``sam2`` (HF SAM2 / sam2_video and any other HF snapshot not classified as SAM3).
    """
    if _is_local_sam1(sam_id):
        return "sam1"
    if _looks_like_hf_sam3(sam_id):
        return "sam3"
    return "sam2"


def _filter_sam_models_by_family_flags(sam_ids: list[str], args) -> list[str]:
    """
    Keep only models whose family matches enabled ``--sam1`` / ``--sam2`` / ``--sam3`` flags.
    If **none** of these flags are set, returns an empty list (caller should error in sweep).
    """
    if not getattr(args, "sam1", False) and not getattr(args, "sam2", False) and not getattr(
        args, "sam3", False
    ):
        return []
    out: list[str] = []
    for s in sam_ids:
        fam = _classify_sam_model_family(s)
        if fam == "sam1" and args.sam1:
            out.append(s)
        elif fam == "sam2" and args.sam2:
            out.append(s)
        elif fam == "sam3" and args.sam3:
            out.append(s)
    return out


def _format_sam3_needs_newer_transformers(model_id: str) -> str:
    """
    SAM3 uses ``model_type: sam3_video``. That is **not** the same as SAM2's ``sam2_video``.
    Weights under ``/mnt/.../sam/sam3`` are fine — the failure is the **Python package**
    ``transformers`` not registering ``sam3_video`` (4.x does not; use **5.x**, e.g. 5.5.0+ on PyPI).

    Older HF threads mentioned ``--pre`` or git install; stable **5.x is on PyPI** now.
    """
    ver = ".".join(str(x) for x in _transformers_version_tuple())
    return "\n".join(
        [
            f"SAM3 checkpoint {model_id!r} is present, but this Python environment cannot load it.",
            "",
            "Reason: config says ``model_type: sam3_video``. Your installed ``transformers`` "
            f"({ver}) does not register that architecture. This is NOT a missing-weights issue.",
            "",
            "Fix (recommended — PyPI has transformers 5.x, e.g. 5.5.0):",
            "  pip install -U \"transformers>=5.5.0\"",
            "",
            "If you still need bleeding-edge fixes:",
            "  pip install -U git+https://github.com/huggingface/transformers.git",
            "",
            "Then re-run the same command; local paths under /mnt/data/.../sam/sam3 stay valid.",
        ]
    )


def _import_transformers_pipeline():
    """
    Import ``transformers.pipeline`` for HF SAM2.

    A misleading ``ModuleNotFoundError: Could not import module 'pipeline'`` often means
    **not** transformers itself but a broken dependency chain: ``accelerate`` pulls code
    that imports ``boto3`` from **user** ``site-packages`` (``~/.local``) with missing
    deps (e.g. ``jmespath``). Use ``PYTHONNOUSERSITE=1`` or ``pip install jmespath``.
    """
    try:
        from transformers import pipeline as hf_pipeline

        return hf_pipeline
    except ModuleNotFoundError as e:
        cause = e.__cause__ or e
        chain = repr(cause)
        extra = ""
        if "pipeline" in str(e) or "jmespath" in chain.lower() or "boto3" in chain.lower():
            extra = (
                "\n\nLikely cause: **user** site-packages (``~/.local``) shadowing conda — "
                "``boto3``/``botocore`` used by ``accelerate`` import path, missing ``jmespath`` "
                "or similar. Try:\n"
                "  export PYTHONNOUSERSITE=1\n"
                "  # or: pip install jmespath\n"
                "  # or: remove broken boto3 from ~/.local for this Python\n"
            )
        raise RuntimeError(
            f"Cannot import transformers.pipeline (needed for HF SAM2): {e!s}\n"
            f"Underlying: {chain}{extra}"
        ) from e


def _format_hf_sam_load_error(model_id: str, exc: BaseException) -> str:
    exc_s = str(exc).lower()
    if "sam3_video" in exc_s:
        return "\n".join(
            [
                f"Failed to load Hugging Face SAM model {model_id!r}: {exc}",
                "",
                _format_sam3_needs_newer_transformers(model_id),
            ]
        )
    ver = _transformers_version_tuple()
    vstr = ".".join(str(x) for x in ver)
    lines = [
        f"Failed to load Hugging Face SAM model {model_id!r}: {exc}",
        "",
        "Hub checkpoints for Meta SAM2 use architecture ``Sam2VideoModel`` "
        "(config key ``model_type: sam2_video``). Your ``transformers`` must be "
        f"recent enough (>= {_HF_SAM2_MIN_VERSION[0]}.{_HF_SAM2_MIN_VERSION[1]}). "
        f"Detected: {vstr}.",
        "",
        "  pip install -U \"transformers>=4.56.0\"",
        "",
        "Alternatively, use segment_anything (SAM1) without Hugging Face:",
        "  --sam_models local --sam_ckpt ckpts/sam_vit_b_01ec64.pth",
    ]
    return "\n".join(lines)


def _collect_pretrained_weight_paths(model_id: str) -> list[Path]:
    """
    Paths to ``*.safetensors`` / ``pytorch_model.bin`` for a local snapshot or a
    Hugging Face Hub id (uses ``snapshot_download`` cache).
    """
    root = Path(model_id)
    if root.is_dir() and (root / "config.json").is_file():
        paths = sorted(root.glob("*.safetensors"))
        binp = root / "pytorch_model.bin"
        if binp.is_file():
            paths.append(binp)
        return paths
    try:
        from huggingface_hub import snapshot_download

        cached = Path(snapshot_download(repo_id=str(model_id), local_files_only=True))
        paths = sorted(cached.glob("*.safetensors"))
        binp = cached / "pytorch_model.bin"
        if binp.is_file():
            paths.append(binp)
        if paths:
            return paths
    except Exception:
        pass
    try:
        from huggingface_hub import snapshot_download

        cached = Path(snapshot_download(repo_id=str(model_id)))
        paths = sorted(cached.glob("*.safetensors"))
        binp = cached / "pytorch_model.bin"
        if binp.is_file():
            paths.append(binp)
        return paths
    except Exception:
        return []


def _load_raw_state_dict_from_file(path: Path) -> dict[str, Any]:
    """Load a single HF weight file (sharded safetensors or pytorch_model.bin)."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path)))
    import torch

    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "state_dict" in blob:
        blob = blob["state_dict"]
    if not isinstance(blob, dict):
        return {}
    return blob


def _remap_sam3_video_tracker_weights_into_model(model: Any, model_id: str) -> int:
    """
    ``facebook/sam3`` stores ``Sam3VideoModel`` tensors under ``tracker_model.*``.
    The ``mask-generation`` pipeline instantiates a **flat** ``Sam3TrackerModel``
    (only ``tracker_config``); default ``from_pretrained`` then leaves
    ``mask_decoder`` / ``prompt_encoder`` randomly initialized and reports
    MISSING / UNEXPECTED keys.  Copy ``tracker_model.*`` into the flat module
    by stripping the prefix so AMG returns real masks.
    """
    cls_name = getattr(model, "__class__", type(model)).__name__
    if cls_name != "Sam3TrackerModel":
        try:
            from transformers.models.sam3_tracker.modeling_sam3_tracker import (
                Sam3TrackerModel,
            )

            if not isinstance(model, Sam3TrackerModel):
                return 0
        except Exception:
            return 0

    paths = _collect_pretrained_weight_paths(model_id)
    if not paths:
        return 0

    merged: dict[str, Any] = {}
    for p in paths:
        merged.update(_load_raw_state_dict_from_file(p))

    prefix = "tracker_model."
    if not any(k.startswith(prefix) for k in merged):
        return 0

    tgt = model.state_dict()
    to_load: dict[str, Any] = {}
    for k, v in merged.items():
        if not k.startswith(prefix):
            continue
        nk = k[len(prefix) :]
        if nk not in tgt:
            continue
        if getattr(v, "shape", None) != tgt[nk].shape:
            continue
        w = v
        if hasattr(w, "to") and tgt[nk].dtype != w.dtype:
            w = w.to(dtype=tgt[nk].dtype)
        to_load[nk] = w

    if not to_load:
        return 0

    model.load_state_dict(to_load, strict=False)
    return len(to_load)


class SamHFBackend:
    """
    Thin wrapper around the HuggingFace ``pipeline("mask-generation")`` API.
    Normalises the output to a list of mask-dicts compatible with the old
    ``SamAutomaticMaskGenerator`` schema.

    Always loads weights in **fp32**: SAM2 postprocess uses ``torchvision`` NMS, which
    raises ``RuntimeError: dets should have the same type as scores`` when internal
    boxes/scores dtypes disagree under fp16.
    """

    def __init__(self, model_id: str, device: torch.device, half: bool = True):
        hf_pipeline = _import_transformers_pipeline()

        self.model_id = str(model_id)
        self.device   = device
        dev = device.index if device.type == "cuda" else -1
        # fp32 only — see class docstring (torchvision batched_nms dtype mismatch in fp16).
        dtype = torch.float32

        try:
            try:
                self._pipe = hf_pipeline(
                    "mask-generation",
                    model=self.model_id,
                    device=dev,
                    dtype=dtype,
                )
            except TypeError:
                self._pipe = hf_pipeline(
                    "mask-generation",
                    model=self.model_id,
                    device=dev,
                    torch_dtype=dtype,
                )
        except Exception as e:
            raise RuntimeError(_format_hf_sam_load_error(model_id, e)) from e

        self._pipe.model.eval()
        if _looks_like_hf_sam3(self.model_id):
            n = _remap_sam3_video_tracker_weights_into_model(self._pipe.model, self.model_id)
            if n > 0:
                print(
                    f"SAM3: remapped {n} weight tensors from ``tracker_model.*`` "
                    f"into flat Sam3TrackerModel (mask-generation).",
                    file=sys.stderr,
                )
            self._pipe.model.eval()

    def generate(
        self,
        image_rgb:        np.ndarray,
        points_per_batch: int   = 64,
        min_score:        float = _SAM_CACHE_MIN_SCORE,
    ) -> list[dict]:
        """Return mask-dicts (keys: segmentation, score, predicted_iou, area, bbox)."""
        pil = Image.fromarray(image_rgb)
        with torch.inference_mode():
            try:
                raw = self._pipe(
                    pil,
                    points_per_batch=points_per_batch,
                    pred_iou_thresh=min_score,
                    stability_score_thresh=min_score,
                )
            except TypeError:
                # Older pipeline versions may not accept threshold kwargs
                raw = self._pipe(pil, points_per_batch=points_per_batch)

        return _normalise_hf_output(raw, image_rgb.shape[:2])

    def unload(self) -> None:
        del self._pipe
        _flush_vram()


class Sam1Backend:
    """
    Original Meta ``segment_anything.SamAutomaticMaskGenerator`` (SAM1, local .pth).
    Same mask schema as ``sam_query_match.load_sam_generator`` — no Hugging Face.
    """

    def __init__(self, ckpt: Path, device: torch.device):
        import sam_query_match as sqm

        self.model_id = f"sam1:{ckpt.name}"
        self._generator, self._model_type = sqm.load_sam_generator(str(ckpt), device)

    def generate(
        self,
        image_rgb:        np.ndarray,
        points_per_batch: int   = 64,   # SAM1 API has no batch knob; ignored
        min_score:        float = _SAM_CACHE_MIN_SCORE,
    ) -> list[dict]:
        masks = self._generator.generate(image_rgb)
        out: list[dict] = []
        for m in masks:
            md = dict(m)
            piou = float(md.get("predicted_iou", 0.5))
            stab = float(md.get("stability_score", piou))
            md["predicted_iou"] = piou
            # Grid ``mask_threshold`` column uses ``score`` (HF masks expose one scalar)
            md["score"] = stab
            out.append(md)
        return out

    def unload(self) -> None:
        del self._generator
        _flush_vram()


def _is_local_sam1(sam_id: str) -> bool:
    s = sam_id.strip()
    low = s.lower()
    if low in ("local", "sam1", "segment_anything", "sam_v1"):
        return True
    return s.endswith((".pth", ".pt"))


def _default_sam1_ckpt() -> Path:
    """Prefer ``sam/sam1``, then legacy ``sam1``, then project ``ckpts/``."""
    for p in (
        _CANON_SAM1 / "sam_vit_b_01ec64.pth",
        DEFAULT_MODEL_CKPTS_ROOT / "sam1" / "sam_vit_b_01ec64.pth",
    ):
        if p.is_file():
            return p
    return _resolve("ckpts/sam_vit_b_01ec64.pth")


def _resolve_sam1_checkpoint(sam_id: str, sam_ckpt: str | Path | None) -> Path:
    s = sam_id.strip()
    if s.endswith((".pth", ".pt")):
        p = Path(s)
        return p if p.is_absolute() else _resolve(p)
    if not sam_ckpt:
        sam_ckpt = _default_sam1_ckpt()
    p = Path(sam_ckpt)
    return p if p.is_absolute() else _resolve(p)


def load_sam_backend(
    sam_id: str,
    device: torch.device,
    sam_ckpt: str | Path | None = None,
    *,
    half: bool = True,
) -> SamHFBackend | Sam1Backend:
    """
    * ``local`` / ``sam1`` / ``*.pth`` → ``segment_anything`` (SAM1); default
      checkpoint ``ckpts/sam_vit_b_01ec64.pth`` if ``--sam_ckpt`` omitted.
    * ``facebook/sam2-hiera-*`` → Hugging Face mask-generation (optional;
      ``transformers>=4.56``, ``Sam2VideoModel`` / ``sam2_video``).
    """
    if _is_local_sam1(sam_id):
        ck = _resolve_sam1_checkpoint(sam_id, sam_ckpt)
        if not ck.is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {ck}")
        return Sam1Backend(ck, device)
    if _looks_like_hf_sam3(sam_id) and not _transformers_supports_sam3_video():
        raise RuntimeError(_format_sam3_needs_newer_transformers(sam_id))
    return SamHFBackend(sam_id, device, half=half)


def _normalise_hf_output(
    raw,
    img_hw: tuple[int, int],
) -> list[dict]:
    """Convert the heterogeneous HF pipeline output to a uniform list of dicts."""
    H, W = img_hw

    # Unpack different possible output shapes
    if isinstance(raw, dict):
        masks_raw  = raw.get("masks", [])
        scores_raw = raw.get("scores", [])
        if hasattr(scores_raw, "tolist"):
            scores_raw = scores_raw.tolist()
        if not scores_raw:
            scores_raw = [1.0] * len(masks_raw)
    elif isinstance(raw, list):
        # Each element is {"mask": ..., "score": ...} or {"masks": ..., "scores": ...}
        masks_raw  = []
        scores_raw = []
        for r in raw:
            m = r.get("mask", r.get("masks"))
            s = r.get("score", r.get("scores", 1.0))
            if isinstance(s, (list, np.ndarray)):
                s = float(s[0])
            masks_raw.append(m)
            scores_raw.append(float(s))
    else:
        return []

    out: list[dict] = []
    for mask_item, score in zip(masks_raw, scores_raw):
        seg = _item_to_bool_mask(mask_item)
        if seg is None or seg.ndim != 2:
            continue
        # Ensure spatial dims match the source image
        if seg.shape != (H, W):
            seg = cv2.resize(
                seg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        area = int(seg.sum())
        if area == 0:
            continue

        ys, xs = np.where(seg)
        bbox = [
            int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        ]
        out.append({
            "segmentation":  seg,
            "score":         float(score),
            "predicted_iou": float(score),
            "area":          area,
            "bbox":          bbox,
        })
    return out


def _item_to_bool_mask(item) -> np.ndarray | None:
    if item is None:
        return None
    if isinstance(item, Image.Image):
        arr = np.array(item)
    elif isinstance(item, torch.Tensor):
        arr = item.cpu().numpy()
    elif isinstance(item, np.ndarray):
        arr = item
    else:
        try:
            arr = np.array(item)
        except Exception:
            return None
    if arr.ndim == 3:
        arr = arr.squeeze(axis=0 if arr.shape[0] == 1 else -1)
    return arr.astype(bool)


# ══════════════════════════════════════════════════════════════════════════════
# CLIP BACKEND  (open_clip_torch)
# ══════════════════════════════════════════════════════════════════════════════

def _default_clip_hf_home() -> Path | None:
    """``.../clip/huggingface_hub`` if ``hub/`` exists (same layout as ``hf download``)."""
    root = DEFAULT_MODEL_CKPTS_ROOT / "clip" / "huggingface_hub"
    if (root / "hub").is_dir():
        return root.resolve()
    return None


def _ensure_hf_home_for_clip_cache() -> None:
    """
    If the user did not ``export HF_HOME`` but the default disk cache exists, set
    ``HF_HOME`` so ``huggingface_hub`` resolves ``hub/models--laion--…`` instead of
    ``~/.cache/huggingface``.
    """
    if os.environ.get("HF_HOME"):
        return
    clip_home = _default_clip_hf_home()
    if clip_home is not None:
        os.environ["HF_HOME"] = str(clip_home)


def _open_clip_weight_cache_dir() -> str | None:
    """
    Directory for OpenCLIP ``cache_dir`` passed to ``huggingface_hub`` — must be the
    **Hub cache root** (folder that **directly** contains ``models--org--repo``), i.e.
    ``HF_HOME/hub`` or ``HF_HUB_CACHE``.

    **Do not** point this at ``OPENCLIP_CACHE`` for Laion/OpenCLIP HF weights: that
    path is usually a different tree; if it wins, Hub looks in the wrong place and
    re-downloads ``open_clip_model.safetensors``.
    """
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        return str(Path(hf_hub_cache).expanduser().resolve())
    hf = os.environ.get("HF_HOME")
    if hf:
        root = Path(hf).expanduser().resolve()
        hub = root / "hub"
        if hub.is_dir():
            return str(hub)
        return str(root)
    hub = DEFAULT_MODEL_CKPTS_ROOT / "clip" / "huggingface_hub" / "hub"
    if hub.is_dir():
        return str(hub.resolve())
    # Legacy / URL downloads (non-Hub layout); only used if no HF hub cache exists.
    for key in ("OPENCLIP_CACHE", "OPEN_CLIP_CACHE"):
        v = os.environ.get(key)
        if v:
            return str(Path(v).expanduser().resolve())
    return None


class ClipBackend:
    """
    open_clip wrapper with:
    - FP16 inference on CUDA
    - torch.compile (PyTorch ≥ 2.0) for speed
    - Correct per-model preprocessing (size, mean, std)

    For Hub weights (e.g. ``laion2b_s34b_b79k``), set ``HF_HOME`` to the directory
    **above** ``hub/`` (or ``HF_HUB_CACHE`` = ``…/huggingface_hub/hub``). Do not rely
    on ``OPENCLIP_CACHE`` for those — it is a different cache layout.
    """

    def __init__(self, model_name: str, pretrained: str, device: torch.device):
        _ensure_hf_home_for_clip_cache()
        import open_clip

        self.model_name = model_name
        self.pretrained = pretrained
        self.device     = device

        _cache = _open_clip_weight_cache_dir()
        model, _, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            precision="fp16" if device.type == "cuda" else "fp32",
            device=device,
            cache_dir=_cache,
        )
        model.eval()

        if hasattr(torch, "compile") and device.type == "cuda":
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception:
                pass

        self.model         = model
        self.tokenizer     = open_clip.get_tokenizer(model_name)
        self.preprocess    = preprocess_val   # torchvision.Compose; handles PIL images
        self._img_size, self._mean, self._std = _extract_transform_params(preprocess_val)

        # Probe embed dimension
        with torch.no_grad():
            _dummy = torch.zeros(
                1, 3, self._img_size, self._img_size,
                device=device,
                dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )
            self._dim: int = int(model.encode_image(_dummy).shape[-1])

    def embed_dim(self) -> int:
        return self._dim

    @torch.inference_mode()
    def embed_crops(self, crops: list[np.ndarray]) -> torch.Tensor:
        """
        Batch-embed a list of numpy uint8 RGB crops → (N, D) L2-normalised float32.
        Uses fast tensor path (no PIL round-trip) with correct per-model normalisation.
        """
        if not crops:
            return torch.zeros(0, self._dim, device=self.device)

        sz = self._img_size
        mean_t = torch.tensor(self._mean, device=self.device).view(1, 3, 1, 1)
        std_t  = torch.tensor(self._std,  device=self.device).view(1, 3, 1, 1)

        tensors = []
        for crop in crops:
            resized = cv2.resize(crop, (sz, sz))
            t = torch.from_numpy(resized.astype(np.float32)).permute(2, 0, 1) / 255.0
            tensors.append(t)

        batch = torch.stack(tensors).to(self.device)  # (N, 3, sz, sz) float32
        batch = (batch - mean_t) / std_t
        if self.device.type == "cuda":
            batch = batch.half()

        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=self.device.type == "cuda"):
            emb = self.model.encode_image(batch)

        return F.normalize(emb.float(), dim=-1)

    @torch.inference_mode()
    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """(Q, D) L2-normalised float32."""
        tokens = self.tokenizer(texts).to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=self.device.type == "cuda"):
            emb = self.model.encode_text(tokens)
        return F.normalize(emb.float(), dim=-1)

    def unload(self) -> None:
        del self.model
        _flush_vram()


def _extract_transform_params(preprocess) -> tuple[int, list[float], list[float]]:
    """Extract (img_size, mean, std) from a torchvision.transforms.Compose."""
    import torchvision.transforms as T

    img_size = 224
    mean = [0.48145466, 0.4578275,  0.40821073]   # CLIP default
    std  = [0.26862954, 0.26130258, 0.27577711]

    for t in getattr(preprocess, "transforms", []):
        if isinstance(t, (T.Resize, T.CenterCrop)):
            s = getattr(t, "size", None)
            if s is not None:
                img_size = s if isinstance(s, int) else (s[0] if hasattr(s, "__getitem__") else s)
        elif isinstance(t, T.Normalize):
            mean = [float(v) for v in t.mean]
            std  = [float(v) for v in t.std]

    return int(img_size), mean, std


# ══════════════════════════════════════════════════════════════════════════════
# CROP UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _tight_crop(mask_dict: dict, image_rgb: np.ndarray) -> np.ndarray:
    """Tight bounding-box crop with background set to black (mask pixels only)."""
    seg = mask_dict["segmentation"].astype(bool)
    ys, xs = np.where(seg)
    if len(ys) == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = image_rgb[y0:y1, x0:x1].copy()
    crop[~seg[y0:y1, x0:x1]] = 0
    return _pad_square(crop)


def _pad_square(img: np.ndarray) -> np.ndarray:
    """Pad image to square with zeros (black)."""
    h, w = img.shape[:2]
    if h == w:
        return img
    s   = max(h, w)
    out = np.zeros((s, s, 3), dtype=img.dtype)
    dh, dw = (s - h) // 2, (s - w) // 2
    out[dh:dh + h, dw:dw + w] = img
    return out


def _mask_bbox_xywh(mask_dict: dict) -> tuple[int, int, int, int]:
    """``(x, y, w, h)`` for the mask; same convention as HF-normalised masks."""
    b = mask_dict.get("bbox")
    if b is not None and len(b) >= 4:
        return int(b[0]), int(b[1]), int(b[2]), int(b[3])
    seg = mask_dict["segmentation"].astype(bool)
    ys, xs = np.where(seg)
    if len(ys) == 0:
        return 0, 0, 1, 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1 - x0, y1 - y0


def _padded_box_crop(mask_dict: dict, image_rgb: np.ndarray, pad: int) -> np.ndarray:
    """
    Rectangular crop from bbox + ``pad`` px per side (clamped to image). Background
    is kept; then :func:`_pad_square` for CLIP. Used only for CLIP embeddings.
    """
    H, W = image_rgb.shape[:2]
    x, y, bw, bh = _mask_bbox_xywh(mask_dict)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + bw + pad)
    y1 = min(H, y + bh + pad)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    crop = image_rgb[y0:y1, x0:x1].copy()
    return _pad_square(crop)


def _clip_crop_for_embedding(
    mask_dict: dict,
    image_rgb: np.ndarray,
    *,
    use_boxes: bool,
    box_pad: int,
) -> np.ndarray:
    """Dispatch: tight masked crop (default) vs padded bbox crop for CLIP."""
    if use_boxes:
        return _padded_box_crop(mask_dict, image_rgb, pad=box_pad)
    return _tight_crop(mask_dict, image_rgb)


def _clip_crop_label(args) -> str:
    if getattr(args, "use_boxes", False):
        return f"bbox_pad_{int(getattr(args, 'box_pad', 20))}"
    return "tight"


def _sanitize_class_col(name: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", str(name).strip())
    return s or "class"


def _fill_per_class_nan_row(
    ordered_class_names: list[str],
    row: dict[str, Any],
) -> None:
    for cn in ordered_class_names:
        sk = _sanitize_class_col(cn)
        row[f"iou__{sk}"] = float("nan")
        row[f"precision_hit__{sk}"] = float("nan")


def _write_per_class_csv(path: Path, rows: list[dict[str, Any]], ordered_class_names: list[str]) -> None:
    """
    Wide CSV: fixed metadata columns + pairs ``iou__*`` / ``precision_hit__*`` per class.
    Overwrites ``path`` — call after each experiment so the file exists while a long
    sweep is still running (not only after the full run finishes).
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fixed = [
        "sample_id",
        "rgb",
        "sam_model",
        "clip_model",
        "clip_pretrained",
        "threshold",
        "mask_threshold",
        "clip_crop",
        "approach",
        "dbscan_eps",
        "hit_iou_thr",
    ]
    tail: list[str] = []
    for cn in ordered_class_names:
        sk = _sanitize_class_col(cn)
        tail.append(f"iou__{sk}")
        tail.append(f"precision_hit__{sk}")
    cols = fixed + tail
    df = pd.DataFrame(rows)
    df = df.reindex(columns=[c for c in cols if c in df.columns])
    df.to_csv(path, index=False)


def _flush_per_class_csv_if(
    pc_path: Path | None,
    rows: list[dict[str, Any]],
    ordered_class_names: list[str] | None,
) -> None:
    """Rewrite per-class CSV from accumulated rows (incremental progress)."""
    if pc_path is None or not rows or ordered_class_names is None:
        return
    _write_per_class_csv(Path(pc_path), rows, ordered_class_names)


# ══════════════════════════════════════════════════════════════════════════════
# DBSCAN ON PRECOMPUTED TIGHT-CROP EMBEDDINGS  (Approach 2)
# ══════════════════════════════════════════════════════════════════════════════

def _dbscan_merge_masks_from_embeddings(
    image_rgb: np.ndarray,
    masks:     list[dict],
    embs_np:   np.ndarray,
    dbscan_eps: float,
    device:    torch.device,
) -> tuple[torch.Tensor, list[dict]]:
    """
    DBSCAN (cosine distance) on per-mask embeddings, then OR-merge segmentations.
    ``embs_np`` rows align with ``masks`` (e.g. CLIP tight-crop, L2-normalised).
    """
    D = int(embs_np.shape[1]) if embs_np.size else 512
    if not masks:
        return torch.zeros(0, D, device=device), []

    H, W = image_rgb.shape[:2]
    N = len(masks)
    if embs_np.shape[0] != N:
        raise ValueError("embs_np rows must match masks length")

    if N == 1:
        m = masks[0]
        emb = torch.from_numpy(embs_np[0].astype(np.float32)).to(device)
        seg = m["segmentation"].astype(bool)
        ys, xs = np.where(seg)
        bbox = (
            [int(xs.min()), int(ys.min()),
             int(xs.max() - xs.min() + 1),
             int(ys.max() - ys.min() + 1)]
            if len(ys) > 0 else [0, 0, 0, 0]
        )
        return emb.unsqueeze(0), [{
            "segmentation":  seg,
            "area":          int(seg.sum()),
            "score":         float(m.get("score", 0.0)),
            "predicted_iou": float(m.get("predicted_iou", m.get("score", 0.0))),
            "bbox":          bbox,
            "n_merged":      1,
        }]

    if len(embs_np) > 1:
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.metrics.pairwise import cosine_distances

            dists  = cosine_distances(embs_np).astype(np.float64)
            labels = DBSCAN(
                eps=dbscan_eps, min_samples=1, metric="precomputed", n_jobs=-1
            ).fit_predict(dists)
        except ImportError:
            labels = np.arange(N, dtype=np.int64)
    else:
        labels = np.zeros(N, dtype=np.int64)

    unique_labels = sorted(set(labels.tolist()))
    cluster_embs_list: list[np.ndarray] = []
    cluster_masks_out: list[dict]       = []

    for lbl in unique_labels:
        member_idxs = np.where(labels == lbl)[0]

        merged_emb = embs_np[member_idxs].mean(0)
        norm = float(np.linalg.norm(merged_emb))
        if norm > 1e-8:
            merged_emb /= norm
        cluster_embs_list.append(merged_emb)

        union_seg  = np.zeros((H, W), dtype=bool)
        best_score = 0.0
        for i in member_idxs:
            union_seg |= masks[int(i)]["segmentation"].astype(bool)
            best_score = max(
                best_score,
                float(masks[int(i)].get("predicted_iou", masks[int(i)].get("score", 0.0))),
            )

        area   = int(union_seg.sum())
        ys, xs = np.where(union_seg)
        bbox   = (
            [int(xs.min()), int(ys.min()),
             int(xs.max() - xs.min() + 1),
             int(ys.max() - ys.min() + 1)]
            if len(ys) > 0 else [0, 0, 0, 0]
        )
        cluster_masks_out.append({
            "segmentation":  union_seg,
            "area":          area,
            "score":         best_score,
            "predicted_iou": best_score,
            "bbox":          bbox,
            "n_merged":      int(len(member_idxs)),
        })

    cluster_emb_tensor = torch.from_numpy(
        np.stack(cluster_embs_list).astype(np.float32)
    ).to(device)

    return cluster_emb_tensor, cluster_masks_out


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union) if union > 0 else float("nan")


def _average_precision(
    scores:    np.ndarray,          # (N,) higher = more confident
    segs:      list[np.ndarray],    # list of bool masks
    gt:        np.ndarray,          # bool GT mask
    iou_thr:   float = 0.5,
) -> float:
    """Mean Average Precision for single-class mask retrieval."""
    order = np.argsort(-scores)
    n_pos, prec_sum = 0, 0.0
    for rank, idx in enumerate(order, 1):
        if _binary_iou(segs[idx], gt) >= iou_thr:
            n_pos  += 1
            prec_sum += n_pos / rank
    return (prec_sum / n_pos) if n_pos > 0 else 0.0


def _fragmentation_heuristic(masks: list[dict]) -> float:
    """
    Heuristic fragmentation rate (no GT needed).
    A mask is flagged as 'fragmented' when:
        predicted_iou < 0.7  OR  area/bbox_area < 0.3 (very thin/patchy region).
    """
    if not masks:
        return float("nan")
    n_frag = 0
    for m in masks:
        piou     = float(m.get("predicted_iou", m.get("score", 1.0)))
        area     = int(m.get("area", 1))
        _, _, bw, bh = m.get("bbox", [1, 1, 1, 1])
        fill     = area / max(bw * bh, 1)
        if piou < 0.7 or fill < 0.3:
            n_frag += 1
    return n_frag / len(masks)


def _nanmean(lst: list) -> float:
    vals = [x for x in lst if isinstance(x, float) and not np.isnan(x)]
    return float(np.mean(vals)) if vals else float("nan")


def _max_iou_over_preds(segs: list[np.ndarray], gt: np.ndarray) -> float:
    """Return max IoU across predicted segs against one GT mask."""
    if not segs:
        return 0.0
    if not gt.any():
        return float("nan")
    best = 0.0
    for s in segs:
        v = _binary_iou(s, gt)
        if np.isnan(v):
            continue
        if v > best:
            best = float(v)
    return best


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING HELPERS  (manifest / queries / GT)
# ══════════════════════════════════════════════════════════════════════════════

def _resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (_PROJ / path).resolve()


def _resolve_rel(manifest_path: Path, p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    cand = (manifest_path.parent / path).resolve()
    if cand.exists():
        return cand
    return (_PROJ / path).resolve()


def _load_name_to_id(info_path: Path) -> dict[str, int]:
    data = json.loads(info_path.read_text(encoding="utf-8"))
    m: dict[str, int] = {}
    for c in data["classes"]:
        name = c["name"].strip().lower()
        m[name] = int(c["id"])
        m[name.replace("-", " ")] = int(c["id"])
    return m


def _load_id_to_name(info_path: Path) -> dict[int, str]:
    data = json.loads(info_path.read_text(encoding="utf-8"))
    return {int(c["id"]): c["name"] for c in data["classes"]}


def _resolve_class_id(query: str, name_to_id: dict[str, int]) -> int:
    q = query.strip().lower()
    for prefix in ("a ", "the ", "an "):
        if q.startswith(prefix):
            q = q[len(prefix) :]
    q_hyp = q.replace(" ", "-")
    if q in name_to_id:
        return name_to_id[q]
    if q_hyp in name_to_id:
        return name_to_id[q_hyp]
    raise ValueError(f"Class not found in info_semantic: {query!r}")


def _classes_from_semantic(sem: np.ndarray, id_to_name: dict[int, str]) -> list[str]:
    return [
        id_to_name[int(uid)]
        for uid in np.unique(sem.astype(np.int64))
        if uid > 0 and int(uid) in id_to_name
    ]


def _global_union_class_names_from_manifest(
    samples: list[dict],
    man_path: Path,
    id_to_name: dict[int, str],
) -> list[str]:
    """
    All semantic class names that appear in at least one frame (union over samples).
    Sorted by class id for stable ordering.
    """
    cids: set[int] = set()
    for sample in samples:
        sem_path = _resolve_rel(man_path, sample["semantic"])
        if not sem_path.is_file():
            continue
        try:
            sem = np.load(str(sem_path))
        except Exception:
            continue
        for uid in np.unique(sem.astype(np.int64)):
            u = int(uid)
            if u > 0 and u in id_to_name:
                cids.add(u)
    return [id_to_name[cid] for cid in sorted(cids)]


def _parse_queries_entry(entry: Any) -> tuple[list[str], dict[str, str]]:
    if isinstance(entry, list):
        return [str(x).strip() for x in entry], {}
    if isinstance(entry, dict):
        classes = [str(x).strip() for x in entry.get("classes", [])]
        qmap    = {k.strip().lower(): str(v) for k, v in entry.get("queries", {}).items()}
        return classes, qmap
    raise ValueError(f"Bad queries entry: {entry!r}")


def _load_queries_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "by_sample_id" in data:
        return data["by_sample_id"]
    if "samples" in data:
        out: dict[str, Any] = {}
        for row in data["samples"]:
            sid = str(row["id"])
            out[sid] = row.get("classes", row)
        return out
    raise ValueError("queries JSON: expected keys 'by_sample_id' or 'samples'")


def _build_sample_queries(
    sample:        dict,
    sem:           np.ndarray,
    id_to_name:    dict[int, str],
    name_to_id:    dict[str, int],
    queries_by_id: dict[str, Any],
    args,
    global_gt_class_names: list[str] | None = None,
) -> tuple[list[str], list[int]] | None:
    """
    Return (texts, class_ids) for a sample, or None if the sample should be skipped.

    In sweep mode with ``--queries_from_gt``, pass ``global_gt_class_names`` (union
    over the manifest) so every frame uses the same query set.
    """
    sid = str(sample["id"])

    if args.class_name is not None and not args.queries_from_gt and args.queries_json is None:
        # Single-class mode
        try:
            cid  = _resolve_class_id(args.class_name, name_to_id)
            text = args.text_query or f"a {args.class_name.strip()}"
            return [text], [cid]
        except ValueError:
            return None

    if args.queries_from_gt:
        if global_gt_class_names is not None:
            class_names = list(global_gt_class_names)
        else:
            class_names = _classes_from_semantic(sem, id_to_name)
        query_overrides: dict[str, str] = {}
    else:
        if sid not in queries_by_id:
            return None
        class_names, query_overrides = _parse_queries_entry(queries_by_id[sid])

    if not class_names:
        return None

    texts: list[str] = []
    cids:  list[int] = []
    for cn in class_names:
        cn_l = cn.strip().lower()
        qtxt = query_overrides.get(cn_l) or args.text_template.format(class_name=cn)
        try:
            cid = _resolve_class_id(cn, name_to_id)
        except ValueError:
            continue
        texts.append(qtxt)
        cids.append(cid)

    return (texts, cids) if texts else None


# ══════════════════════════════════════════════════════════════════════════════
# SAM MASK CACHE  (run SAM once per sample per model, then unload)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _SampleData:
    sample:    dict
    image_rgb: np.ndarray
    sem:       np.ndarray
    masks:     list[dict]       # SAM output at min_score=0.10
    texts:     list[str]
    class_ids: list[int]


def _resolve_ci_for_class_name(name: str, sd: _SampleData, id_to_name: dict[int, str]) -> int | None:
    """Index into ``sd.class_ids`` / ``best_indices`` for this class label, or None."""
    for ci, cid in enumerate(sd.class_ids):
        if id_to_name.get(cid) == name:
            return ci
    return None


def _ordered_union_class_names(
    cache: list[_SampleData],
    global_gt_class_names: list[str] | None,
    id_to_name: dict[int, str],
) -> list[str]:
    """Column order for wide per-class CSV: global GT union, else sorted union from cache."""
    if global_gt_class_names:
        return list(global_gt_class_names)
    s: set[str] = set()
    for sd in cache:
        for cid in sd.class_ids:
            s.add(id_to_name.get(cid, str(cid)))
    return sorted(s)


def _base_per_class_row(
    sd: _SampleData,
    sam_model_id: str,
    clip_model_name: str,
    clip_pretrained: str,
    conf_thresh: float,
    mask_thresh: float,
    clip_crop: str,
    approach: str,
    dbscan_eps: float | None,
    hit_thr: float,
) -> dict[str, Any]:
    return {
        "sample_id": str(sd.sample["id"]),
        "rgb": str(sd.sample.get("rgb", "")),
        "sam_model": sam_model_id,
        "clip_model": clip_model_name,
        "clip_pretrained": clip_pretrained,
        "threshold": conf_thresh,
        "mask_threshold": mask_thresh,
        "clip_crop": clip_crop,
        "approach": approach,
        "dbscan_eps": "" if dbscan_eps is None else float(dbscan_eps),
        "hit_iou_thr": float(hit_thr),
    }


def _cache_sam_masks(
    sam:           SamHFBackend | Sam1Backend,
    samples:       list[dict],
    man_path:      Path,
    name_to_id:    dict[str, int],
    id_to_name:    dict[int, str],
    queries_by_id: dict[str, Any],
    args,
    global_gt_class_names: list[str] | None = None,
    verbose:       bool = True,
) -> list[_SampleData]:
    """Run SAM on all samples, return cached _SampleData list."""
    cache: list[_SampleData] = []
    gen_s: list[float] = []  # wall seconds per image for sam.generate only
    n = len(samples)
    for i, sample in enumerate(samples):
        rgb_path = _resolve_rel(man_path, sample["rgb"])
        sem_path = _resolve_rel(man_path, sample["semantic"])

        bgr = cv2.imread(str(rgb_path))
        if bgr is None:
            if verbose:
                print(f"    [skip] cannot read {rgb_path.name}")
            continue
        sem = np.load(str(sem_path))
        if sem.shape[:2] != bgr.shape[:2]:
            if verbose:
                print(f"    [skip] shape mismatch {rgb_path.name}")
            continue

        qres = _build_sample_queries(
            sample,
            sem,
            id_to_name,
            name_to_id,
            queries_by_id,
            args,
            global_gt_class_names=global_gt_class_names,
        )
        if qres is None:
            continue

        texts, class_ids = qres
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        t_gen0 = time.perf_counter()
        masks = sam.generate(image_rgb)
        gen_s.append(time.perf_counter() - t_gen0)

        if verbose and (i % max(1, n // 10) == 0 or i == n - 1):
            print(f"    sam cache  [{i+1:3d}/{n}]  {rgb_path.name}  "
                  f"masks={len(masks)}")

        if masks:
            masks.sort(key=lambda x: -x["area"])
        cache.append(_SampleData(
            sample=sample,
            image_rgb=image_rgb,
            sem=sem,
            masks=masks,
            texts=texts,
            class_ids=class_ids,
        ))

    if verbose and gen_s:
        tot = sum(gen_s)
        m = tot / len(gen_s)
        print(
            f"    SAM generate timing: {len(gen_s)} images, {tot:.1f}s total "
            f"({m:.2f}s/img mean, {min(gen_s):.2f}–{max(gen_s):.2f}s min–max) — "
            f"mask count varies; large N → slower postprocess (NMS)."
        )

    return cache


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION  — threshold grid-search (Approach 1)
# ══════════════════════════════════════════════════════════════════════════════

def _precompute_threshold_grid_clip(
    cache: list[_SampleData],
    clip:  ClipBackend,
    args,
) -> list[dict[str, Any] | None]:
    """
    One CLIP vision + text forward per sample (all SAM masks at cache resolution).
    Threshold pairs only subset rows — no re-embedding.
    """
    use_boxes = bool(getattr(args, "use_boxes", False))
    box_pad   = int(getattr(args, "box_pad", 20))
    out: list[dict[str, Any] | None] = []
    for sd in cache:
        if not sd.masks:
            out.append(None)
            continue
        img_emb = clip.embed_crops([
            _clip_crop_for_embedding(
                m, sd.image_rgb, use_boxes=use_boxes, box_pad=box_pad
            )
            for m in sd.masks
        ])
        txt_emb = clip.embed_texts(sd.texts)
        out.append({
            "sd": sd,
            "img_emb": img_emb.detach().cpu(),
            "txt_emb": txt_emb.detach().cpu(),
        })
    return out


def eval_threshold_precomputed(
    precomputed:  list[dict[str, Any] | None],
    conf_thresh:  float,
    mask_thresh:  float,
    sam_model_id: str,
    args,
    device:       torch.device,
    clip_model_name: str,
    clip_pretrained: str,
    *,
    id_to_name: dict[int, str] | None = None,
    ordered_class_names: list[str] | None = None,
    per_class_sink: list[dict[str, Any]] | None = None,
) -> RunRecord:
    """
    Same metrics as the former per-threshold ``embed_crops`` loop, but uses
    :func:`_precompute_threshold_grid_clip` tensors. ``avg_speed_ms`` is filter +
    matmul + metric time per frame (embeddings excluded).

    Optional wide per-class CSV: pass ``per_class_sink`` + ``ordered_class_names``
    + ``id_to_name``; appends one row per (sample × this experiment) with
    ``iou__*`` / ``precision_hit__*`` per class (NaN if class absent on image or
    not in this frame's query list, or if no masks pass the threshold).
    """
    rec = RunRecord(
        sam_model=sam_model_id,
        clip_model=clip_model_name,
        clip_pretrained=clip_pretrained,
        threshold=conf_thresh,
        mask_threshold=mask_thresh,
        post_processing=f"ThresholdGrid(conf={conf_thresh},mask={mask_thresh})",
        clip_crop=_clip_crop_label(args),
    )

    ious, clip_scores, aps, sa_hits = [], [], [], []
    obj_hits_any: list[bool] = []
    obj_hits_chosen: list[bool] = []
    masks_per_img, frag_per_img, speed_ms = [], [], []

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    hit_thr = float(getattr(args, "hit_iou_thr", 0.1))
    prec_hit_on = bool(getattr(args, "enable_precision_hit", True))
    clip_crop_s = _clip_crop_label(args)
    want_pc = (
        per_class_sink is not None
        and ordered_class_names is not None
        and id_to_name is not None
    )

    for item in precomputed:
        if item is None:
            continue
        sd: _SampleData = item["sd"]
        img_full: torch.Tensor = item["img_emb"]   # (N, D) CPU
        txt_full: torch.Tensor = item["txt_emb"]   # (Q, D) CPU

        keep = [
            i for i, m in enumerate(sd.masks)
            if m.get("predicted_iou", m.get("score", 1.0)) >= conf_thresh
            and m.get("score", 1.0) >= mask_thresh
        ]
        if not keep:
            if want_pc:
                row = _base_per_class_row(
                    sd,
                    sam_model_id,
                    clip_model_name,
                    clip_pretrained,
                    conf_thresh,
                    mask_thresh,
                    clip_crop_s,
                    "threshold_grid",
                    None,
                    hit_thr,
                )
                _fill_per_class_nan_row(ordered_class_names, row)
                per_class_sink.append(row)
            continue

        t0 = time.perf_counter()
        img_emb = img_full[keep].to(device, non_blocking=True)
        txt_emb = txt_full.to(device, non_blocking=True)
        sim     = torch.matmul(img_emb, txt_emb.T)
        t1 = time.perf_counter()
        speed_ms.append((t1 - t0) * 1000.0)

        best_indices = sim.argmax(dim=0).cpu().numpy()

        masks_f = [sd.masks[i] for i in keep]
        masks_per_img.append(len(masks_f))
        frag_per_img.append(_fragmentation_heuristic(masks_f))
        segs = [m["segmentation"].astype(bool) for m in masks_f]

        if want_pc:
            row = _base_per_class_row(
                sd,
                sam_model_id,
                clip_model_name,
                clip_pretrained,
                conf_thresh,
                mask_thresh,
                clip_crop_s,
                "threshold_grid",
                None,
                hit_thr,
            )
            assert ordered_class_names is not None and id_to_name is not None
            for cn in ordered_class_names:
                sk = _sanitize_class_col(cn)
                ci = _resolve_ci_for_class_name(cn, sd, id_to_name)
                if ci is None:
                    row[f"iou__{sk}"] = float("nan")
                    row[f"precision_hit__{sk}"] = float("nan")
                    continue
                cid = sd.class_ids[ci]
                gt = (sd.sem.astype(np.int64) == cid)
                if not gt.any():
                    row[f"iou__{sk}"] = float("nan")
                    row[f"precision_hit__{sk}"] = float("nan")
                    continue
                best_i = int(best_indices[ci])
                pred = segs[best_i]
                iou = _binary_iou(pred, gt)
                row[f"iou__{sk}"] = float(iou)
                if prec_hit_on:
                    row[f"precision_hit__{sk}"] = (
                        1.0 if (not np.isnan(iou) and iou >= hit_thr) else 0.0
                    )
                else:
                    row[f"precision_hit__{sk}"] = float("nan")
            per_class_sink.append(row)

        for ci, cid in enumerate(sd.class_ids):
            gt = (sd.sem.astype(np.int64) == cid)
            if not gt.any() and not args.include_empty_gt:
                continue
            best_i   = int(best_indices[ci])
            pred     = segs[best_i]
            iou      = _binary_iou(pred, gt) if gt.any() else (0.0 if pred.any() else float("nan"))
            cos_val  = float(sim[best_i, ci].item())
            ap       = _average_precision(sim[:, ci].cpu().numpy(), segs, gt)

            if not np.isnan(iou):
                ious.append(iou)
                sa_hits.append(iou >= 0.5)
            clip_scores.append(cos_val)
            aps.append(ap)

            # Extra hit metrics (object-level).
            if gt.any():
                best_iou_any = _max_iou_over_preds(segs, gt)
                obj_hits_any.append(best_iou_any >= hit_thr)
                if getattr(args, "enable_precision_hit", True):
                    obj_hits_chosen.append(iou >= hit_thr if not np.isnan(iou) else False)

    _fill_record(
        rec,
        ious,
        clip_scores,
        aps,
        sa_hits,
        obj_hits_any,
        obj_hits_chosen,
        masks_per_img,
        frag_per_img,
        speed_ms,
    )
    return rec


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION  — threshold grid + DBSCAN (Approach 2)
# ══════════════════════════════════════════════════════════════════════════════

def eval_threshold_dbscan_precomputed(
    precomputed:  list[dict[str, Any] | None],
    conf_thresh:  float,
    mask_thresh:  float,
    dbscan_eps:   float,
    sam_model_id: str,
    args,
    device:       torch.device,
    clip_model_name: str,
    clip_pretrained: str,
    *,
    id_to_name: dict[int, str] | None = None,
    ordered_class_names: list[str] | None = None,
    per_class_sink: list[dict[str, Any]] | None = None,
) -> RunRecord:
    """
    Same CLIP precompute as Approach 1; filter by (conf, mask), DBSCAN-merge,
    then argmax over clusters. ``avg_speed_ms`` = filter + DBSCAN + matmul (+metrics).
    """
    rec = RunRecord(
        sam_model=sam_model_id,
        clip_model=clip_model_name,
        clip_pretrained=clip_pretrained,
        threshold=conf_thresh,
        mask_threshold=mask_thresh,
        post_processing=(
            f"Grid+DBSCAN(conf={conf_thresh},mask={mask_thresh},eps={dbscan_eps})"
        ),
        clip_crop=_clip_crop_label(args),
    )

    ious, clip_scores, aps, sa_hits = [], [], [], []
    obj_hits_any: list[bool] = []
    obj_hits_chosen: list[bool] = []
    masks_per_img, frag_per_img, speed_ms = [], [], []

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    hit_thr = float(getattr(args, "hit_iou_thr", 0.1))
    prec_hit_on = bool(getattr(args, "enable_precision_hit", True))
    clip_crop_s = _clip_crop_label(args)
    want_pc = (
        per_class_sink is not None
        and ordered_class_names is not None
        and id_to_name is not None
    )

    for item in precomputed:
        if item is None:
            continue
        sd: _SampleData = item["sd"]
        img_full: torch.Tensor = item["img_emb"]
        txt_full: torch.Tensor = item["txt_emb"]

        keep = [
            i for i, m in enumerate(sd.masks)
            if m.get("predicted_iou", m.get("score", 1.0)) >= conf_thresh
            and m.get("score", 1.0) >= mask_thresh
        ]
        if not keep:
            if want_pc:
                row = _base_per_class_row(
                    sd,
                    sam_model_id,
                    clip_model_name,
                    clip_pretrained,
                    conf_thresh,
                    mask_thresh,
                    clip_crop_s,
                    "threshold_grid_dbscan",
                    float(dbscan_eps),
                    hit_thr,
                )
                _fill_per_class_nan_row(ordered_class_names, row)
                per_class_sink.append(row)
            continue

        embs_np = img_full[keep].numpy().astype(np.float64)
        masks_f = [sd.masks[i] for i in keep]

        t0 = time.perf_counter()
        cluster_embs, cluster_masks = _dbscan_merge_masks_from_embeddings(
            sd.image_rgb, masks_f, embs_np, dbscan_eps, device,
        )
        txt_emb = txt_full.to(device, non_blocking=True)
        sim     = torch.matmul(cluster_embs, txt_emb.T)
        t1 = time.perf_counter()
        speed_ms.append((t1 - t0) * 1000.0)

        if len(cluster_embs) == 0:
            if want_pc:
                row = _base_per_class_row(
                    sd,
                    sam_model_id,
                    clip_model_name,
                    clip_pretrained,
                    conf_thresh,
                    mask_thresh,
                    clip_crop_s,
                    "threshold_grid_dbscan",
                    float(dbscan_eps),
                    hit_thr,
                )
                _fill_per_class_nan_row(ordered_class_names, row)
                per_class_sink.append(row)
            continue

        best_indices = sim.argmax(dim=0).cpu().numpy()

        masks_per_img.append(len(cluster_masks))
        frag_per_img.append(_fragmentation_heuristic(cluster_masks))
        segs = [m["segmentation"].astype(bool) for m in cluster_masks]

        if want_pc:
            row = _base_per_class_row(
                sd,
                sam_model_id,
                clip_model_name,
                clip_pretrained,
                conf_thresh,
                mask_thresh,
                clip_crop_s,
                "threshold_grid_dbscan",
                float(dbscan_eps),
                hit_thr,
            )
            assert ordered_class_names is not None and id_to_name is not None
            for cn in ordered_class_names:
                sk = _sanitize_class_col(cn)
                ci = _resolve_ci_for_class_name(cn, sd, id_to_name)
                if ci is None:
                    row[f"iou__{sk}"] = float("nan")
                    row[f"precision_hit__{sk}"] = float("nan")
                    continue
                cid = sd.class_ids[ci]
                gt = (sd.sem.astype(np.int64) == cid)
                if not gt.any():
                    row[f"iou__{sk}"] = float("nan")
                    row[f"precision_hit__{sk}"] = float("nan")
                    continue
                best_i = int(best_indices[ci])
                pred = segs[best_i]
                iou = _binary_iou(pred, gt)
                row[f"iou__{sk}"] = float(iou)
                if prec_hit_on:
                    row[f"precision_hit__{sk}"] = (
                        1.0 if (not np.isnan(iou) and iou >= hit_thr) else 0.0
                    )
                else:
                    row[f"precision_hit__{sk}"] = float("nan")
            per_class_sink.append(row)

        inc_empty = getattr(args, "include_empty_gt", False)
        for ci, cid in enumerate(sd.class_ids):
            gt = (sd.sem.astype(np.int64) == cid)
            if not gt.any() and not inc_empty:
                continue
            best_i  = int(best_indices[ci])
            pred    = segs[best_i]
            iou     = _binary_iou(pred, gt) if gt.any() else (0.0 if pred.any() else float("nan"))
            cos_val = float(sim[best_i, ci].item())
            ap      = _average_precision(sim[:, ci].cpu().numpy(), segs, gt)

            if not np.isnan(iou):
                ious.append(iou)
                sa_hits.append(iou >= 0.5)
            clip_scores.append(cos_val)
            aps.append(ap)

            if gt.any():
                best_iou_any = _max_iou_over_preds(segs, gt)
                obj_hits_any.append(best_iou_any >= hit_thr)
                if getattr(args, "enable_precision_hit", True):
                    obj_hits_chosen.append(iou >= hit_thr if not np.isnan(iou) else False)

    _fill_record(
        rec,
        ious,
        clip_scores,
        aps,
        sa_hits,
        obj_hits_any,
        obj_hits_chosen,
        masks_per_img,
        frag_per_img,
        speed_ms,
    )
    return rec


def _fill_record(
    rec:          RunRecord,
    ious:         list[float],
    clip_scores:  list[float],
    aps:          list[float],
    sa_hits:      list[bool],
    obj_hits_any: list[bool],
    obj_hits_chosen: list[bool],
    masks_per_img: list[int],
    frag_per_img: list[float],
    speed_ms:     list[float],
) -> None:
    vram_b = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    rsv_b  = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0

    rec.miou                    = _nanmean(ious)
    rec.sa_co_f1                = (sum(sa_hits) / len(sa_hits)) if sa_hits else float("nan")
    rec.recall_obj              = (sum(obj_hits_any) / len(obj_hits_any)) if obj_hits_any else float("nan")
    rec.precision_hit           = (
        (sum(obj_hits_chosen) / len(obj_hits_chosen)) if obj_hits_chosen else float("nan")
    )
    rec.avg_masks_per_image     = _nanmean([float(x) for x in masks_per_img])
    rec.percent_fragmented_masks = _nanmean(frag_per_img)
    rec.avg_clip_score_selected = _nanmean(clip_scores)
    rec.mAP_retrieval           = _nanmean(aps)
    rec.avg_speed_ms_per_frame  = _nanmean(speed_ms)
    rec.vram_gb_per_frame       = vram_b / (1024 ** 3)
    rec.vram_reserved_gb_per_frame = rsv_b / (1024 ** 3)


# ══════════════════════════════════════════════════════════════════════════════
# SWEEP ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def _parse_clip_cfgs(clip_args: list[str] | None) -> list[tuple[str, str]]:
    if clip_args is None:
        return clip_configs_for_eval()
    out = []
    for s in clip_args:
        parts = s.split("/", 1)
        out.append((parts[0], parts[1]) if len(parts) == 2 else (parts[0], "openai"))
    return out


def _flush_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _print_run(label: str, rec: RunRecord) -> None:
    print(
        f"      {label:<45s}  "
        f"mIoU={rec.miou:.3f}  F1={rec.sa_co_f1:.3f}  "
        f"Robj={rec.recall_obj:.3f}  Phit={rec.precision_hit:.3f}  "
        f"masks/img={rec.avg_masks_per_image:.1f}  "
        f"frag={rec.percent_fragmented_masks*100:.1f}%  "
        f"clip={rec.avg_clip_score_selected:.3f}  "
        f"mAP={rec.mAP_retrieval:.3f}  "
        f"{rec.avg_speed_ms_per_frame:.0f}ms  "
        f"alloc={rec.vram_gb_per_frame:.2f}GB "
        f"rsvd={rec.vram_reserved_gb_per_frame:.2f}GB"
    )


def _sink_record_csv(out_path: Path, rec: RunRecord, row_count: list[int]) -> None:
    """Append one CSV row; header written on first row only."""
    df = pd.DataFrame([asdict(rec)])
    n = row_count[0]
    df.to_csv(out_path, mode="w" if n == 0 else "a", index=False, header=(n == 0))
    row_count[0] = n + 1


def run_sweep(
    args,
    samples:       list[dict],
    man_path:      Path,
    name_to_id:    dict[str, int],
    id_to_name:    dict[int, str],
    queries_by_id: dict[str, Any],
) -> list[RunRecord]:

    sam_ids   = args.sam_models if args.sam_models else SAM_MODELS_DEFAULT
    clip_cfgs = _parse_clip_cfgs(args.clip_models)
    approaches = set(args.approaches or ["threshold_grid", "threshold_grid_dbscan"])

    csv_path: Path | None = None
    csv_rows: list[int] = [0]
    oc = getattr(args, "out_csv", None)
    if oc:
        csv_path = Path(oc)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("", encoding="utf-8")

    # Threshold combinations (Approach 1)
    if args.full_grid:
        thresh_combos = list(product(_CONF_VALS, _MASK_VALS))
    else:
        thresh_combos = _THRESH_COMBOS_9

    # Approach 2 (DBSCAN): only the 3 representative pairs — do **not** use
    # ``thresh_combos`` here when ``--full_grid`` is set (would be 25×|eps|).
    # Previously: ``for conf_t, mask_t in thresh_combos`` with full grid.
    thresh_combos_dbscan = _THRESH_COMBOS_9

    # W&B (optional)
    wandb_run = None
    if getattr(args, "wandb_project", None):
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, config=vars(args))
        except ImportError:
            print("  [warn] wandb not installed, skipping W&B logging")

    records: list[RunRecord] = []
    device  = torch.device(args.device if torch.cuda.is_available() else "cpu")

    pc_path = getattr(args, "_per_class_csv_path", None)
    per_class_all_rows: list[dict[str, Any]] = []
    ordered_for_csv: list[str] | None = None

    global_gt_class_names: list[str] | None = None
    if getattr(args, "queries_from_gt", False):
        global_gt_class_names = _global_union_class_names_from_manifest(
            samples, man_path, id_to_name
        )
        print(
            f"  GT class union: {len(global_gt_class_names)} unique classes "
            f"across {len(samples)} images (same queries every frame)."
        )
        if not global_gt_class_names:
            print("  [error] no semantic class ids found in manifest; abort sweep.")
            return []

    for sam_id in sam_ids:
        sam_vram = _sam_vram_estimate(sam_id)
        if sam_vram > args.vram_limit_gb:
            print(f"\n[SKIP] {sam_id}: estimated {sam_vram:.1f} GB > limit {args.vram_limit_gb} GB")
            continue

        print(f"\n{'═'*70}")
        print(f"SAM model : {sam_id}")
        print(f"{'═'*70}")

        # ── 1. Load SAM and cache masks ───────────────────────────────────────
        try:
            sam = load_sam_backend(
                sam_id, device, sam_ckpt=getattr(args, "sam_ckpt", None), half=True
            )
        except Exception as e:
            print(f"  [ERROR] cannot load {sam_id}:\n{e}")
            continue

        sam_label = sam.model_id  # e.g. sam1:sam_vit_b_01ec64.pth or facebook/sam2-…

        print(f"  Caching SAM masks for {len(samples)} samples …")
        t_sam0 = time.perf_counter()
        cache  = _cache_sam_masks(
            sam,
            samples,
            man_path,
            name_to_id,
            id_to_name,
            queries_by_id,
            args,
            global_gt_class_names=global_gt_class_names,
        )
        t_sam1 = time.perf_counter()
        sam.unload()
        _flush_vram()
        print(f"  SAM caching done: {len(cache)} valid samples "
              f"({t_sam1-t_sam0:.1f}s) – SAM unloaded.")

        if not cache:
            print("  [skip] no valid samples for this SAM model")
            continue

        if ordered_for_csv is None:
            ordered_for_csv = _ordered_union_class_names(
                cache, global_gt_class_names, id_to_name
            )

        # ── 2. For each CLIP model: embed + evaluate ──────────────────────────
        for clip_name, clip_pretrained in clip_cfgs:
            clip_vram = _CLIP_VRAM.get(clip_name, 1.0)
            if clip_vram > args.vram_limit_gb:
                print(f"\n  [SKIP] {clip_name}: {clip_vram:.1f} GB > limit")
                continue

            print(f"\n  CLIP: {clip_name} / {clip_pretrained}")

            try:
                clip = ClipBackend(clip_name, clip_pretrained, device)
            except Exception as e:
                print(f"    [ERROR] cannot load CLIP {clip_name}/{clip_pretrained}: {e}")
                continue

            need_clip_pre = (
                "threshold_grid" in approaches
                or "threshold_grid_dbscan" in approaches
            )
            pre_th: list[dict[str, Any] | None] | None = None
            if need_clip_pre:
                t_pre0 = time.perf_counter()
                pre_th = _precompute_threshold_grid_clip(cache, clip, args)
                t_pre1 = time.perf_counter()
                n_ok = sum(1 for x in pre_th if x is not None)
                _cm = (
                    f"bbox+{int(getattr(args, 'box_pad', 20))}px pad"
                    if getattr(args, "use_boxes", False)
                    else "tight mask crops"
                )
                print(
                    f"    CLIP precompute ({_cm}, all cached masks, once): "
                    f"{t_pre1 - t_pre0:.1f}s  ({n_ok}/{len(cache)} samples)"
                )

            pc_kw: dict[str, Any] = {}
            if pc_path is not None and ordered_for_csv is not None:
                pc_kw = {
                    "id_to_name": id_to_name,
                    "ordered_class_names": ordered_for_csv,
                    "per_class_sink": per_class_all_rows,
                }

            # ── Approach 1: plain threshold grid-search ─────────────────────
            if "threshold_grid" in approaches:
                assert pre_th is not None
                print(
                    f"    ── Approach 1: threshold grid ({len(thresh_combos)} combos) ──"
                )
                for conf_t, mask_t in thresh_combos:
                    rec = eval_threshold_precomputed(
                        pre_th,
                        conf_t,
                        mask_t,
                        sam_label,
                        args,
                        device,
                        clip.model_name,
                        clip.pretrained,
                        **pc_kw,
                    )
                    _print_run(rec.post_processing, rec)
                    records.append(rec)
                    if csv_path is not None:
                        _sink_record_csv(csv_path, rec, csv_rows)
                    if wandb_run:
                        wandb_run.log(asdict(rec))
                    _flush_per_class_csv_if(pc_path, per_class_all_rows, ordered_for_csv)

            # ── Approach 2: 3 fixed threshold pairs + DBSCAN (eps grid) ───────
            if "threshold_grid_dbscan" in approaches:
                assert pre_th is not None
                eps_list = list(args.dbscan_eps_grid)
                n_db = len(eps_list)
                n_tdb = len(thresh_combos_dbscan)
                print(
                    f"    ── Approach 2: threshold grid + DBSCAN "
                    f"({n_tdb}×{n_db} = {n_tdb * n_db} combos; not --full_grid) ──"
                )
                for conf_t, mask_t in thresh_combos_dbscan:
                    for db_eps in eps_list:
                        rec = eval_threshold_dbscan_precomputed(
                            pre_th,
                            conf_t,
                            mask_t,
                            db_eps,
                            sam_label,
                            args,
                            device,
                            clip.model_name,
                            clip.pretrained,
                            **pc_kw,
                        )
                        _print_run(rec.post_processing, rec)
                        records.append(rec)
                        if csv_path is not None:
                            _sink_record_csv(csv_path, rec, csv_rows)
                        if wandb_run:
                            wandb_run.log(asdict(rec))
                        _flush_per_class_csv_if(pc_path, per_class_all_rows, ordered_for_csv)

            clip.unload()
            _flush_vram()

        del cache
        _flush_vram()

    if wandb_run:
        wandb_run.finish()

    if pc_path and ordered_for_csv and per_class_all_rows:
        _flush_per_class_csv_if(pc_path, per_class_all_rows, ordered_for_csv)
        print(
            f"\nPer-class CSV (final): {pc_path}  "
            f"({len(per_class_all_rows)} rows × {len(ordered_for_csv)} classes)"
        )

    return records


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def _print_results_table(records: list[RunRecord]) -> None:
    if not records:
        print("No records.")
        return

    df = pd.DataFrame([asdict(r) for r in records])
    df["sam_model"] = df["sam_model"].str.replace("facebook/", "", regex=False)
    df["clip_label"] = df["clip_model"] + "/" + df["clip_pretrained"].str[:16]

    display_cols = [
        "sam_model", "clip_label", "clip_crop", "threshold", "mask_threshold",
        "post_processing", "miou", "sa_co_f1", "recall_obj", "precision_hit",
        "avg_masks_per_image", "percent_fragmented_masks",
        "avg_clip_score_selected", "mAP_retrieval",
        "avg_speed_ms_per_frame", "vram_gb_per_frame", "vram_reserved_gb_per_frame",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    df_show = df[display_cols].copy()

    float_fmt = {
        "miou": ".3f", "sa_co_f1": ".3f", "avg_masks_per_image": ".1f",
        "recall_obj": ".3f", "precision_hit": ".3f",
        "percent_fragmented_masks": ".3f", "avg_clip_score_selected": ".3f",
        "mAP_retrieval": ".3f", "avg_speed_ms_per_frame": ".0f",
        "vram_gb_per_frame": ".2f", "vram_reserved_gb_per_frame": ".2f",
        "threshold": ".1f", "mask_threshold": ".1f",
    }
    for col, fmt in float_fmt.items():
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(
                lambda x: f"{x:{fmt}}" if isinstance(x, float) and not np.isnan(x) else "nan"
            )

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        tbl = Table(title="SAM × CLIP Systematic Evaluation", show_lines=True,
                    header_style="bold cyan")
        for col in df_show.columns:
            tbl.add_column(col, no_wrap=(col in ("post_processing",)))
        for _, row in df_show.iterrows():
            tbl.add_row(*[str(v) for v in row])
        console.print(tbl)
    except ImportError:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 220)
        pd.set_option("display.max_colwidth", 40)
        print(df_show.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY SINGLE-RUN MODE  (backward compatible with old eval_clip_sam_miou.py)
# ══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def run_legacy(args) -> None:
    """Single-model run; mirrors old eval_clip_sam_miou.py behaviour."""
    try:
        import sam_query_match as sqm
    except ImportError:
        raise SystemExit(
            "Legacy mode requires scripts/sam_query_match.py to be importable. "
            "Add New-Proj/scripts/ to PYTHONPATH or run from the New-Proj directory."
        )

    device   = torch.device(args.device if torch.cuda.is_available() else "cpu")
    man_path = _resolve(args.manifest)
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    info_path = _resolve_rel(
        man_path,
        manifest.get("info_semantic", "data/replica_v1/office_0/habitat/info_semantic.json"),
    )
    name_to_id = _load_name_to_id(info_path)
    id_to_name = _load_id_to_name(info_path)

    queries_by_id: dict[str, Any] = {}
    if args.queries_json is not None:
        queries_by_id = _load_queries_json(_resolve(args.queries_json))

    ae = sqm.load_ae(_resolve(args.encoder), device) if args.encoder else None
    clip_model, tok, preprocess = sqm.load_clip(args.clip_model, args.clip_pretrained, device)
    sam_ckpt = args.sam_ckpt or "ckpts/sam_vit_b_01ec64.pth"
    generator, _ = sqm.load_sam_generator(_resolve(sam_ckpt), device)

    ious: list[float] = []
    skipped = 0
    n_pairs = 0

    samples = manifest["samples"]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    single = args.class_name is not None and not args.queries_from_gt and args.queries_json is None
    if single:
        g_class_id = _resolve_class_id(args.class_name, name_to_id)
        g_text     = args.text_query or f"a {args.class_name.strip()}"

    for sample in samples:
        sid      = str(sample["id"])
        rgb_path = _resolve_rel(man_path, sample["rgb"])
        sem_path = _resolve_rel(man_path, sample["semantic"])
        bgr      = cv2.imread(str(rgb_path))
        if bgr is None:
            skipped += 1; continue
        sem = np.load(str(sem_path))
        if sem.shape[:2] != bgr.shape[:2]:
            skipped += 1; continue

        if single:
            texts_clip = [g_text]; class_ids = [g_class_id]
        elif args.queries_from_gt:
            cnames        = _classes_from_semantic(sem, id_to_name)
            texts_clip    = [args.text_template.format(class_name=cn) for cn in cnames]
            try:
                class_ids = [_resolve_class_id(cn, name_to_id) for cn in cnames]
            except ValueError:
                skipped += 1; continue
        else:
            if sid not in queries_by_id:
                skipped += 1; continue
            cnames, qmap  = _parse_queries_entry(queries_by_id[sid])
            texts_clip    = []
            class_ids     = []
            for cn in cnames:
                qtxt = qmap.get(cn.strip().lower()) or args.text_template.format(class_name=cn)
                try:
                    class_ids.append(_resolve_class_id(cn, name_to_id))
                    texts_clip.append(qtxt)
                except ValueError:
                    continue
        if not texts_clip:
            skipped += 1; continue

        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        masks_all = generator.generate(image_rgb)
        masks_all.sort(key=lambda r: -r["area"])
        if not masks_all:
            skipped += 1; continue

        img_emb = sqm.embed_masks_clip(image_rgb, masks_all, clip_model, preprocess, device)
        if ae is not None:
            img_emb = ae.encode(img_emb.float())
        tokens  = tok(texts_clip).to(device)
        q       = F.normalize(clip_model.encode_text(tokens), dim=-1)
        if ae is not None:
            q = ae.encode(q.float())
        sim     = torch.matmul(img_emb.float(), q.float().T)
        best_i  = sim.argmax(dim=0).cpu().numpy()

        for ci, cid in enumerate(class_ids):
            gt = (sem.astype(np.int64) == cid)
            if not gt.any() and not args.include_empty_gt:
                continue
            pred = masks_all[int(best_i[ci])]["segmentation"].astype(bool)
            iou  = _binary_iou(pred, gt) if gt.any() else (0.0 if pred.any() else float("nan"))
            n_pairs += 1
            if not np.isnan(iou):
                ious.append(iou)

    miou = float(np.nanmean(ious)) if ious else float("nan")
    mode = "single_class" if single else "multi_class"
    print(f"mode={mode}  pairs={len(ious)}  total_seen={n_pairs}  skipped={skipped}")
    print(f"mIoU: {miou:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Systematic SAM2/SAM3 × CLIP concept segmentation benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ─ Data ───────────────────────────────────────────────────────────────────
    ap.add_argument("--manifest",        type=Path, required=True,
                    help="Path to manifest.json")
    ap.add_argument("--queries_json",    type=Path, default=None,
                    help="JSON with class lists per sample_id")
    ap.add_argument("--queries_from_gt", action="store_true",
                    help="Derive class list from GT semantic map (auto-mode)")
    ap.add_argument("--class_name",      default=None,
                    help="Single class for all frames (legacy single-class mode)")
    ap.add_argument("--text_query",      default=None,
                    help="Override text prompt for --class_name mode")
    ap.add_argument("--text_template",   default="a {class_name}",
                    help="Template for text prompts (default: 'a {class_name}')")
    ap.add_argument("--include_empty_gt", action="store_true",
                    help="Include (image, class) pairs where GT has no pixels")
    ap.add_argument("--max_samples",     type=int, default=None,
                    help="Limit number of samples (useful for quick tests)")

    # ─ Hit metrics ────────────────────────────────────────────────────────────
    ap.add_argument(
        "--hit_iou_thr",
        type=float,
        default=0.10,
        help=(
            "IoU threshold for hit-based metrics: recall_obj and precision_hit. "
            "Default: 0.10 (counts partial overlaps as hits)."
        ),
    )
    ap.add_argument(
        "--enable_precision_hit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compute precision_hit (Approach 2 hit metric: chosen-mask hit rate). "
            "Disable to save time: --no-enable_precision_hit"
        ),
    )

    # ─ Run mode ───────────────────────────────────────────────────────────────
    ap.add_argument(
        "--mode",
        choices=["sweep", "legacy"],
        default="sweep",
        help=(
            "'sweep' (default): systematic SAM×CLIP grid (default SAM: SAM1); "
            "'legacy': single-model run, backward-compat with old eval_clip_sam_miou.py"
        ),
    )

    # ─ Sweep: SAM models ─────────────────────────────────────────────────────
    ap.add_argument(
        "--sam_models_root",
        type=Path,
        default=None,
        help=(
            "Folder with downloaded HF snapshots (one subdir per model, each with "
            f"config.json). Used by preset:hf_local*. Default: {default_sam_hf_snapshots_root()}"
        ),
    )
    ap.add_argument(
        "--sam_models",
        nargs="+",
        default=None,
        metavar="HF_OR_LOCAL_OR_PRESET",
        help=(
            "Default: local (SAM1). Checkpoints: --sam_ckpt or "
            f"{DEFAULT_MODEL_CKPTS_ROOT}/sam/sam1/sam_vit_b_01ec64.pth if present. "
            "HF Hub id or local snapshot path: facebook/sam2-hiera-tiny or "
            "/mnt/data/model-ckpts/sam/sam2-hiera-tiny (needs transformers>=4.56). "
            "Presets: preset:hf_local — all ``sam1/*.pth`` + HF dirs with config.json "
            "(SAM2/SAM3) under --sam_models_root; preset:hf_local_no_sam3 — same but "
            "skip HF folders whose name contains ``sam3``. "
            "Sweep: also pass at least one of --sam1 / --sam2 / --sam3 to include those families."
        ),
    )
    ap.add_argument(
        "--sam1",
        action="store_true",
        help="Sweep: include SAM1 (segment_anything: local / *.pth under sam1/).",
    )
    ap.add_argument(
        "--sam2",
        action="store_true",
        help="Sweep: include all SAM2 Hugging Face checkpoints (sam2_video, e.g. sam2-hiera-*).",
    )
    ap.add_argument(
        "--sam3",
        action="store_true",
        help="Sweep: include SAM3 / SAM3.1 (sam3_video, e.g. folders sam3, sam3.1).",
    )

    # ─ Sweep: CLIP models ─────────────────────────────────────────────────────
    ap.add_argument(
        "--clip_models",
        nargs="+",
        default=None,
        metavar="MODEL/PRETRAINED",
        help=(
            "CLIP configs as 'ModelName/pretrained', e.g. ViT-B-16/laion2b_s34b_b88k. "
            "Default: full list in replica_sem_benchmark/clip_model_catalog.py "
            "(+ MobileCLIP2-* if your open_clip build lists them)."
        ),
    )

    # ─ Sweep: approaches ──────────────────────────────────────────────────────
    ap.add_argument(
        "--approaches",
        nargs="+",
        choices=["threshold_grid", "threshold_grid_dbscan"],
        default=["threshold_grid", "threshold_grid_dbscan"],
        help=(
            "threshold_grid: CLIP argmax on filtered masks; "
            "threshold_grid_dbscan: same precompute + DBSCAN merge + argmax on clusters."
        ),
    )
    ap.add_argument(
        "--enable_dbscan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable/disable Approach 2 (threshold_grid_dbscan = threshold grid + DBSCAN). "
            "Use --no-enable_dbscan to skip the extra 3×|dbscan_eps_grid| runs "
            "(three fixed threshold pairs, independent of --full_grid)."
        ),
    )
    ap.add_argument(
        "--full_grid",
        action="store_true",
        help="Run full 5×5=25 threshold combos for Approach 1 instead of 3 representative pairs.",
    )

    # ─ Approach 2 (threshold_grid_dbscan) ─────────────────────────────────────
    ap.add_argument(
        "--dbscan_eps_grid",
        type=float,
        nargs="+",
        default=[0.12, 0.16, 0.20, 0.24, 0.28],
        metavar="EPS",
        help=(
            "DBSCAN cosine-distance eps values for Approach 2 (each × three fixed threshold pairs). "
            "Default: five steps around 0.20."
        ),
    )

    # ─ Hardware ───────────────────────────────────────────────────────────────
    ap.add_argument("--device",         default="cuda:0",
                    help="Torch device (default: cuda:0)")
    ap.add_argument("--vram_limit_gb",  type=float, default=14.0,
                    help="Skip models exceeding this VRAM estimate (GB); default 14 (2 GB buffer).")

    ap.add_argument(
        "--use-boxes",
        action="store_true",
        help=(
            "CLIP: embed padded bounding-box crops (background visible) instead of tight "
            "mask crops with black background. SAM masks are unchanged for metrics."
        ),
    )
    ap.add_argument(
        "--box-pad",
        type=int,
        default=20,
        metavar="PX",
        help="Expand each bbox side by this many pixels (clamped to image); used with --use-boxes.",
    )

    # ─ Output ─────────────────────────────────────────────────────────────────
    ap.add_argument(
        "--out_csv",
        type=Path,
        default=_DEFAULT_OUT_CSV,
        help=f"Save results as CSV (pandas). Default: {_DEFAULT_OUT_CSV}",
    )
    ap.add_argument(
        "--out_csv_per_class",
        type=Path,
        default=None,
        help=(
            "Wide CSV: one row per (sample × experiment) with columns iou__* and "
            "precision_hit__* per class (NaN if class absent on image / not queried / "
            "no masks pass threshold). Default: <out_csv_stem>_per_class.csv beside --out_csv."
        ),
    )
    ap.add_argument(
        "--no-per-class-csv",
        action="store_true",
        help="Do not write the per-class CSV (overrides default when --out_csv is set).",
    )
    ap.add_argument("--wandb_project",  default=None,
                    help="Weights & Biases project name (optional).")

    # ─ SAM1 checkpoint (sweep: --sam_models local; legacy: default if omitted) ─
    ap.add_argument(
        "--sam_ckpt",
        default=None,
        help=(
            "Sweep: SAM1 .pth when --sam_models is local|sam1 (default file: "
            "ckpts/sam_vit_b_01ec64.pth if omitted). Legacy: same default if omitted."
        ),
    )
    ap.add_argument("--clip_model",     default="ViT-B-16",
                    help="[legacy] open_clip model name.")
    ap.add_argument("--clip_pretrained", default="laion2b_s34b_b88k",
                    help="[legacy] open_clip pretrained weight tag.")
    ap.add_argument("--encoder",        default=None,
                    help="[legacy] Autoencoder .pth (CLIP → latent).")

    return ap


def _apply_sam_presets(args: argparse.Namespace, ap: argparse.ArgumentParser) -> None:
    """Expand preset:hf_local* into SAM1 ``.pth`` paths + HF snapshot dirs under root."""
    if not args.sam_models or len(args.sam_models) != 1:
        return
    preset = args.sam_models[0].strip()
    if preset not in ("preset:hf_local", "preset:hf_local_no_sam3"):
        return
    root = (args.sam_models_root or default_sam_hf_snapshots_root()).resolve()
    exclude_s3 = preset == "preset:hf_local_no_sam3"
    args.sam_models = expand_preset_sam_dir_models(root, exclude_sam3_hf_folders=exclude_s3)
    if exclude_s3:
        args._sam_preset_note = (
            "SAM3: not listed — preset:hf_local_no_sam3 skips HF dirs whose name contains "
            "``sam3``. Use ``--sam_models preset:hf_local`` to include SAM3 snapshots."
        )
    else:
        args._sam_preset_note = (
            "SAM3: any HF snapshot under --sam_models_root is included if present "
            "(folder with config.json)."
        )
    if not args.sam_models:
        ap.error(
            f"{preset}: no models under {root}. "
            "Add ``sam1/*.pth`` (SAM1) and/or subfolders with ``config.json`` (HF SAM2/SAM3)."
        )


def main() -> None:
    # Not a TTY under ``nohup`` / pipes — default stdio is block-buffered, so ``nohup.out``
    # stays empty until flush. Line-buffer so logs appear immediately.
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(line_buffering=True)
            except Exception:
                pass

    ap   = _build_parser()
    args = ap.parse_args()

    _oc = getattr(args, "out_csv", None)
    if getattr(args, "no_per_class_csv", False):
        args._per_class_csv_path = None
    elif getattr(args, "out_csv_per_class", None) is not None:
        args._per_class_csv_path = _resolve(args.out_csv_per_class)
    elif _oc:
        args._per_class_csv_path = _resolve(
            Path(_oc).with_name(f"{Path(_oc).stem}_per_class.csv")
        )
    else:
        args._per_class_csv_path = None

    _ensure_hf_home_for_clip_cache()

    # ── Legacy mode ────────────────────────────────────────────────────────────
    if args.mode == "legacy":
        if args.class_name is None and args.queries_json is None and not args.queries_from_gt:
            ap.error("Legacy mode requires --class_name, --queries_json, or --queries_from_gt.")
        run_legacy(args)
        return

    # ── Sweep mode ─────────────────────────────────────────────────────────────
    if args.class_name is None and args.queries_json is None and not args.queries_from_gt:
        ap.error("Provide --class_name, --queries_json, or --queries_from_gt.")

    _apply_sam_presets(args, ap)

    # Convenience toggle: let users skip Approach 2 (DBSCAN) without having to
    # spell out --approaches threshold_grid explicitly.
    if not getattr(args, "enable_dbscan", True):
        args.approaches = [a for a in (args.approaches or []) if a != "threshold_grid_dbscan"]
        if not args.approaches:
            # Keep at least Approach 1 so the run does something useful.
            args.approaches = ["threshold_grid"]

    man_path = _resolve(args.manifest)
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    info_path = _resolve_rel(
        man_path,
        manifest.get("info_semantic", "data/replica_v1/office_0/habitat/info_semantic.json"),
    )
    name_to_id    = _load_name_to_id(info_path)
    id_to_name    = _load_id_to_name(info_path)
    queries_by_id: dict[str, Any] = {}
    if args.queries_json is not None:
        queries_by_id = _load_queries_json(_resolve(args.queries_json))

    samples = manifest["samples"]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    sam_ids = args.sam_models or SAM_MODELS_DEFAULT
    sam_ids = _filter_sam_models_by_family_flags(sam_ids, args)
    if not sam_ids:
        ap.error(
            "Sweep: pass at least one of --sam1, --sam2, --sam3 so that some SAM family is "
            "included (each flag enables that family from --sam_models / presets). "
            "With no family flags, the model list is empty."
        )
    sam_ids, skipped_hf_tf = _partition_sam_models_for_transformers(sam_ids)
    if skipped_hf_tf:
        try:
            import transformers as tr
            tf_ver = tr.__version__
        except Exception:
            tf_ver = "unknown"
        print(
            f"[SKIP] transformers {tf_ver} — need >= {_HF_SAM2_MIN_VERSION[0]}.{_HF_SAM2_MIN_VERSION[1]} "
            f"for Hugging Face SAM2/SAM3 ({len(skipped_hf_tf)} model(s) excluded):"
        )
        for s in skipped_hf_tf:
            print(f"       · {s}")
        print(
            '  Fix: pip install -U "transformers>=4.56.0"  OR  '
            "--sam_models local --sam_ckpt …\n"
        )
    args.sam_models = sam_ids
    if not sam_ids:
        print(
            "[error] No SAM models left to run. Upgrade transformers for HF checkpoints, "
            "or use SAM1: --sam_models local [--sam_ckpt path/to/sam_vit_b_01ec64.pth]"
        )
        sys.exit(1)

    clip_cfgs = _parse_clip_cfgs(args.clip_models)
    approaches = set(args.approaches or ["threshold_grid", "threshold_grid_dbscan"])
    n_thresh   = 25 if args.full_grid else len(_THRESH_COMBOS_9)
    app_set    = approaches

    print(f"\n{'═'*70}")
    print("SAM × CLIP Systematic Benchmark")
    print(f"{'═'*70}")
    print(f"  SAM  models  : {', '.join(sam_ids)}")
    print(
        f"  SAM families : --sam1={bool(args.sam1)}  --sam2={bool(args.sam2)}  --sam3={bool(args.sam3)}"
    )
    _preset_note = getattr(args, "_sam_preset_note", None)
    if _preset_note:
        print(f"  {_preset_note}")
    print(f"  CLIP models  : {len(clip_cfgs)}")
    for _i, (_mn, _pre) in enumerate(clip_cfgs, 1):
        print(f"    {_i:2d}. {_mn} / {_pre}")
    print(f"  Approaches   : {approaches}")
    print(
        f"  Thresh combos: {n_thresh} "
        f"({'full 5×5' if args.full_grid else f'{len(_THRESH_COMBOS_9)} representative'})"
    )
    if "threshold_grid_dbscan" in app_set:
        eps_g = list(args.dbscan_eps_grid)
        n_dbscan_thresh = len(_THRESH_COMBOS_9)
        print(
            f"  DBSCAN ε grid: {len(eps_g)} values  →  "
            f"Approach 2 runs: {n_dbscan_thresh * len(eps_g)} configs "
            f"(3 threshold pairs × ε; ignores --full_grid for DBSCAN)"
        )
    print(f"  Samples      : {len(samples)}")
    print(f"  VRAM limit   : {args.vram_limit_gb} GB")
    print(
        f"  CLIP crops   : "
        f"{'bbox+' + str(int(args.box_pad)) + 'px pad' if args.use_boxes else 'tight mask (bg black)'}"
    )
    print(f"  Device       : {args.device}")
    if any(_is_local_sam1(s) for s in sam_ids):
        ckpt_show = _resolve_sam1_checkpoint(sam_ids[0], args.sam_ckpt)
        print(f"  SAM1 ckpt    : {ckpt_show}")
    if getattr(args, "_per_class_csv_path", None):
        print(f"  Per-class CSV: {args._per_class_csv_path}  (--no-per-class-csv to skip)")
    print(f"{'═'*70}\n")

    records = run_sweep(
        args, samples, man_path, name_to_id, id_to_name, queries_by_id
    )

    print(f"\n{'═'*70}")
    print(f"RESULTS  ({len(records)} configurations evaluated)")
    print(f"{'═'*70}")
    _print_results_table(records)

    if getattr(args, "out_csv", None):
        op = Path(args.out_csv)
        if records:
            print(f"\nCSV (incremental rows): {op}  ({len(records)} rows)")
        else:
            print(f"\nCSV placeholder (no rows): {op}")


if __name__ == "__main__":
    main()
