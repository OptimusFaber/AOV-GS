#!/usr/bin/env python3
"""
Visualize SAM+CLIP segmentation maps saved during ActiveSGM training (same as LangSplat ``*_s.npy``).

Training writes, per keyframe ``frame_id``::

  <result_dir>/language_features/{frame_id:06d}_s.npy   # int32 (4, H, W)
  <result_dir>/language_features/{frame_id:06d}_f.npy   # float16 (N, 512)

The four slices are levels ``default``, ``s``, ``m``, ``l`` (see ``sam_clip_extractor._LEVELS``).
Pixel values are mask indices (cumulative across levels in ``_f.npy``) or ``-1`` for background.

If you ran ``activesgm.py`` with ``--debug``, raster overlays were also saved as::

  <result_dir>/segmentframes/frame_{frame_id:06d}.jpg

Examples
--------
  # One frame → PNGs per level in ./seg_viz/
  python scripts/visualize_language_seg_maps.py \\
    --lang_dir results/.../language_features --frame 12 --out_dir seg_viz

  # All *_s.npy in folder
  python scripts/visualize_language_seg_maps.py --lang_dir results/.../language_features \\
    --out_dir seg_viz --all
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

_LEVEL_NAMES = ("default", "s", "m", "l")


def _seg_layer_to_bgr(seg_hw: np.ndarray, seed: int) -> np.ndarray:
    """Map int32 instance ids (-1 = bg) to random colors (same idea as SAM debug overlay)."""
    seg_hw = np.asarray(seg_hw, dtype=np.int64)
    h, w = seg_hw.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    ids = np.unique(seg_hw)
    ids = ids[ids >= 0]
    rng = np.random.default_rng(seed)
    for i in ids:
        c = rng.integers(0, 256, 3, dtype=np.uint8)
        out[seg_hw == i] = c
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def main() -> None:
    p = argparse.ArgumentParser(description="Render PNGs from language_features *_s.npy (SAM levels).")
    p.add_argument("--lang_dir", type=str, required=True, help=".../language_features")
    p.add_argument("--frame", type=int, default=None, help="Keyframe id (matches 000012_s.npy).")
    p.add_argument("--all", action="store_true", help="Process every *_s.npy in lang_dir.")
    p.add_argument("--out_dir", type=str, default="seg_maps_viz")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    lang = os.path.abspath(args.lang_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.all:
        names = sorted(f for f in os.listdir(lang) if f.endswith("_s.npy"))
        if not names:
            print(f"No *_s.npy under {lang}", file=sys.stderr)
            sys.exit(1)
        frames = [int(x.replace("_s.npy", "")) for x in names]
    else:
        if args.frame is None:
            print("Pass --frame ID or --all", file=sys.stderr)
            sys.exit(1)
        frames = [int(args.frame)]

    for fid in frames:
        path = os.path.join(lang, f"{fid:06d}_s.npy")
        if not os.path.isfile(path):
            print(f"Missing {path}", file=sys.stderr)
            continue
        seg = np.load(path)  # (4, H, W) int32
        if seg.ndim != 3 or seg.shape[0] != 4:
            print(f"Unexpected shape {seg.shape} in {path} (expected (4,H,W))", file=sys.stderr)
            continue
        base = os.path.join(args.out_dir, f"frame_{fid:06d}")
        os.makedirs(base, exist_ok=True)
        for li, name in enumerate(_LEVEL_NAMES):
            bgr = _seg_layer_to_bgr(seg[li], seed=args.seed + li * 997)
            out_p = os.path.join(base, f"level_{name}.png")
            cv2.imwrite(out_p, bgr)
        print(f"Wrote {base}/level_{{default,s,m,l}}.png")


if __name__ == "__main__":
    main()
