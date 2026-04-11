#!/usr/bin/env python3
"""
Render semantic .npy overlay on top of an image.

Inputs:
  --image    path to RGB image (jpg/png)
  --semantic path to semantic map (.npy), shape (H,W) with integer class ids

Optional:
  --info_semantic path to Replica info_semantic.json (for names; optional)
  --alpha overlay opacity (default 0.45)
  --out output path (default: ./semantic_overlay.png)
  --out_mask output colored mask image (default: none)
  --min_area_px only show classes with >= this many pixels (default 50)
  --draw_legend draw a small legend with top classes (default on)
  --legend_k number of classes in legend (default 12)

Example:
  python3 scripts/render_semantic_overlay.py \
    --image replica_sem_benchmark/images/00000.jpg \
    --semantic replica_sem_benchmark/semantic/00000.npy \
    --out output/overlay_00000.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _add_bool_optional(
    ap: argparse.ArgumentParser, name: str, *, default: bool, help: str | None = None
) -> None:
    """
    Python 3.7/3.8 compatibility for argparse.BooleanOptionalAction (added in 3.9).
    Provides both --foo and --no-foo when BooleanOptionalAction is unavailable.
    """
    action = getattr(argparse, "BooleanOptionalAction", None)
    if action is not None:
        ap.add_argument(name, action=action, default=default, help=help)
        return

    # Fallback: add a mutually exclusive group with explicit on/off flags.
    dest = name.lstrip("-").replace("-", "_")
    group = ap.add_mutually_exclusive_group(required=False)
    group.add_argument(name, dest=dest, action="store_true", help=help)
    group.add_argument(f"--no-{name.lstrip('-')}", dest=dest, action="store_false", help=help)
    ap.set_defaults(**{dest: default})


def _load_id_to_name(info_semantic_path: Path) -> dict[int, str]:
    data = json.loads(info_semantic_path.read_text(encoding="utf-8"))
    return {int(c["id"]): str(c["name"]) for c in data.get("classes", [])}


def _color_for_id(cid: int) -> tuple[int, int, int]:
    """
    Deterministic vivid-ish BGR color from class id.
    (Avoids requiring matplotlib.)
    """
    # Simple hash → HSV → BGR
    h = (cid * 37) % 179
    s = 220
    v = 255
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _make_color_mask(sem: np.ndarray, *, min_area_px: int) -> tuple[np.ndarray, dict[int, int]]:
    """
    sem: (H,W) int
    Returns (mask_bgr_uint8, counts_by_id).
    """
    h, w = sem.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    ids, counts = np.unique(sem.astype(np.int64), return_counts=True)
    counts_by_id = {int(i): int(c) for i, c in zip(ids.tolist(), counts.tolist()) if int(i) > 0}
    for cid, cnt in counts_by_id.items():
        if cnt < int(min_area_px):
            continue
        out[sem == cid] = _color_for_id(cid)
    return out, counts_by_id


def _draw_legend(
    canvas_bgr: np.ndarray,
    counts_by_id: dict[int, int],
    *,
    id_to_name: dict[int, str] | None,
    k: int,
) -> None:
    """Draw top-k classes by pixel count in the top-left corner (in-place)."""
    if not counts_by_id:
        return
    h, w = canvas_bgr.shape[:2]
    items = sorted(counts_by_id.items(), key=lambda x: x[1], reverse=True)[: int(k)]

    pad = 10
    box_w = min(520, w - 2 * pad)
    line_h = 20
    box_h = pad + line_h * (len(items) + 1)

    x0, y0 = pad, pad
    x1, y1 = x0 + box_w, y0 + box_h

    overlay = canvas_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, canvas_bgr, 0.45, 0.0, dst=canvas_bgr)

    cv2.putText(
        canvas_bgr,
        "Top semantic classes (by pixels)",
        (x0 + 10, y0 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    y = y0 + 22 + line_h
    for cid, cnt in items:
        color = _color_for_id(cid)
        cv2.rectangle(canvas_bgr, (x0 + 10, y - 12), (x0 + 26, y + 2), color, thickness=-1)
        name = id_to_name.get(cid, str(cid)) if id_to_name is not None else str(cid)
        label = f"{cid:>2d}: {name}  ({cnt}px)"
        cv2.putText(
            canvas_bgr,
            label[:60],
            (x0 + 34, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += line_h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--semantic", type=Path, required=True)
    ap.add_argument("--info_semantic", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--out", type=Path, default=Path("semantic_overlay.png"))
    ap.add_argument("--out_mask", type=Path, default=None)
    ap.add_argument("--min_area_px", type=int, default=50)
    _add_bool_optional(ap, "--draw_legend", default=True)
    ap.add_argument("--legend_k", type=int, default=12)
    args = ap.parse_args()

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    sem = np.load(str(args.semantic))
    if sem.ndim != 2:
        raise SystemExit(f"Expected semantic shape (H,W), got {sem.shape} in {args.semantic}")
    if sem.shape[:2] != bgr.shape[:2]:
        raise SystemExit(f"Shape mismatch: image {bgr.shape[:2]} vs semantic {sem.shape[:2]}")

    id_to_name = _load_id_to_name(args.info_semantic) if args.info_semantic is not None else None
    mask_bgr, counts_by_id = _make_color_mask(sem, min_area_px=int(args.min_area_px))

    overlay = cv2.addWeighted(bgr, 1.0, mask_bgr, float(args.alpha), 0.0)
    if bool(args.draw_legend):
        _draw_legend(overlay, counts_by_id, id_to_name=id_to_name, k=int(args.legend_k))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(args.out), overlay)
    if not ok:
        raise SystemExit(f"Cannot write: {args.out}")

    if args.out_mask is not None:
        args.out_mask.parent.mkdir(parents=True, exist_ok=True)
        ok2 = cv2.imwrite(str(args.out_mask), mask_bgr)
        if not ok2:
            raise SystemExit(f"Cannot write: {args.out_mask}")

    print(f"Saved overlay: {args.out}")
    if args.out_mask is not None:
        print(f"Saved mask:    {args.out_mask}")


if __name__ == "__main__":
    main()

