#!/usr/bin/env python3
"""
Render RGB scene views from each pose in keyframe_poses.json (w2c as in SplaTAM).

Uses only the Gaussian checkpoint (params.npz); language field not required.

Important: ``keyframe_poses.json`` and ``params.npz`` must come from the same
run (one ``result_dir``). Otherwise the camera will be “outside the scene” — black/empty
frame, even though the geometry in the npz is correct.

Example
-------
python scripts/render_keyframe_poses.py \\
  --checkpoint results/splatam/final/params.npz \\
  --poses      results/keyframe_poses.json \\
  --out_dir    results/keyframe_rgb_renders

Check without rendering (recommended when frames look wrong)::

  python scripts/render_keyframe_poses.py --checkpoint ... --poses ... \\
    --out_dir /tmp/x --check_only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from src.slam.langsplatam.langsplatam import LangSplatam


def _camera_centers_from_poses_json(
    poses_raw: dict, max_frames: int
) -> Tuple[List[int], np.ndarray]:
    """Parse keyframe_poses.json → frame ids and camera centers (world, c2w t)."""
    items = sorted(((int(k), v) for k, v in poses_raw.items()), key=lambda x: x[0])
    if max_frames > 0:
        items = items[:max_frames]
    centers = []
    fids = []
    for fid, mat in items:
        w2c = np.asarray(mat, dtype=np.float64)
        c2w = np.linalg.inv(w2c)
        centers.append(c2w[:3, 3].copy())
        fids.append(fid)
    return fids, np.stack(centers, axis=0) if centers else np.zeros((0, 3))


def check_poses_vs_gaussians(
    checkpoint_path: str,
    poses_path: Path,
    max_frames: int = 0,
    margin_m: float = 0.5,
) -> None:
    """
    Compares camera centers from JSON to the means3D cloud from the checkpoint.
    Prints stats; on large mismatch suggests a mixed run.
    """
    raw = dict(np.load(checkpoint_path, allow_pickle=True))
    if "means3D" not in raw:
        print("check: no means3D in npz — skip.")
        return
    means = np.asarray(raw["means3D"], dtype=np.float64).reshape(-1, 3)
    lo = np.percentile(means, 1.0, axis=0)
    hi = np.percentile(means, 99.0, axis=0)
    center_cloud = 0.5 * (lo + hi)
    extent = np.linalg.norm(hi - lo)

    with open(poses_path, encoding="utf-8") as f:
        poses_raw = json.load(f)
    fids, cam_centers = _camera_centers_from_poses_json(poses_raw, max_frames)
    if len(cam_centers) == 0:
        print("check: no poses in JSON.")
        return

    dists = np.linalg.norm(cam_centers - center_cloud.reshape(1, 3), axis=1)
    inside = np.all(
        (cam_centers >= lo - margin_m) & (cam_centers <= hi + margin_m), axis=1
    )
    n_out = int(np.sum(~inside))
    print(
        f"check: means3D AABB (p1–p99) lo={lo}, hi={hi}, "
        f"||hi-lo||≈{extent:.3f} m"
    )
    print(
        f"check: cameras in JSON: {len(fids)}, distance to cloud center: "
        f"min={dists.min():.3f} m, median={np.median(dists):.3f} m, max={dists.max():.3f} m"
    )
    print(
        f"check: inside expanded AABB (±{margin_m} m): "
        f"{len(fids) - n_out}/{len(fids)}"
    )
    if n_out > len(fids) // 2 or np.median(dists) > 0.5 * max(extent, 1.0):
        print(
            "check: WARNING — most cameras are far from the Gaussian cloud. "
            "Often these are different runs: take keyframe_poses.json and params.npz "
            "from the same results/ folder of one experiment."
        )


def render_rgb(model: LangSplatam, w2c: torch.Tensor, H: int, W: int) -> np.ndarray:
    sys.path.insert(0, str(ROOT / "third_parties" / "splatam"))
    from utils.slam_helpers import transformed_params2rendervar
    from diff_gaussian_rasterization import GaussianRasterizer as Renderer

    cam = model._setup_camera(H, W, w2c)
    tr = model._transform_gaussians(w2c)
    rv = transformed_params2rendervar(model.params, tr)
    with torch.no_grad():
        rgb, _, _ = Renderer(raster_settings=cam)(**rv)
    img = rgb.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RGB render each keyframe w2c from JSON.")
    p.add_argument("--checkpoint", required=True, help="params*.npz from SplaTAM")
    p.add_argument("--poses", required=True, help="keyframe_poses.json")
    p.add_argument("--out_dir", required=True, help="Where to save PNG + manifest")
    p.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for rendering (default cuda:0).",
    )
    p.add_argument(
        "--latent_dim",
        type=int,
        default=3,
        help="Only for the LangSplatam constructor; RGB does not use lang_feats.",
    )
    p.add_argument("--max_frames", type=int, default=0, help="0 = all keyframes")
    p.add_argument(
        "--check_only",
        action="store_true",
        help="Only check pose consistency with means3D (no render).",
    )
    p.add_argument(
        "--no_precheck",
        action="store_true",
        help="Skip the quick AABB check before rendering.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    poses_path = Path(args.poses).expanduser().resolve()
    if not poses_path.is_file():
        raise FileNotFoundError(poses_path)

    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    if args.check_only:
        check_poses_vs_gaussians(str(ckpt), poses_path, args.max_frames)
        return

    if not args.no_precheck:
        print("Precheck (means3D vs camera poses from JSON)...")
        check_poses_vs_gaussians(str(ckpt), poses_path, args.max_frames)
        print("")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = LangSplatam(
        checkpoint_path=str(ckpt),
        latent_dim=args.latent_dim,
        device=args.device,
    )
    H = int(model.params["org_height"])
    W = int(model.params["org_width"])
    print(f"Resolution {W}×{H}")

    with open(poses_path, encoding="utf-8") as f:
        poses_raw = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = sorted(((int(k), v) for k, v in poses_raw.items()), key=lambda x: x[0])
    if args.max_frames > 0:
        items = items[: args.max_frames]

    manifest_lines = [
        f"# poses_file: {poses_path}",
        f"# checkpoint: {Path(args.checkpoint).resolve()}",
        f"# resolution: {W}x{H}",
        f"# frames: {len(items)}",
        "",
    ]

    for fid, mat in items:
        w2c = torch.tensor(mat, dtype=torch.float32, device=device)
        bgr = render_rgb(model, w2c, H, W)
        name = f"{fid:06d}_rgb.png"
        out_path = out_dir / name
        cv2.imwrite(str(out_path), bgr)
        manifest_lines.append(f"{fid:06d}\t{name}")
        print(f"  saved {out_path}")

    man_path = out_dir / "render_manifest.txt"
    man_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"Manifest: {man_path}")
    print("Done.")


if __name__ == "__main__":
    main()
