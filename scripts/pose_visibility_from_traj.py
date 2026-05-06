#!/usr/bin/env python3
"""
Pick a few camera poses from replica_sim_nvs traj.txt and render how much of the
Gaussian scene is visible from each pose.

RU:
- `traj.txt` stores absolute RDF camera-to-world (c2w) poses.
- Gaussian checkpoint lives in the **training** trajectory frame where pose #0 is
  treated as the origin (first SLAM frame). To convert a dataset/world pose to the
  checkpoint frame we use:
    \( w2c_{gs} = inv(c2w_{abs}) @ c2w_{train0} \)
  where `c2w_train0` is pose #0 from `data/Replica/<scene>/traj.txt` (also RDF).
- For each pose we render:
  - RGB (3DGS color render)
  - "visibility" heatmap = per-pixel norm of rendered language latents
    (higher means more Gaussians contribute to that pixel).

EN:
This script samples up to N poses from traj.txt, renders RGB from the splat
checkpoint, and ranks poses by a simple visibility score derived from the
rendered latent field norm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.query_language_field import infer_latent_dim, render_rgb  # noqa: E402
from src.slam.langsplatam.langsplatam import LangSplatam  # noqa: E402


def _read_traj_c2w_rdf(traj_path: Path) -> list[np.ndarray]:
    lines = traj_path.read_text().strip().splitlines()
    mats: list[np.ndarray] = []
    for ln in lines:
        parts = ln.strip().split()
        if not parts:
            continue
        if len(parts) != 16:
            raise ValueError(f"Expected 16 floats per line in traj.txt, got {len(parts)}")
        m = np.array(list(map(float, parts)), dtype=np.float64).reshape(4, 4)
        mats.append(m)
    if not mats:
        raise ValueError(f"No poses found in {traj_path}")
    return mats


def _w2c_in_checkpoint_frame(c2w_abs_rdf: np.ndarray, c2w_train0_rdf: np.ndarray) -> np.ndarray:
    """
    Map an absolute RDF c2w pose into the GS checkpoint's "train0-relative" frame.
    See notes in src/data/generate_Replica_NVS_data.py (gt_w2c = inv(stored_pose) @ train_pose_0).
    """
    c2w_abs_rdf = np.asarray(c2w_abs_rdf, dtype=np.float64).reshape(4, 4)
    c2w_train0_rdf = np.asarray(c2w_train0_rdf, dtype=np.float64).reshape(4, 4)
    return (np.linalg.inv(c2w_abs_rdf) @ c2w_train0_rdf).astype(np.float64)


def _pick_indices(total: int, max_poses: int, seed: int) -> list[int]:
    if total <= max_poses:
        return list(range(total))
    # Prefer spread-out indices for coverage, plus a tiny jitter to avoid symmetry ties.
    rng = np.random.default_rng(seed)
    base = np.linspace(0, total - 1, num=max_poses, dtype=np.int64).tolist()
    # jitter by at most 1 frame, staying in bounds, to avoid exact duplicates across runs
    out: list[int] = []
    for idx in base:
        j = int(rng.integers(-1, 2))
        jj = int(np.clip(idx + j, 0, total - 1))
        out.append(jj)
    # de-dup while preserving order
    seen = set()
    uniq: list[int] = []
    for i in out:
        if i in seen:
            continue
        seen.add(i)
        uniq.append(i)
    # if we lost too many due to de-dup, fill with random distinct
    if len(uniq) < max_poses:
        remain = [i for i in range(total) if i not in seen]
        rng.shuffle(remain)
        uniq.extend(remain[: (max_poses - len(uniq))])
    return uniq[:max_poses]


def _norm_to_heatmap(norm01: np.ndarray) -> np.ndarray:
    u8 = (np.clip(norm01, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def _visibility_from_rendered_latents(lat: torch.Tensor) -> tuple[np.ndarray, float, float]:
    """
    lat: [D, H, W] float32
    Returns (norm01 [H,W], frac_visible, mean_norm)
    """
    with torch.no_grad():
        n = torch.linalg.norm(lat.to(torch.float32), dim=0)  # [H,W]
        n_np = n.detach().cpu().numpy().astype(np.float32, copy=False)

    # Robust normalization: stretch between low and high percentiles
    lo = float(np.percentile(n_np, 5.0))
    hi = float(np.percentile(n_np, 99.0))
    if hi <= lo + 1e-8:
        norm01 = np.zeros_like(n_np)
    else:
        norm01 = np.clip((n_np - lo) / (hi - lo), 0.0, 1.0)

    # Visibility: fraction above a small threshold in the *raw* norm space
    thr = float(np.percentile(n_np, 70.0))  # scene-dependent; rankable across poses
    frac = float(np.mean(n_np > max(thr, 1e-6)))
    mean_n = float(np.mean(n_np))
    return norm01, frac, mean_n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to params.npz (SplaTAM final).")
    p.add_argument("--lang_field", required=True, help="Path to lang_field.pt (to infer latent_dim and load feats).")
    p.add_argument("--traj", required=True, help="Path to replica_sim_nvs/.../traj.txt (c2w RDF).")
    p.add_argument(
        "--train_traj0",
        required=True,
        help="Path to training data/Replica/<scene>/traj.txt (c2w RDF). Pose #0 is used as train0.",
    )
    p.add_argument("--max_poses", type=int, default=10, help="Max poses to evaluate (<=10 requested).")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    p.add_argument("--device", default="cuda:0", help="Torch device for rendering.")
    p.add_argument("--out_dir", required=True, help="Output directory for renders and JSON summary.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    max_poses = int(min(max(args.max_poses, 1), 10))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_path = Path(args.traj)
    ckpt_path = Path(args.checkpoint)
    lf_path = Path(args.lang_field)

    latent_dim = infer_latent_dim(lf_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load splat + language field (we use render_lang just for a visibility proxy).
    model = LangSplatam(checkpoint_path=str(ckpt_path), latent_dim=int(latent_dim), device=str(device))
    model.load_lang_field(lf_path)
    H = int(model.params["org_height"])
    W = int(model.params["org_width"])

    c2w_list = _read_traj_c2w_rdf(traj_path)
    c2w_train0 = _read_traj_c2w_rdf(Path(args.train_traj0))[0]
    idxs = _pick_indices(len(c2w_list), max_poses, int(args.seed))
    print(f"traj poses: {len(c2w_list)}  picked: {idxs}")
    print(f"render: HxW={H}x{W}  latent_dim={latent_dim}  device={device}")

    results = []
    for rank_i, pose_idx in enumerate(idxs):
        w2c_np = _w2c_in_checkpoint_frame(c2w_list[pose_idx], c2w_train0)
        w2c = torch.tensor(w2c_np, dtype=torch.float32, device=device)

        rgb_bgr = render_rgb(model, w2c, H, W)
        with torch.no_grad():
            lat = model.render_lang(w2c, H, W)
        norm01, frac_vis, mean_norm = _visibility_from_rendered_latents(lat)

        heat = _norm_to_heatmap(norm01)
        overlay = cv2.addWeighted(rgb_bgr, 0.60, heat, 0.40, 0.0)

        stem = out_dir / f"traj_{pose_idx:04d}"
        cv2.imwrite(str(stem) + "_rgb.png", rgb_bgr)
        cv2.imwrite(str(stem) + "_vis.png", heat)
        cv2.imwrite(str(stem) + "_overlay.png", overlay)

        results.append(
            {
                "traj_pose_idx": int(pose_idx),
                "picked_order": int(rank_i),
                "visibility_frac": float(frac_vis),
                "visibility_mean_norm": float(mean_norm),
                "rgb_png": str((stem.name + "_rgb.png")),
                "vis_png": str((stem.name + "_vis.png")),
                "overlay_png": str((stem.name + "_overlay.png")),
                "w2c": w2c_np.tolist(),
            }
        )
        print(f"pose {pose_idx:4d}: frac_vis={frac_vis:.4f}  mean_norm={mean_norm:.4f}  -> {stem.name}_*.png")

    # Rank poses by visibility (higher is better).
    results_sorted = sorted(results, key=lambda r: (-r["visibility_frac"], -r["visibility_mean_norm"]))
    summary = {
        "checkpoint": str(ckpt_path.resolve()),
        "lang_field": str(lf_path.resolve()),
        "traj": str(traj_path.resolve()),
        "latent_dim": int(latent_dim),
        "H": int(H),
        "W": int(W),
        "picked_indices": [int(i) for i in idxs],
        "ranked": results_sorted,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved summary: {out_dir / 'summary.json'}")
    print("Top-5 poses by visibility:")
    for i, r in enumerate(results_sorted[:5], 1):
        print(f"  {i}. idx={r['traj_pose_idx']}  frac={r['visibility_frac']:.4f}  mean_norm={r['visibility_mean_norm']:.4f}")


if __name__ == "__main__":
    os.chdir(str(ROOT))
    main()

