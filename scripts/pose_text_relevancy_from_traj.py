#!/usr/bin/env python3
"""
Render text-query relevancy heatmaps from a few poses in replica_sim_nvs traj.txt.

RU:
- Берём до 10 поз (c2w, RDF) из `data/replica_sim_nvs/<scene>/traj.txt`.
- Переводим их в систему координат Gaussian checkpoint (train0-relative):
    w2c_gs = inv(c2w_abs) @ c2w_train0
  где `c2w_train0` берём как позу #0 из `data/Replica/<scene>/traj.txt` (тоже RDF).
- Для каждой позы рендерим:
  1) RGB из `params.npz`
  2) тепловую карту релевантности тексту:
     cos( decode(latent_pixel), CLIP_text ) по пикселям (видимые гауссианы дают вклад)
  3) overlay (RGB + heatmap)

EN:
Sample up to 10 traj poses, map them into the Gaussian checkpoint frame, then render
RGB and a per-pixel text relevancy heatmap (cosine in CLIP space).
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

from scripts.query_language_field import (  # noqa: E402
    _load_ae,
    encode_query_clip,
    infer_latent_dim,
    render_relevancy_map,
    render_rgb,
)
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
    c2w_abs_rdf = np.asarray(c2w_abs_rdf, dtype=np.float64).reshape(4, 4)
    c2w_train0_rdf = np.asarray(c2w_train0_rdf, dtype=np.float64).reshape(4, 4)
    return (np.linalg.inv(c2w_abs_rdf) @ c2w_train0_rdf).astype(np.float64)


def _pick_indices(total: int, max_poses: int, seed: int) -> list[int]:
    if total <= max_poses:
        return list(range(total))
    rng = np.random.default_rng(seed)
    base = np.linspace(0, total - 1, num=max_poses, dtype=np.int64).tolist()
    out: list[int] = []
    for idx in base:
        j = int(rng.integers(-1, 2))
        out.append(int(np.clip(idx + j, 0, total - 1)))
    seen = set()
    uniq: list[int] = []
    for i in out:
        if i in seen:
            continue
        seen.add(i)
        uniq.append(i)
    if len(uniq) < max_poses:
        remain = [i for i in range(total) if i not in seen]
        rng.shuffle(remain)
        uniq.extend(remain[: (max_poses - len(uniq))])
    return uniq[:max_poses]


def _frame_score_from_sim(sim_raw: np.ndarray, top_frac: float = 0.05) -> float:
    """Scalar for ranking frames: mean of top fraction of similarity pixels."""
    flat = sim_raw.astype(np.float64, copy=False).ravel()
    k = max(1, int(round(float(top_frac) * flat.size)))
    top = np.partition(flat, -k)[-k:]
    return float(np.mean(top))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to params.npz (SplaTAM final).")
    p.add_argument("--lang_field", required=True, help="Path to lang_field.pt.")
    p.add_argument("--ae_ckpt", required=True, help="Autoencoder checkpoint (best_ckpt.pth).")
    p.add_argument("--text", required=True, help='Text query, e.g. "a chair".')

    p.add_argument("--traj", required=True, help="Path to replica_sim_nvs/.../traj.txt (c2w RDF).")
    p.add_argument("--train_traj0", required=True, help="Path to data/Replica/<scene>/traj.txt (c2w RDF).")

    p.add_argument("--max_poses", type=int, default=10, help="Max poses to render (<=10).")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    p.add_argument("--device", default="cuda:0", help="Torch device.")
    p.add_argument("--out_dir", required=True, help="Output directory.")

    p.add_argument("--clip_model", default="ViT-B-16")
    p.add_argument("--clip_pretrained", default="laion2b_s34b_b88k")
    p.add_argument("--encoder_dims", nargs="+", type=int, default=None)
    p.add_argument("--decoder_dims", nargs="+", type=int, default=None)

    p.add_argument("--heatmap_norm", choices=("percentile", "minmax"), default="percentile")
    p.add_argument("--heatmap_p_low", type=float, default=8.0)
    p.add_argument("--heatmap_p_high", type=float, default=98.0)
    p.add_argument("--heatmap_blur", type=float, default=3.0)
    p.add_argument("--overlay_alpha", type=float, default=0.45, help="Heatmap alpha in overlay.")
    p.add_argument("--score_top_frac", type=float, default=0.05, help="Top fraction for frame scoring.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    max_poses = int(min(max(int(args.max_poses), 1), 10))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_path = Path(args.traj)
    ckpt_path = Path(args.checkpoint)
    lf_path = Path(args.lang_field)
    train0_path = Path(args.train_traj0)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    latent_dim = infer_latent_dim(lf_path)

    model = LangSplatam(checkpoint_path=str(ckpt_path), latent_dim=int(latent_dim), device=str(device))
    model.load_lang_field(lf_path)
    H = int(model.params["org_height"])
    W = int(model.params["org_width"])

    c2w_list = _read_traj_c2w_rdf(traj_path)
    c2w_train0 = _read_traj_c2w_rdf(train0_path)[0]
    idxs = _pick_indices(len(c2w_list), max_poses, int(args.seed))

    print(f'Query: "{args.text}"')
    print(f"traj poses: {len(c2w_list)}  picked: {idxs}")
    print(f"render: HxW={H}x{W}  latent_dim={latent_dim}  device={device}")

    ae = _load_ae(Path(args.ae_ckpt), args.encoder_dims, args.decoder_dims, device)
    clip_query = encode_query_clip(args.text, args.clip_model, args.clip_pretrained, device)

    rmap_kw = dict(
        heatmap_norm=args.heatmap_norm,
        heatmap_p_low=float(args.heatmap_p_low),
        heatmap_p_high=float(args.heatmap_p_high),
        blur_sigma=float(args.heatmap_blur),
    )

    results = []
    for picked_order, pose_idx in enumerate(idxs):
        w2c_np = _w2c_in_checkpoint_frame(c2w_list[pose_idx], c2w_train0)
        w2c = torch.tensor(w2c_np, dtype=torch.float32, device=device)

        rgb_bgr = render_rgb(model, w2c, H, W)
        jet_bgr, _norm01, sim_raw = render_relevancy_map(model, clip_query, ae, w2c, H, W, **rmap_kw)
        overlay = cv2.addWeighted(rgb_bgr, 1.0 - float(args.overlay_alpha), jet_bgr, float(args.overlay_alpha), 0.0)

        score = _frame_score_from_sim(sim_raw, top_frac=float(args.score_top_frac))

        stem = out_dir / f"traj_{pose_idx:04d}"
        cv2.imwrite(str(stem) + "_rgb.png", rgb_bgr)
        cv2.imwrite(str(stem) + "_heat.png", jet_bgr)
        cv2.imwrite(str(stem) + "_overlay.png", overlay)

        results.append(
            {
                "traj_pose_idx": int(pose_idx),
                "picked_order": int(picked_order),
                "score_topmean": float(score),
                "rgb_png": str(stem.name + "_rgb.png"),
                "heat_png": str(stem.name + "_heat.png"),
                "overlay_png": str(stem.name + "_overlay.png"),
                "w2c": w2c_np.tolist(),
            }
        )
        print(f"pose {pose_idx:4d}: score={score:.4f}  -> {stem.name}_{{rgb,heat,overlay}}.png")

    ranked = sorted(results, key=lambda r: -r["score_topmean"])
    summary = {
        "text": str(args.text),
        "checkpoint": str(ckpt_path.resolve()),
        "lang_field": str(lf_path.resolve()),
        "ae_ckpt": str(Path(args.ae_ckpt).resolve()),
        "traj": str(traj_path.resolve()),
        "train_traj0": str(train0_path.resolve()),
        "clip_model": f"{args.clip_model}/{args.clip_pretrained}",
        "latent_dim": int(latent_dim),
        "H": int(H),
        "W": int(W),
        "picked_indices": [int(i) for i in idxs],
        "ranked": ranked,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved summary: {out_dir / 'summary.json'}")
    print("Top-5 poses by text relevancy score:")
    for i, r in enumerate(ranked[:5], 1):
        print(f"  {i}. idx={r['traj_pose_idx']}  score={r['score_topmean']:.4f}")


if __name__ == "__main__":
    os.chdir(str(ROOT))
    main()

