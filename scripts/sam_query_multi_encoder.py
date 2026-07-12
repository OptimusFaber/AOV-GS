#!/usr/bin/env python3
"""
SAM (all auto-masks) + ranking by text query:

1) **OpenAI CLIP** (ViT-B-16, ``pretrained=openai``) — original CLIP, 512d.
2) Exactly **6** autoencoders: ``<ckpt_root>/<3|4|8|16|32|64>/best_ckpt.pth`` — AE on **Laion CLIP**
   (``laion2b_s34b_b88k``), as trained in this repository.

**7 PNGs total** (default): original CLIP + 6 truncated latents.

**Relevant masks by clusters** (like ``find_clusters`` in ``query_language_field.py``):
2D SAM-mask centroids → DBSCAN → clusters sorted by **sum** of cosines;
best mask = **argmax cosine inside the winning cluster**. If no clusters —
global argmax. Disable with ``--no_cluster``.

**Original CLIP (OpenAI) weights:** not in the repo; ``open_clip`` downloads them into the
Hugging Face cache, usually ``~/.cache/huggingface/hub/`` (repo like
``models--openai--CLIP-ViT-B-16-*`` / blob files). Load: ``ViT-B-16`` + ``pretrained=openai``.

Example::

    python scripts/sam_query_multi_encoder.py \\
        --image results/.../frame_000109.jpg \\
        --query "a sofa" \\
        --sam_ckpt ckpts/sam_vit_b_01ec64.pth \\
        --ckpt_root ckpt/office0

Files under ``<stem>_encoder_vis/``:

- ``semantic__openai_clip.png``
- ``semantic__ae_3.png`` … ``semantic__ae_64.png`` (from the corresponding subfolders)

Laion 512d is printed to the log; a separate PNG only with ``--save_laion_512d``.
Disable saving: ``--no_save``. Directory: ``--out_dir``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# scripts/ on PYTHONPATH to import sam_query_match
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import sam_query_match as sqm  # noqa: E402

# office0 subfolders named by latent dim (best_ckpt.pth in each)
DEFAULT_AE_SUBDIRS = ("3", "4", "8", "16", "32", "64")


def _parse_ae_subdirs(s: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def _resolve_office0_ae_ckpts(root: Path, subdirs: tuple[str, ...]) -> list[tuple[str, Path]]:
    """``<root>/<name>/best_ckpt.pth`` or ``best.pth``."""
    out: list[tuple[str, Path]] = []
    for name in subdirs:
        d = root / name
        p = d / "best_ckpt.pth"
        if not p.is_file():
            p = d / "best.pth"
        if p.is_file():
            out.append((name, p))
        else:
            print(f"  [warn] no best_ckpt.pth / best.pth in {d} — skip latent {name}")
    return out


def _cosine_scores(img_emb: torch.Tensor, q: torch.Tensor) -> np.ndarray:
    """Cosine similarity of each mask to the query, (N,) float."""
    sim = (img_emb.float() * q.float()).sum(dim=-1)
    return sim.detach().cpu().numpy()


def _mask_centroids_xy(masks_all: list) -> np.ndarray:
    """Centroid of each mask in pixels (x, y), shape (N, 2)."""
    out = np.zeros((len(masks_all), 2), dtype=np.float64)
    for i, m in enumerate(masks_all):
        seg = m["segmentation"]
        ys, xs = np.where(seg)
        if len(xs) == 0:
            out[i] = (0.0, 0.0)
        else:
            out[i] = (float(xs.mean()), float(ys.mean()))
    return out


def _find_clusters_2d(
    xy: np.ndarray,
    scores: np.ndarray,
    global_idx: np.ndarray,
    eps: float,
    min_samples: int,
) -> list[dict]:
    """DBSCAN on 2D points; clusters with ``total_score = sum(cos)``, as in ``query_language_field.find_clusters``."""
    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(xy)
    labels = db.labels_
    clusters: list[dict] = []
    for lbl in set(labels):
        if lbl == -1:
            continue
        m = labels == lbl
        gix = global_idx[m]
        clusters.append(
            {
                "centroid": xy[m].mean(axis=0),
                "total_score": float(scores[m].sum()),
                "size": int(m.sum()),
                "indices": gix.astype(np.int64),
                "scores": scores[m].astype(np.float64).copy(),
            }
        )
    clusters.sort(key=lambda c: -c["total_score"])
    return clusters


def pick_best_mask_clustered(
    scores: np.ndarray,
    centroids_xy: np.ndarray,
    *,
    top_percentile: float,
    eps: float,
    min_samples: int,
    use_clusters: bool,
) -> tuple[int, list[dict]]:
    """
    Returns mask index and cluster list (empty if disabled or noise-only).
    Cluster selection logic — same as 3D Gaussians: max ``sum(scores)`` in a cluster,
    then best mask by its own cos inside that cluster.
    """
    n = len(scores)
    if not use_clusters:
        return int(np.argmax(scores)), []

    if top_percentile >= 100.0:
        thr = -np.inf
    else:
        thr = np.percentile(scores, 100.0 - top_percentile)
    sel = scores >= thr
    if int(sel.sum()) < max(min_samples, 1):
        sel = np.ones(n, dtype=bool)
    idx = np.where(sel)[0]
    clusters = _find_clusters_2d(
        centroids_xy[idx], scores[idx], idx, eps, min_samples
    )
    if not clusters:
        return int(np.argmax(scores)), []
    c0 = clusters[0]
    local = int(np.argmax(c0["scores"]))
    return int(c0["indices"][local]), clusters


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--sam_ckpt", default="ckpts/sam_vit_b_01ec64.pth")
    p.add_argument(
        "--ckpt_root",
        default="ckpt/office0",
        help="Root (office0): subfolders 3,4,8,16,32,64 with best_ckpt.pth",
    )
    p.add_argument(
        "--ae_subdirs",
        default=",".join(DEFAULT_AE_SUBDIRS),
        help=f"Latent subfolders comma-separated (default: {','.join(DEFAULT_AE_SUBDIRS)})",
    )
    p.add_argument(
        "--save_laion_512d",
        action="store_true",
        help="Also save semantic__laion_512d.png (8th image)",
    )
    p.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for SAM+CLIP (default cuda:0).",
    )
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument(
        "--out_dir",
        default=None,
        help="PNG output directory (default: next to the frame, <stem>_encoder_vis/)",
    )
    p.add_argument(
        "--no_save",
        action="store_true",
        help="Do not save images (console log only)",
    )
    p.add_argument(
        "--no_cluster",
        action="store_true",
        help="Do not use DBSCAN on centroids; only global argmax by cos",
    )
    p.add_argument(
        "--mask_top_percentile",
        type=float,
        default=15.0,
        help="Top percent of masks by cos for DBSCAN (like Gaussian top_percentile; default 15)",
    )
    p.add_argument(
        "--dbscan_eps_px",
        type=float,
        default=None,
        help="DBSCAN eps in pixels; if unset — dbscan_eps_frac * min(H,W)",
    )
    p.add_argument(
        "--dbscan_eps_frac",
        type=float,
        default=0.08,
        help="If --dbscan_eps_px unset: eps = frac * min(H,W) (default 0.08)",
    )
    p.add_argument(
        "--dbscan_min_samples",
        type=int,
        default=3,
        help="DBSCAN min_samples for mask centroids (default 3)",
    )
    args = p.parse_args()
    ae_subdirs = _parse_ae_subdirs(args.ae_subdirs)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root = Path(args.ckpt_root).resolve()
    in_path = Path(args.image).resolve()
    if args.no_save:
        out_d: Path | None = None
    elif args.out_dir:
        out_d = Path(args.out_dir).resolve()
    else:
        out_d = in_path.parent / f"{in_path.stem}_encoder_vis"

    bgr = cv2.imread(str(in_path))
    if bgr is None:
        raise SystemExit(f"Cannot read image: {in_path}")
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print("Loading SAM...")
    generator, sam_name = sqm.load_sam_generator(args.sam_ckpt, device)
    print(f"  SAM: {sam_name}")
    masks_all = generator.generate(image_rgb)
    masks_all.sort(key=lambda r: -r["area"])
    n = len(masks_all)
    if n == 0:
        raise SystemExit("SAM returned no masks.")
    print(f"  Masks: {n}\n")

    H, W = bgr.shape[:2]
    eps = args.dbscan_eps_px if args.dbscan_eps_px is not None else float(args.dbscan_eps_frac * min(H, W))
    use_cl = not args.no_cluster
    print(
        f"Mask clustering: {'ON (DBSCAN)' if use_cl else 'OFF'}  "
        f"top_percentile={args.mask_top_percentile}  eps={eps:.1f}px  min_samples={args.dbscan_min_samples}\n"
    )
    centroids_xy = _mask_centroids_xy(masks_all)

    # ----- 1) OpenAI CLIP (original) -----
    print("OpenAI CLIP (ViT-B-16, pretrained=openai)")
    clip_oai, tok_oai, pre_oai = sqm.load_clip("ViT-B-16", "openai", device)
    img_oai = sqm.embed_masks_clip(image_rgb, masks_all, clip_oai, pre_oai, device)
    q_oai = sqm.embed_query(args.query, clip_oai, tok_oai, device, None)
    scores_oai = _cosine_scores(img_oai, q_oai)
    best_oai, cl_oai = pick_best_mask_clustered(
        scores_oai,
        centroids_xy,
        top_percentile=args.mask_top_percentile,
        eps=eps,
        min_samples=args.dbscan_min_samples,
        use_clusters=use_cl,
    )
    max_oai = float(scores_oai[best_oai])
    order_oai = np.argsort(-scores_oai)
    if use_cl and cl_oai:
        c0 = cl_oai[0]
        print(
            f"  [clusters] found {len(cl_oai)}  best_cluster: size={c0['size']}  "
            f"total_cos={c0['total_score']:.4f}  centroid_xy=({c0['centroid'][0]:.0f},{c0['centroid'][1]:.0f})"
        )
    elif use_cl and not cl_oai:
        print("  [clusters] none (noise / fallback) → global argmax")
    print(f"  best_mask_idx={best_oai}  cosine={max_oai:.4f}")
    for rank in range(min(args.top_k, n)):
        j = int(order_oai[rank])
        print(f"    #{rank+1}  idx={j:4d}  cos={scores_oai[j]:.4f}  area={int(masks_all[j]['area'])}")
    if out_d is not None:
        out_d.mkdir(parents=True, exist_ok=True)
        vis = sqm.render_all_masks_and_best(bgr, masks_all, best_oai)
        p_openai = out_d / "semantic__openai_clip.png"
        cv2.imwrite(str(p_openai), vis)
        print(f"  → saved {p_openai}")
    del clip_oai, tok_oai, pre_oai
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ----- 2) Laion CLIP — 512d ranking + embeddings for AE -----
    print("\nLaion CLIP (ViT-B-16, laion2b_s34b_b88k) — 512d and AE from", root)
    clip_lai, tok_lai, pre_lai = sqm.load_clip("ViT-B-16", "laion2b_s34b_b88k", device)
    img_lai = sqm.embed_masks_clip(image_rgb, masks_all, clip_lai, pre_lai, device)
    q_lai = sqm.embed_query(args.query, clip_lai, tok_lai, device, None)
    scores_lai = _cosine_scores(img_lai, q_lai)
    best_lai, cl_lai = pick_best_mask_clustered(
        scores_lai,
        centroids_xy,
        top_percentile=args.mask_top_percentile,
        eps=eps,
        min_samples=args.dbscan_min_samples,
        use_clusters=use_cl,
    )
    max_lai = float(scores_lai[best_lai])
    order_lai = np.argsort(-scores_lai)
    print("  [512d, no compression]")
    if use_cl and cl_lai:
        c0 = cl_lai[0]
        print(
            f"  [clusters] found {len(cl_lai)}  best_cluster: size={c0['size']}  "
            f"total_cos={c0['total_score']:.4f}"
        )
    print(f"  best_mask_idx={best_lai}  cosine={max_lai:.4f}")
    for rank in range(min(args.top_k, n)):
        j = int(order_lai[rank])
        print(f"    #{rank+1}  idx={j:4d}  cos={scores_lai[j]:.4f}  area={int(masks_all[j]['area'])}")
    if out_d is not None and args.save_laion_512d:
        out_d.mkdir(parents=True, exist_ok=True)
        vis_l = sqm.render_all_masks_and_best(bgr, masks_all, best_lai)
        p_lai = out_d / "semantic__laion_512d.png"
        cv2.imwrite(str(p_lai), vis_l)
        print(f"  → saved {p_lai}")

    ckpt_list = _resolve_office0_ae_ckpts(root, ae_subdirs)
    if not ckpt_list:
        print(f"  [warn] No best_ckpt.pth in subfolders {ae_subdirs} under {root}")
    rows: list[tuple[str, int, float]] = []

    for latent_name, ckpt_path in ckpt_list:
        label = f"{latent_name}/best_ckpt.pth"
        try:
            ae = sqm.load_ae(ckpt_path, device)
        except Exception as e:
            print(f"  [skip] {label}: {e}")
            continue
        img_e = ae.encode(img_lai.float())
        q_e = sqm.embed_query(args.query, clip_lai, tok_lai, device, ae)
        scores_e = _cosine_scores(img_e, q_e)
        best_e, cl_e = pick_best_mask_clustered(
            scores_e,
            centroids_xy,
            top_percentile=args.mask_top_percentile,
            eps=eps,
            min_samples=args.dbscan_min_samples,
            use_clusters=use_cl,
        )
        max_e = float(scores_e[best_e])
        rows.append((f"AE latent {latent_name}d", best_e, max_e))
        order_e = np.argsort(-scores_e)
        print(f"\n  AE  latent {latent_name}d  ({ckpt_path})")
        if use_cl and cl_e:
            c0 = cl_e[0]
            print(
                f"      [clusters] found {len(cl_e)}  best_cluster: size={c0['size']}  "
                f"total_cos={c0['total_score']:.4f}"
            )
        print(f"      best_mask_idx={best_e}  cosine={max_e:.4f}")
        for rank in range(min(args.top_k, n)):
            j = int(order_e[rank])
            print(f"      #{rank+1}  idx={j:4d}  cos={scores_e[j]:.4f}  area={int(masks_all[j]['area'])}")

        if out_d is not None:
            out_d.mkdir(parents=True, exist_ok=True)
            vis = sqm.render_all_masks_and_best(bgr, masks_all, best_e)
            p_ae = out_d / f"semantic__ae_{latent_name}.png"
            cv2.imwrite(str(p_ae), vis)
            print(f"      → saved {p_ae}")

        del ae
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary: 1 + 6 methods (Laion only as reference in the log above)
    print("\n--- Summary (1 OpenAI + 6 AE = 7 overlays) ---")
    print(f"{'method':<48} {'best_idx':>8}  {'cos':>8}")
    print(f"{'CLIP OpenAI (512d)':<48} {best_oai:>8}  {max_oai:>8.4f}")
    for label, bi, mx in rows:
        short = label if len(label) <= 46 else "…" + label[-44:]
        print(f"{short:<48} {bi:>8}  {mx:>8.4f}")

    if out_d is not None:
        n_png = 1 + len(rows)
        if args.save_laion_512d:
            n_png += 1
        print(f"\nSaved {n_png} PNG(s) (semantic overlays) → {out_d}")


if __name__ == "__main__":
    main()
