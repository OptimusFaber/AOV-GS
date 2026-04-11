#!/usr/bin/env python3
"""
Demo: SAM mask proposals + CLIP text-query selection.

Given an input image and a text prompt (e.g. "a sofa"), the script:
  1) runs SAM to generate all masks
  2) embeds each proposal with CLIP (default: tight mask crop with black bg; optional
     ``--use-boxes``: full bbox crop + padding, background kept)
  3) embeds the text prompt with CLIP
  4) selects the best mask by cosine similarity
  5) saves:
     - overlay image with the selected mask filled in green
     - heatmap image showing per-pixel max similarity score

This script reuses the project implementations from:
  replica_sem_benchmark/eval_clip_sam_systematic.py
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch


_HERE = Path(__file__).resolve().parent          # New-Proj/scripts
_PROJ = _HERE.parent                              # New-Proj/
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))


def _dbscan_merge_masks_from_embeddings(
    image_rgb: np.ndarray,
    masks: list[dict],
    embs_np: np.ndarray,
    scores_masks: np.ndarray,
    dbscan_eps: float,
    device: torch.device,
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
    if scores_masks.shape[0] != N:
        raise ValueError("scores_masks length must match masks length")

    if N == 1:
        m = masks[0]
        emb = torch.from_numpy(embs_np[0].astype(np.float32)).to(device)
        seg = m["segmentation"].astype(bool)
        ys, xs = np.where(seg)
        bbox = (
            [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
            if len(ys) > 0
            else [0, 0, 0, 0]
        )
        return emb.unsqueeze(0), [
            {
                "segmentation": seg,
                "area": int(seg.sum()),
                "score": float(m.get("score", 0.0)),
                "predicted_iou": float(m.get("predicted_iou", m.get("score", 0.0))),
                "bbox": bbox,
                "n_merged": 1,
            }
        ]

    # Step 2: filter top-K masks (removes obvious junk before clustering).
    scores = scores_masks.astype(np.float32, copy=False)
    try:
        thr = float(np.quantile(scores, 0.7))
    except Exception:
        thr = float(np.median(scores))
    keep = scores > thr
    if int(keep.sum()) == 0:
        keep[int(np.argmax(scores))] = True
    embs_np = embs_np[keep]
    scores = scores[keep]
    masks = [m for m, k in zip(masks, keep.tolist()) if k]
    N = len(masks)

    if N == 1:
        m = masks[0]
        emb = torch.from_numpy(embs_np[0].astype(np.float32)).to(device)
        seg = m["segmentation"].astype(bool)
        ys, xs = np.where(seg)
        bbox = (
            [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
            if len(ys) > 0
            else [0, 0, 0, 0]
        )
        return emb.unsqueeze(0), [
            {
                "segmentation": seg,
                "area": int(seg.sum()),
                "score": float(m.get("score", 0.0)),
                "predicted_iou": float(m.get("predicted_iou", m.get("score", 0.0))),
                "bbox": bbox,
                "n_merged": 1,
            }
        ]

    # Step 5: spatial constraint (precompute centers, normalised to [0,1]^2).
    centers = []
    for m in masks:
        seg = m["segmentation"].astype(bool)
        ys, xs = np.where(seg)
        if len(ys) == 0:
            centers.append([0.0, 0.0])
        else:
            centers.append([float(xs.mean()), float(ys.mean())])
    centers = np.asarray(centers, dtype=np.float32)
    centers = centers / np.asarray([[max(W, 1), max(H, 1)]], dtype=np.float32)

    if len(embs_np) > 1:
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

            d_emb = cosine_distances(embs_np).astype(np.float64)
            d_spatial = euclidean_distances(centers).astype(np.float64)
            dists = 0.7 * d_emb + 0.3 * d_spatial
            labels = DBSCAN(eps=dbscan_eps, min_samples=1, metric="precomputed", n_jobs=-1).fit_predict(dists)
        except ImportError:
            labels = np.arange(N, dtype=np.int64)
    else:
        labels = np.zeros(N, dtype=np.int64)

    unique_labels = sorted(set(labels.tolist()))
    cluster_embs_list: list[np.ndarray] = []
    cluster_masks_out: list[dict] = []

    for lbl in unique_labels:
        member_idxs = np.where(labels == lbl)[0].tolist()

        # Step 4: forbid merging weak masks inside the cluster.
        member_idxs = [int(i) for i in member_idxs if float(scores[int(i)]) > 0.2]
        if len(member_idxs) == 0:
            continue
        member_idxs_np = np.asarray(member_idxs, dtype=np.int64)

        # Step 3: weighted embedding merge (stronger masks dominate).
        w = scores[member_idxs_np].astype(np.float64, copy=False)
        w = w / (float(w.sum()) + 1e-8)
        merged_emb = (embs_np[member_idxs_np].astype(np.float64, copy=False) * w[:, None]).sum(0)
        merged_emb = merged_emb.astype(np.float32, copy=False)
        norm = float(np.linalg.norm(merged_emb))
        if norm > 1e-8:
            merged_emb /= norm
        cluster_embs_list.append(merged_emb)

        union_seg = np.zeros((H, W), dtype=bool)
        best_score = 0.0
        for i in member_idxs_np.tolist():
            union_seg |= masks[int(i)]["segmentation"].astype(bool)
            best_score = max(
                best_score,
                float(masks[int(i)].get("predicted_iou", masks[int(i)].get("score", 0.0))),
            )

        area = int(union_seg.sum())
        ys, xs = np.where(union_seg)
        bbox = (
            [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
            if len(ys) > 0
            else [0, 0, 0, 0]
        )
        cluster_masks_out.append(
            {
                "segmentation": union_seg,
                "area": area,
                "score": best_score,
                "predicted_iou": best_score,
                "bbox": bbox,
                "n_merged": int(len(member_idxs_np)),
            }
        )

    if not cluster_embs_list:
        return torch.zeros(0, D, device=device), []

    cluster_emb_tensor = torch.from_numpy(np.stack(cluster_embs_list).astype(np.float32)).to(device)
    return cluster_emb_tensor, cluster_masks_out


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (_PROJ / p).resolve()


def _make_overlay_green(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Return RGB image with mask region filled green (alpha blend)."""
    out = image_rgb.copy()
    m = mask.astype(bool)
    if m.any():
        green = np.zeros_like(out)
        green[..., 1] = 255
        out[m] = (out[m].astype(np.float32) * (1 - alpha) + green[m].astype(np.float32) * alpha).astype(
            np.uint8
        )
    return out


def _mask_bbox_xywh(mask_dict: dict) -> tuple[int, int, int, int]:
    """Return ``(x, y, w, h)`` in image coordinates (same convention as benchmark)."""
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


def _padded_box_crop_for_clip(
    mask_dict: dict,
    image_rgb: np.ndarray,
    *,
    pad: int = 20,
    pad_square_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Rectangular crop from the image using the mask bounding box + ``pad`` pixels per
    side (clamped to image bounds). Background is **not** masked out — CLIP sees the
    full box region. Result is padded to a square like ``_tight_crop`` for the encoder.
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
    return pad_square_fn(crop)


def _scores_to_heatmap_overlay(
    image_rgb: np.ndarray,
    masks: list[dict],
    scores: np.ndarray,
    *,
    alpha: float = 0.55,
    colormap: int = cv2.COLORMAP_JET,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a per-pixel heatmap by taking max score across masks that cover a pixel.
    Returns (heatmap_bgr_uint8, overlay_rgb_uint8).
    """
    h, w = image_rgb.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for m, s in zip(masks, scores.tolist()):
        seg = m["segmentation"].astype(bool)
        if not seg.any():
            continue
        heat[seg] = np.maximum(heat[seg], float(s))

    # Normalise for visualization (robust range)
    if np.isfinite(heat).any() and float(heat.max()) > float(heat.min()):
        lo = float(np.quantile(heat[heat > 0], 0.05)) if np.any(heat > 0) else float(heat.min())
        hi = float(np.quantile(heat, 0.995))
        if hi <= lo:
            lo, hi = float(heat.min()), float(heat.max())
        heat_n = np.clip((heat - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    else:
        heat_n = heat

    heat_u8 = (heat_n * 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, colormap)
    overlay_bgr = cv2.addWeighted(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), 1 - alpha, heat_bgr, alpha, 0.0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return heat_bgr, overlay_rgb


def _short_sam_to_id(short: str, sam_models_root: Path | None, sam_ckpt: str | None) -> str:
    """
    Map short SAM names to ids accepted by load_sam_backend():
      - 'sam1' uses local SAM1 checkpoint (sam_ckpt or project default)
      - 'sam2_tiny|small|base|large' resolved under sam_models_root if provided,
        else uses HF hub id (requires transformers>=4.56).
    """
    s = short.strip().lower()
    if s in ("sam1", "sam1b", "sam_vit_b", "vitb"):
        return "local"
    sam2_map = {
        "sam2_tiny": "facebook/sam2-hiera-tiny",
        "sam2_small": "facebook/sam2-hiera-small",
        "sam2_base": "facebook/sam2-hiera-base-plus",
        "sam2_large": "facebook/sam2-hiera-large",
    }
    if s not in sam2_map:
        raise ValueError(f"Unknown --sam short name: {short!r}. Try: sam1, sam2_tiny, sam2_small, sam2_base, sam2_large")
    hub_id = sam2_map[s]
    if sam_models_root is not None:
        cand = (sam_models_root / hub_id.split("/", 1)[1]).resolve()
        if cand.is_dir():
            return str(cand)
    return hub_id


def _short_clip_to_cfg(short: str) -> tuple[str, str]:
    """
    Map short CLIP names to (model_name, pretrained) used by open_clip.
    These are the best-performing families from your sweep CSV.
    """
    s = short.strip().lower()
    clip_map = {
        "mclip2s2": ("MobileCLIP2-S2", "dfndr2b"),
        "mclip2b": ("MobileCLIP2-B", "dfndr2b"),
        "mclipb": ("MobileCLIP-B", "datacompdr"),
        "mclips2": ("MobileCLIP-S2", "datacompdr"),
        "vitb32": ("ViT-B-32", "laion2b_s34b_b79k"),
        "vith14_meta": ("ViT-H-14", "metaclip_fullcc"),
    }
    if s not in clip_map:
        raise ValueError(
            f"Unknown --clip short name: {short!r}. "
            f"Try one of: {', '.join(sorted(clip_map))}  "
            f"(or use --clip_model/--clip_pretrained)."
        )
    return clip_map[s]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True, help="Input image path (jpg/png).")
    ap.add_argument("--text_query", required=True, help="Text query, e.g. 'a sofa'.")

    ap.add_argument("--sam", default="sam1", help="Short SAM name: sam1, sam2_tiny, sam2_small, sam2_base, sam2_large.")
    ap.add_argument("--sam_models_root", type=Path, default=None, help="Root with local SAM HF snapshots, e.g. /mnt/data/model-ckpts/sam")
    ap.add_argument(
        "--sam_ckpt",
        default="ckpts/sam_vit_h_4b8939.pth",
        help="SAM1 checkpoint path (.pth). Default: ckpts/sam_vit_h_4b8939.pth. "
        "Set to empty string to use project default.",
    )

    ap.add_argument("--clip", default="mclip2s2", help="Short CLIP name, e.g. mclip2s2, mclipb, vitb32.")
    ap.add_argument("--clip_model", default=None, help="Override open_clip model name (advanced).")
    ap.add_argument("--clip_pretrained", default=None, help="Override open_clip pretrained tag (advanced).")

    ap.add_argument("--device", default="cuda:0", help="Torch device (default cuda:0).")
    ap.add_argument("--min_sam_score", type=float, default=0.10, help="Min score threshold during SAM caching/generation.")
    ap.add_argument("--topk", type=int, default=10, help="Also save top-k scores to a text file.")

    ap.add_argument("--out_dir", type=Path, default=Path("replica_sem_benchmark/results/demo_query"), help="Output directory.")
    ap.add_argument("--alpha_mask", type=float, default=0.45, help="Alpha for green mask overlay.")
    ap.add_argument("--alpha_heat", type=float, default=0.55, help="Alpha for heatmap overlay.")

    ap.add_argument(
        "--use-boxes",
        action="store_true",
        help="For CLIP only: embed rectangular crops from mask bboxes (+padding), keeping "
        "background inside the box. Overlays / heatmap still use mask shapes.",
    )
    ap.add_argument(
        "--use-dbscan",
        action="store_true",
        help="Cluster SAM proposals with DBSCAN (cosine distance on CLIP embeddings), "
        "merge masks within each cluster (pixel OR), then select the best cluster for the text query.",
    )
    ap.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.15,
        help="DBSCAN eps for cosine distance on CLIP embeddings. Used with --use-dbscan.",
    )
    ap.add_argument(
        "--box-pad",
        type=int,
        default=20,
        help="Extra pixels to expand each side of the bbox (clamped to image); used with --use-boxes.",
    )
    args = ap.parse_args()

    img_path = _resolve(args.image)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise SystemExit(f"Cannot read image: {img_path}")
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Import the project's implementations (keeps behavior consistent with the benchmark).
    from replica_sem_benchmark.eval_clip_sam_systematic import (  # noqa: WPS433
        ClipBackend,
        _pad_square,
        _tight_crop,
        load_sam_backend,
    )

    sam_models_root = _resolve(args.sam_models_root) if args.sam_models_root is not None else None
    sam_id = _short_sam_to_id(args.sam, sam_models_root, args.sam_ckpt)

    clip_model, clip_pretrained = (
        (args.clip_model, args.clip_pretrained) if (args.clip_model and args.clip_pretrained) else _short_clip_to_cfg(args.clip)
    )

    print(f"SAM:  {args.sam}  -> {sam_id}")
    if args.sam_ckpt:
        print(f"SAM1 ckpt: {args.sam_ckpt}")
    print(f"CLIP: {args.clip} -> {clip_model}/{clip_pretrained}")
    print(f"Device: {device}")
    print(f"Image:  {img_path}")
    print(f"Query:  {args.text_query!r}")
    print(f"CLIP input: {'bbox crops (+' + str(args.box_pad) + 'px pad)' if args.use_boxes else 'tight mask crops (bg black)'}")
    if args.use_dbscan:
        print(f"DBSCAN: enabled (eps={float(args.dbscan_eps):.4f})")

    # 1) SAM masks
    sam_ckpt_arg = None if (args.sam_ckpt is None or str(args.sam_ckpt).strip() == "") else args.sam_ckpt
    sam = load_sam_backend(sam_id, device, sam_ckpt=sam_ckpt_arg, half=True)
    masks = sam.generate(image_rgb, min_score=float(args.min_sam_score))
    if masks:
        masks.sort(key=lambda x: -int(x.get("area", 0)))
    print(f"Generated masks: {len(masks)}")
    sam.unload()

    if not masks:
        raise SystemExit("No masks generated by SAM.")

    # 2) CLIP embeddings (mask crops vs bbox crops — visualization below always uses masks)
    clip = ClipBackend(clip_model, clip_pretrained, device)
    if args.use_boxes:
        crops = [
            _padded_box_crop_for_clip(
                m,
                image_rgb,
                pad=int(args.box_pad),
                pad_square_fn=_pad_square,
            )
            for m in masks
        ]
    else:
        crops = [_tight_crop(m, image_rgb) for m in masks]
    img_emb = clip.embed_crops(crops)                 # (N,D) L2-normalized float32 on device
    txt_emb = clip.embed_texts([args.text_query])     # (1,D) L2-normalized float32 on device
    # Always compute per-mask similarity for heatmap visualization.
    sim_masks = torch.matmul(img_emb, txt_emb.T).squeeze(1)  # (N,)
    scores_masks = sim_masks.detach().float().cpu().numpy()
    masks_raw = masks
    if args.use_dbscan:
        embs_np = img_emb.detach().float().cpu().numpy()
        cluster_embs, cluster_masks = _dbscan_merge_masks_from_embeddings(
            image_rgb,
            masks_raw,
            embs_np,
            scores_masks,
            float(args.dbscan_eps),
            device,
        )
        masks = cluster_masks
        sim = torch.matmul(cluster_embs, txt_emb.T).squeeze(1)  # (C,)
        scores = sim.detach().float().cpu().numpy()
        print(f"DBSCAN clusters: {len(masks)}")
    else:
        scores = scores_masks
    clip.unload()

    best_i = int(np.argmax(scores))
    best_score = float(scores[best_i])
    best_mask = masks[best_i]["segmentation"].astype(bool)
    sel_kind = "cluster" if args.use_dbscan else "mask"
    merged_n = int(masks[best_i].get("n_merged", 1)) if masks else 1
    merged_s = f"  n_merged={merged_n}" if args.use_dbscan else ""
    print(f"Best {sel_kind} idx={best_i}  score={best_score:.4f}  area={int(best_mask.sum())}{merged_s}")

    # Save overlay with selected mask
    overlay = _make_overlay_green(image_rgb, best_mask, alpha=float(args.alpha_mask))
    out_overlay = out_dir / "selected_mask_overlay.png"
    cv2.imwrite(str(out_overlay), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Save heatmap + overlay
    heat_bgr, heat_overlay_rgb = _scores_to_heatmap_overlay(
        image_rgb,
        masks_raw,
        scores_masks,
        alpha=float(args.alpha_heat),
    )
    out_heat = out_dir / "heatmap.png"
    out_heat_overlay = out_dir / "heatmap_overlay.png"
    cv2.imwrite(str(out_heat), heat_bgr)
    cv2.imwrite(str(out_heat_overlay), cv2.cvtColor(heat_overlay_rgb, cv2.COLOR_RGB2BGR))

    # Save a small text report
    order = np.argsort(-scores)
    out_txt = out_dir / "topk_scores.txt"
    k = int(max(1, min(args.topk, len(masks))))
    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"image: {img_path}\\n")
        f.write(f"query: {args.text_query!r}\\n")
        f.write(f"sam_short: {args.sam}\\n")
        f.write(f"sam_id: {sam_id}\\n")
        f.write(f"clip_short: {args.clip}\\n")
        f.write(f"clip: {clip_model}/{clip_pretrained}\\n")
        f.write(f"clip_use_boxes: {bool(args.use_boxes)}  box_pad_px: {int(args.box_pad)}\\n")
        f.write(f"use_dbscan: {bool(args.use_dbscan)}  dbscan_eps: {float(args.dbscan_eps):.6f}\\n")
        f.write(f"num_masks: {len(masks)}\\n")
        f.write(f"best_idx: {best_i}\\n")
        f.write(f"best_score: {best_score:.6f}\\n")
        f.write(f"item_kind: {'cluster' if args.use_dbscan else 'mask'}\\n")
        f.write(f"\\nTop-k {'clusters' if args.use_dbscan else 'masks'} by cosine similarity:\\n")
        for r, idx in enumerate(order[:k], 1):
            m = masks[int(idx)]
            area = int(m.get("area", int(m["segmentation"].sum())))
            n_merged = int(m.get("n_merged", 1))
            merged_txt = f"  n_merged={n_merged}" if args.use_dbscan else ""
            f.write(
                f"{r:2d}. idx={int(idx):4d}  score={float(scores[int(idx)]):.6f}  area={area}{merged_txt}\\n"
            )

    print(f"Saved: {out_overlay}")
    print(f"Saved: {out_heat}")
    print(f"Saved: {out_heat_overlay}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
