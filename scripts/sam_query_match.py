#!/usr/bin/env python3
"""
SAM: all auto-masks on the image → CLIP embedding per mask → compare to text query.

Uses the same crops / normalization as ``src/semantic/sam_clip_extractor.py`` (LangSplat-compatible).

Examples
-------
CLIP only (512d):

    python scripts/sam_query_match.py \\
        --image path/to/frame.png \\
        --query "a wooden table" \\
        --sam_ckpt ckpts/sam_vit_b_01ec64.pth

With a trained autoencoder (latent, as in ``debug_query.py`` / LangSplat):

    python scripts/sam_query_match.py \\
        --image path/to/frame.png \\
        --query "a sofa" \\
        --sam_ckpt ckpts/sam_vit_b_01ec64.pth \\
        --ae_ckpt ckpt/office0/best_ckpt.pth

``--encoder`` is a convenience alias for ``--ae_ckpt`` (path to AE .pth).

Output: PNG next to the frame (``<image>_sam_query.png``) or ``--out``: all masks
in translucent colors, query-selected mask overlaid in green.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.semantic.language_autoencoder import Autoencoder
from src.semantic.sam_clip_extractor import get_seg_img, pad_img


def _infer_ae_arch(state: dict) -> Tuple[List[int], List[int]]:
    enc_pat = re.compile(r"^encoder\.(\d+)\.weight$")
    dec_pat = re.compile(r"^decoder\.(\d+)\.weight$")
    enc_items: List[Tuple[int, torch.Tensor]] = []
    dec_items: List[Tuple[int, torch.Tensor]] = []
    for k, v in state.items():
        m = enc_pat.match(k)
        if m and isinstance(v, torch.Tensor) and v.ndim == 2:
            enc_items.append((int(m.group(1)), v))
        m = dec_pat.match(k)
        if m and isinstance(v, torch.Tensor) and v.ndim == 2:
            dec_items.append((int(m.group(1)), v))
    enc_items.sort(key=lambda t: t[0])
    dec_items.sort(key=lambda t: t[0])
    if not enc_items or not dec_items:
        raise ValueError("Could not infer AE architecture from checkpoint keys")
    enc_dims = [int(w.shape[0]) for _, w in enc_items]
    dec_dims = [int(w.shape[0]) for _, w in dec_items]
    return enc_dims, dec_dims


def load_ae(ckpt: Path, device: torch.device) -> Autoencoder:
    state = torch.load(str(ckpt), map_location=device)
    enc_dims, dec_dims = _infer_ae_arch(state)
    ae = Autoencoder(enc_dims, dec_dims).to(device)
    ae.load_state_dict(state)
    ae.eval()
    return ae


def load_sam_generator(sam_ckpt: str, device: torch.device):
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    name = Path(sam_ckpt).name
    if "vit_h" in name:
        model_type = "vit_h"
    elif "vit_l" in name:
        model_type = "vit_l"
    else:
        model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_ckpt)
    sam.to(device)
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )
    return generator, model_type


def load_clip(clip_model: str, clip_pretrained: str, device: torch.device):
    import open_clip
    import torchvision

    model, _, _ = open_clip.create_model_and_transforms(
        clip_model, pretrained=clip_pretrained, precision="fp16", device=device
    )
    model.eval()
    tok = open_clip.get_tokenizer(clip_model)
    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    return model, tok, preprocess


@torch.no_grad()
def embed_masks_clip(
    image_rgb: np.ndarray,
    masks: list,
    clip_model,
    preprocess,
    device: torch.device,
) -> torch.Tensor:
    """N × 512, L2-normalized."""
    if not masks:
        return torch.zeros(0, 512, device=device)

    crops = []
    for m in masks:
        seg_img = get_seg_img(m, image_rgb)
        pad = pad_img(seg_img)
        resized = cv2.resize(pad, (224, 224))
        crops.append(resized)

    tensor = torch.from_numpy(np.stack(crops, axis=0).astype(np.float32)).permute(0, 3, 1, 2) / 255.0
    tensor = preprocess(tensor).half().to(device)
    emb = clip_model.encode_image(tensor)
    emb = F.normalize(emb, dim=-1)
    return emb


@torch.no_grad()
def embed_query(
    text: str,
    clip_model,
    tok,
    device: torch.device,
    ae: Autoencoder | None,
) -> torch.Tensor:
    """Single query vector (D,), L2-normalized (CLIP or AE latent)."""
    tokens = tok([text]).to(device)
    q = F.normalize(clip_model.encode_text(tokens), dim=-1)
    if ae is not None:
        # CLIP may be fp16; AE weights are fp32
        q = ae.encode(q.float())[0]
    else:
        q = q[0]
    return q


# Palette without pure green — reserved for the best mask (as in query_language_field.py)
SAM_PALETTE_BGR = [
    (0, 60, 255),
    (0, 165, 255),
    (0, 230, 255),
    (255, 80, 0),
    (180, 0, 255),
    (0, 255, 180),
    (128, 0, 128),
    (255, 255, 0),
    (0, 128, 255),
    (200, 200, 0),
]
GREEN_BGR = (0, 220, 0)
RED_BGR = (0, 0, 255)


def _mask_centroid_xy(mask_u8: np.ndarray) -> tuple[int, int]:
    """Binary mask center of mass (x, y) in pixels."""
    m = cv2.moments(mask_u8, binaryImage=True)
    if m["m00"] > 1e-6:
        return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return 0, 0
    return int(np.round(xs.mean())), int(np.round(ys.mean()))


def _apply_mask_bgr(
    img: np.ndarray,
    mask_u8: np.ndarray,
    color_bgr: tuple,
    alpha: float = 0.4,
    draw_contour: bool = True,
) -> np.ndarray:
    """BGR image + binary uint8 mask, alpha-blend and thin contour."""
    a = (mask_u8 > 0).astype(np.float32)[:, :, None]
    c = np.full_like(img, np.array(color_bgr, dtype=np.uint8))
    out = (img.astype(np.float32) * (1.0 - alpha * a) + c.astype(np.float32) * (alpha * a)).clip(0, 255).astype(np.uint8)
    if draw_contour:
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color_bgr, 2)
    return out


def render_all_masks_and_best(
    bgr: np.ndarray,
    masks_all: list,
    best_idx: int,
    *,
    alpha_other: float = 0.35,
    alpha_best: float = 0.52,
) -> np.ndarray:
    """
    All SAM masks in different colors; best in green; best-mask center as a red point.
    """
    out = bgr.copy()
    for i, m in enumerate(masks_all):
        if i == best_idx:
            continue
        mask2d = (m["segmentation"].astype(np.uint8) * 255)
        col = SAM_PALETTE_BGR[i % len(SAM_PALETTE_BGR)]
        out = _apply_mask_bgr(out, mask2d, col, alpha=alpha_other, draw_contour=True)

    best_u8 = (masks_all[best_idx]["segmentation"].astype(np.uint8) * 255)
    out = _apply_mask_bgr(out, best_u8, GREEN_BGR, alpha=alpha_best, draw_contour=True)

    cx, cy = _mask_centroid_xy(best_u8)
    cv2.circle(out, (cx, cy), 10, RED_BGR, -1)
    cv2.circle(out, (cx, cy), 10, (255, 255, 255), 2)

    return out


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser(description="SAM all masks + CLIP query matching")
    p.add_argument("--image", required=True, help="Path to RGB/BGR image")
    p.add_argument("--query", required=True, help="Text query, e.g. 'a red chair'")
    p.add_argument("--sam_ckpt", default="ckpts/sam_vit_b_01ec64.pth", help="SAM checkpoint")
    p.add_argument(
        "--encoder",
        default=None,
        help="Alias: path to trained Autoencoder .pth (same as --ae_ckpt)",
    )
    p.add_argument(
        "--ae_ckpt",
        default=None,
        help="Optional: compress CLIP 512→latent with AE (must match training)",
    )
    p.add_argument("--clip_model", default="ViT-B-16")
    p.add_argument("--clip_pretrained", default="laion2b_s34b_b88k")
    p.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for SAM+CLIP (default cuda:0).",
    )
    p.add_argument("--top_k", type=int, default=10, help="Print & rank top-K masks by similarity")
    p.add_argument(
        "--out",
        default=None,
        help="Where to save BGR: all masks colored + best in green (default: <image>_sam_query.png)",
    )
    args = p.parse_args()

    ae_path = args.ae_ckpt or args.encoder
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print("Loading SAM...")
    generator, sam_name = load_sam_generator(args.sam_ckpt, device)
    print(f"  SAM: {sam_name}")

    masks_all = generator.generate(image_rgb)
    masks_all.sort(key=lambda r: -r["area"])
    n = len(masks_all)
    print(f"  Masks: {n}")
    if n == 0:
        raise SystemExit("SAM returned no masks.")

    print("Loading CLIP...")
    clip_model, tok, preprocess = load_clip(args.clip_model, args.clip_pretrained, device)
    ae: Autoencoder | None = None
    if ae_path:
        ae = load_ae(Path(ae_path), device)
        print(f"  AE: {ae_path}")

    print("Embedding masks + query...")
    img_emb = embed_masks_clip(image_rgb, masks_all, clip_model, preprocess, device)
    if ae is not None:
        img_emb = ae.encode(img_emb.float())

    q = embed_query(args.query, clip_model, tok, device, ae)
    sim = (img_emb * q).sum(dim=-1)
    scores = sim.detach().cpu().numpy()

    order = np.argsort(-scores)
    top_k = min(args.top_k, n)
    print(f"\nQuery: {args.query!r}\nTop-{top_k}:")
    for rank, j in enumerate(order[:top_k], start=1):
        area = int(masks_all[int(j)]["area"])
        print(f"  {rank:2d}.  idx={int(j):4d}  cos={scores[int(j)]:.4f}  area={area}")

    best_i = int(order[0])
    vis = render_all_masks_and_best(bgr, masks_all, best_i)

    in_path = Path(args.image)
    out_path = Path(args.out) if args.out else (in_path.parent / f"{in_path.stem}_sam_query.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"\nSaved (all masks + best in green): {out_path}")

    print(f"\nBest mask index: {best_i}  (cosine={scores[best_i]:.4f})")


if __name__ == "__main__":
    main()
