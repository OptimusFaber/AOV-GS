#!/usr/bin/env python3
"""
Render RGB + text-query heatmap + GT / predicted masks from a single camera pose.

Output: 2×2 composite PNG:
  top-left     — RGB render
  top-right    — relevancy heatmap + color scale
  bottom-left  — GT binary mask for the query class (Habitat semantics)
  bottom-right — predicted binary mask

Uses LangSplatV2 lang_field(s) with optional s/m/l pyramid (best level by max relevancy).

Example::

    python scripts/render_query_from_pose.py \\
      --checkpoint results/.../splatam/final/params.npz \\
      --result_dir results/Replica/office0/ActiveOpenSem/run_0 \\
      --traj data/replica_sim_nvs/office0/traj.txt \\
      --frame 42 \\
      --align_gs_train_frame \\
      --scene office0 \\
      --levels all \\
      --text "a sofa" \\
      --semantic_mask_thresh 0.50 \\
      --out query_sofa_f42.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPTS))

import lang_field_eval_utils as lfu  # noqa: E402
import query_language_field as qlf  # noqa: E402

_HEAT_P_LOW = 5.0
_HEAT_P_HIGH = 98.0


def _normalize_heatmap(sim_raw: np.ndarray, p_low: float, p_high: float) -> tuple[np.ndarray, float, float]:
    lo = float(np.percentile(sim_raw, p_low))
    hi = float(np.percentile(sim_raw, p_high))
    if hi <= lo + 1e-8:
        norm = np.zeros_like(sim_raw, dtype=np.float32)
    else:
        norm = np.clip((sim_raw.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return norm, lo, hi


def _heatmap_jet_bgr(sim_raw: np.ndarray, p_low: float = _HEAT_P_LOW, p_high: float = _HEAT_P_HIGH) -> np.ndarray:
    norm, _, _ = _normalize_heatmap(sim_raw, p_low, p_high)
    u8 = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def _heatmap_with_colorbar_bgr(
    sim_raw: np.ndarray,
    *,
    p_low: float = _HEAT_P_LOW,
    p_high: float = _HEAT_P_HIGH,
    bar_w: int | None = None,
) -> np.ndarray:
    """Heatmap (JET) with vertical color scale and lo/hi labels."""
    norm, lo, hi = _normalize_heatmap(sim_raw, p_low, p_high)
    H, W = norm.shape[:2]
    heat = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_JET)

    bw = bar_w if bar_w is not None else max(28, W // 24)
    gap = max(4, W // 120)
    bar_h = H
    grad = np.linspace(1.0, 0.0, bar_h, dtype=np.float32)[:, None]
    bar_rgb = cv2.applyColorMap((grad * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    bar = np.repeat(bar_rgb, bw, axis=1)

    fs = max(0.35, min(0.55, H / 900.0))

    pad = np.full((H, gap, 3), 240, dtype=np.uint8)
    panel = np.hstack([heat, pad, bar])

    _put_text(panel, f"{hi:.3f}", (W + gap + 2, int(14 * fs + 10)), font_scale=fs, color=(30, 30, 30))
    _put_text(
        panel,
        f"{lo:.3f}",
        (W + gap + 2, bar_h - 8),
        font_scale=fs,
        color=(30, 30, 30),
    )
    _put_text(
        panel,
        "score",
        (W + gap + 2, bar_h // 2 + 4),
        font_scale=fs * 0.9,
        color=(60, 60, 60),
    )
    return panel


def _letterbox_bgr(
    bgr: np.ndarray,
    target_w: int,
    target_h: int,
    *,
    pad_color: tuple[int, int, int] = (245, 245, 245),
) -> np.ndarray:
    """Fit image into target size without cropping (uniform scale + pad)."""
    h, w = bgr.shape[:2]
    if w <= 0 or h <= 0:
        return np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    scale = min(target_w / float(w), target_h / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    if nw != w or nh != h:
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = bgr
    return canvas


def _put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


def _mask_overlay_bgr(
    rgb_bgr: np.ndarray,
    mask_u8: np.ndarray,
    *,
    tint_bgr=(60, 255, 220),
    empty_label: str | None = None,
) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.float32)[:, :, None]
    if float(m.sum()) < 1.0 and empty_label:
        vis = rgb_bgr.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis, empty_label, (12, 28), font, 0.65, (40, 40, 220), 2, cv2.LINE_AA)
        return vis
    tint = np.full_like(rgb_bgr, np.array(tint_bgr, dtype=np.uint8))
    base = rgb_bgr.astype(np.float32) * (0.35 + 0.65 * m)
    vis = (base * (1 - 0.45 * m) + tint.astype(np.float32) * (0.45 * m)).clip(0, 255).astype(np.uint8)
    cnts, _ = cv2.findContours((mask_u8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, tint_bgr, 2)
    return vis


def _panel_title(img: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    """Draw title bar above a panel (bold labels, fixed height for grid alignment)."""
    bar_h = 54
    banner = np.full((bar_h, img.shape[1], 3), 245, dtype=np.uint8)
    _put_text(banner, title, (8, 26), font_scale=0.62, color=(20, 20, 20))
    if subtitle:
        _put_text(banner, subtitle, (8, 48), font_scale=0.48, color=(70, 70, 70))
    return np.vstack([banner, img])


def _pick_best_level_heatmap(
    model,
    clip_q: torch.Tensor,
    w2c: torch.Tensor,
    lang_field_paths: dict[str, Path],
    active_levels: tuple[str, ...],
    render_kw: dict,
) -> tuple[np.ndarray, str, np.ndarray]:
    """Return (sim_raw, chosen_level, pred_u8) using LangSplat pyramid selection."""
    best_score = -1.0
    best_sim: np.ndarray | None = None
    best_lvl = active_levels[0]
    best_mask: np.ndarray | None = None
    device = render_kw["device"]
    thresh = render_kw["semantic_mask_thresh"]
    large_pool = render_kw["large_pool"]
    smooth_pool = render_kw["smooth_pool"]
    cosine_kw = {
        k: v
        for k, v in render_kw.items()
        if k not in ("semantic_mask_thresh", "large_pool", "smooth_pool")
    }

    for lvl in active_levels:
        model.load_lang_field(lang_field_paths[lvl])
        sim = qlf._render_raw_cosine_map(model, clip_q, w2c=w2c, **cosine_kw)
        fused = lfu.langsplat_fuse_heatmap(sim, large_pool=large_pool, device=device)
        score = float(fused.max().detach().cpu().item())
        if score > best_score:
            best_score = score
            best_sim = sim
            best_lvl = lvl
            best_mask = lfu.langsplat_mask_from_fused(
                fused, thresh=thresh, smooth_pool=smooth_pool,
            )
    assert best_sim is not None and best_mask is not None
    return best_sim, best_lvl, best_mask


def _compose_grid_2x2(
    render_bgr: np.ndarray,
    heat_bgr: np.ndarray,
    gt_bgr: np.ndarray,
    pred_bgr: np.ndarray,
    *,
    query: str,
    chosen_lvl: str,
    thresh: float,
    class_name: str,
    pred_pct: float,
    gt_pct: float | None,
    iou: float | None,
) -> np.ndarray:
    """2×2: [render | heatmap] / [GT | pred] + top banner."""
    content_h = render_bgr.shape[0]
    w_left = render_bgr.shape[1]
    w_right = heat_bgr.shape[1]

    tl = _panel_title(_letterbox_bgr(render_bgr, w_left, content_h), "Render", "RGB view")
    tr = _panel_title(
        _letterbox_bgr(heat_bgr, w_right, content_h),
        "Heatmap",
        f"percentiles p{_HEAT_P_LOW:g}–p{_HEAT_P_HIGH:g}",
    )
    gt_sub = f"class={class_name}"
    if gt_pct is not None:
        gt_sub += f"  area={gt_pct:.1f}%"
    if iou is not None and iou == iou:
        gt_sub += f"  IoU={iou:.3f}"
    bl = _panel_title(_letterbox_bgr(gt_bgr, w_left, content_h), "GT", gt_sub)
    br = _panel_title(
        _letterbox_bgr(pred_bgr, w_right, content_h),
        "Prediction",
        f"level={chosen_lvl}  thresh={thresh:.2f}  area={pred_pct:.1f}%",
    )

    h_top = max(tl.shape[0], tr.shape[0])
    h_bot = max(bl.shape[0], br.shape[0])

    def _pad_h(img: np.ndarray, th: int) -> np.ndarray:
        if img.shape[0] >= th:
            return img
        pad = np.full((th - img.shape[0], img.shape[1], 3), 245, dtype=np.uint8)
        return np.vstack([img, pad])

    tl, tr = _pad_h(tl, h_top), _pad_h(tr, h_top)
    bl, br = _pad_h(bl, h_bot), _pad_h(br, h_bot)

    gap_h = max(6, content_h // 100)
    gap_col = np.full((tl.shape[0], gap_h, 3), 240, dtype=np.uint8)
    gap_col_b = np.full((bl.shape[0], gap_h, 3), 240, dtype=np.uint8)

    top = np.hstack([tl, gap_col, tr])
    bot = np.hstack([bl, gap_col_b, br])
    gap_row = np.full((gap_h, top.shape[1], 3), 240, dtype=np.uint8)
    grid = np.vstack([top, gap_row, bot])

    banner_h = 64
    banner = np.full((banner_h, grid.shape[1], 3), 32, dtype=np.uint8)
    q = query if len(query) <= 120 else query[:117] + "…"
    _put_text(banner, f'query: "{q}"', (10, 26), font_scale=0.58, color=(245, 245, 245))
    _put_text(
        banner,
        "Render | Heatmap (+scale)  /  GT | Prediction",
        (10, 52),
        font_scale=0.46,
        color=(200, 200, 200),
    )
    return np.vstack([banner, grid])


def _resolve_info_semantic(args: argparse.Namespace) -> Path | None:
    if args.info_semantic:
        p = Path(args.info_semantic).expanduser().resolve()
        return p if p.is_file() else None
    if not args.scene:
        return None
    scene_prefix = args.scene[:-1]
    scene_idx = args.scene[-1]
    cand = _ROOT / "data/replica_v1" / f"{scene_prefix}_{scene_idx}" / "habitat" / "info_semantic.json"
    return cand if cand.is_file() else None


def _load_gt_mask_for_frame(
    frame_id: int,
    class_name: str,
    name_to_id: dict[str, int],
    results_habitat: Path,
    H: int,
    W: int,
) -> tuple[np.ndarray | None, str]:
    sem_index = dict(lfu.load_frame_sem_pairs(results_habitat))
    if frame_id not in sem_index:
        return None, f"no semantic_map for frame {frame_id}"
    try:
        cid = lfu.resolve_class_id(class_name, name_to_id)
    except ValueError as e:
        return None, str(e)
    sem_np = np.load(str(sem_index[frame_id])).astype(np.int64)
    if sem_np.ndim == 3:
        sem_np = sem_np.squeeze()
    if sem_np.shape[:2] != (H, W):
        sem_np = cv2.resize(sem_np.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int64)
    gt_u8 = (sem_np == int(cid)).astype(np.uint8)
    return gt_u8, ""


def _class_name_from_query(text: str) -> str:
    q = text.strip().lower()
    for pref in ("a ", "an ", "the "):
        if q.startswith(pref):
            q = q[len(pref) :]
            break
    return q.strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render 2×2 query viz: render, heatmap, GT, prediction.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--result_dir", required=True, help="Used to auto-resolve lang_field_{s,m,l}k*.pt")
    p.add_argument("--lang_field_s", default=None)
    p.add_argument("--lang_field_m", default=None)
    p.add_argument("--lang_field_l", default=None)
    p.add_argument("--codebook_size", type=int, default=64)
    p.add_argument("--vq_layer_num", type=int, default=1)
    p.add_argument("--levels", default="all", help="'all' or comma-separated s,m,l")
    p.add_argument("--traj", required=True, help="traj.txt (c2w, replica_sim_nvs)")
    p.add_argument("--frame", type=int, required=True, help="Row index in traj.txt")
    p.add_argument("--align_gs_train_frame", action="store_true")
    p.add_argument("--replica_train_traj", type=Path, default=None)
    p.add_argument("--scene", default=None, help="Replica scene id for train traj + info_semantic")
    p.add_argument("--results_habitat", default=None, help="Override .../results_habitat (default: traj parent)")
    p.add_argument("--info_semantic", default=None, help="Replica info_semantic.json for GT class ids")
    p.add_argument("--text", required=True, help='Text query, e.g. "a sofa"')
    p.add_argument("--semantic_mask_thresh", type=float, default=0.5)
    p.add_argument("--semantic_mask_large_pool", type=int, default=29)
    p.add_argument("--semantic_mask_smooth_pool", type=int, default=7)
    p.add_argument("--heatmap_blur", type=float, default=3.0)
    p.add_argument("--negative_texts", default="object,things,stuff,texture")
    p.add_argument("--negative_weight", type=float, default=0.35)
    p.add_argument("--negative_score_mode", default="softmax_pair")
    p.add_argument("--softmax_inv_temp", type=float, default=10.0)
    p.add_argument("--clip_model", default="ViT-B-16")
    p.add_argument("--clip_pretrained", default="laion2b_s34b_b88k")
    p.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for rendering and CLIP (default cuda:0).",
    )
    p.add_argument("--out", required=True, help="Output PNG path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    active_levels = lfu.parse_levels(args.levels)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result_dir = Path(args.result_dir).expanduser().resolve()

    lang_field_paths = lfu.resolve_lang_field_paths(
        result_dir,
        levels=active_levels,
        codebook_size=int(args.codebook_size),
        vq_layer_num=int(args.vq_layer_num),
        lang_field_s=args.lang_field_s,
        lang_field_m=args.lang_field_m,
        lang_field_l=args.lang_field_l,
    )
    for lvl, pth in lang_field_paths.items():
        if not pth.is_file():
            raise FileNotFoundError(pth)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    traj_txt = Path(args.traj).expanduser().resolve()

    train0 = None
    if args.align_gs_train_frame:
        rtp = args.replica_train_traj
        if rtp is None and args.scene:
            cand = _ROOT / "data" / "Replica" / args.scene / "traj.txt"
            if cand.is_file():
                rtp = cand
        if rtp is None or not Path(rtp).expanduser().is_file():
            raise SystemExit("--align_gs_train_frame: provide --replica_train_traj or --scene")
        train0 = lfu.first_c2w_from_traj_file(Path(rtp))

    poses = lfu.poses_from_traj(traj_txt, "c2w", device, c2w_train0=train0)
    if args.frame not in poses:
        raise SystemExit(f"frame {args.frame} out of range [0, {max(poses.keys())}]")
    w2c = poses[args.frame]

    first_lvl = active_levels[0]
    latent_dim = qlf.infer_latent_dim(lang_field_paths[first_lvl])
    model = qlf.LangSplatam(checkpoint_path=str(checkpoint), latent_dim=latent_dim, device=str(device))
    model.load_lang_field(lang_field_paths[first_lvl])
    H = int(model.params["org_height"])
    W = int(model.params["org_width"])

    neg_texts = [x.strip() for x in args.negative_texts.split(",") if x.strip()]
    neg_emb = None
    if neg_texts:
        neg_emb = qlf.encode_query_clip_batch(neg_texts, args.clip_model, args.clip_pretrained, device)

    clip_q = qlf.encode_query_clip(args.text, args.clip_model, args.clip_pretrained, device)

    render_kw = dict(
        ae=None,
        negative_clip_queries=neg_emb,
        negative_weight=float(args.negative_weight),
        negative_mode="max",
        negative_relu_floor=False,
        H=H,
        W=W,
        device=device,
        blur_sigma=float(args.heatmap_blur),
        use_v2=True,
        negative_score_mode=str(args.negative_score_mode),
        softmax_inv_temp=float(args.softmax_inv_temp),
        semantic_mask_thresh=float(args.semantic_mask_thresh),
        large_pool=int(args.semantic_mask_large_pool),
        smooth_pool=int(args.semantic_mask_smooth_pool),
    )

    rgb_bgr = qlf.render_rgb(model, w2c, H, W)
    sim_raw, chosen_lvl, pred_u8 = _pick_best_level_heatmap(
        model, clip_q, w2c, lang_field_paths, active_levels, render_kw,
    )

    heat_bgr = _heatmap_with_colorbar_bgr(sim_raw)
    pred_bgr = _mask_overlay_bgr(rgb_bgr, pred_u8, tint_bgr=(60, 255, 220))
    pred_pct = 100.0 * float((pred_u8 > 0).sum()) / float(H * W)

    class_name = _class_name_from_query(args.text)
    gt_u8: np.ndarray | None = None
    iou: float | None = None
    gt_pct: float | None = None
    gt_note = ""

    info_path = _resolve_info_semantic(args)
    if args.results_habitat:
        hab_dir = Path(args.results_habitat).expanduser().resolve()
    else:
        hab_dir = traj_txt.parent / "results_habitat"

    if info_path is not None and hab_dir.is_dir():
        name_to_id = lfu.load_name_to_id(info_path)
        gt_u8, gt_note = _load_gt_mask_for_frame(
            args.frame, class_name, name_to_id, hab_dir, H, W,
        )
        if gt_u8 is not None:
            iou = lfu.binary_iou(pred_u8 > 0, gt_u8 > 0)
            gt_pct = 100.0 * float((gt_u8 > 0).sum()) / float(H * W)
            gt_bgr = _mask_overlay_bgr(rgb_bgr, gt_u8, tint_bgr=(80, 200, 80))
            print(f"  GT area={gt_pct:.1f}%  pred area={pred_pct:.1f}%  IoU={iou:.4f}")
        else:
            gt_bgr = _mask_overlay_bgr(
                rgb_bgr, np.zeros((H, W), dtype=np.uint8), empty_label=f"GT: {gt_note}",
            )
            print(f"  [warn] GT unavailable: {gt_note}")
    else:
        gt_bgr = _mask_overlay_bgr(
            rgb_bgr,
            np.zeros((H, W), dtype=np.uint8),
            empty_label="GT: set --scene or --info_semantic",
        )
        print("  [warn] GT skipped: missing info_semantic or results_habitat")

    canvas = _compose_grid_2x2(
        rgb_bgr,
        heat_bgr,
        gt_bgr,
        pred_bgr,
        query=args.text,
        chosen_lvl=chosen_lvl,
        thresh=float(args.semantic_mask_thresh),
        class_name=class_name,
        pred_pct=pred_pct,
        gt_pct=gt_pct,
        iou=iou,
    )

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"Saved -> {out_path}")
    print(f"  frame={args.frame}  level={chosen_lvl}  mask_area={pred_pct:.1f}%  query={args.text!r}")


if __name__ == "__main__":
    main()
