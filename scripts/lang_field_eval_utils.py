"""Shared helpers for lang-field traj validation and pose-based rendering."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import query_language_field as qlf

ALL_LEVELS: tuple[str, ...] = ("s", "m", "l")
LEVEL_TO_CKPT_SUFFIX: dict[str, int] = {"s": 1, "m": 2, "l": 3}

# Matches ``val-results-sml-levels/`` (s/m/l pyramid, best thresh 0.55 in sweep).
EVAL_PRESET_VAL_RESULTS_SML_LEVELS = "val_results_sml_levels"


def apply_eval_preset(args) -> str | None:
    """
    Apply a named evaluation preset onto ``args`` (mutates in place).

    Returns preset name if applied, else ``None``.
    """
    preset = str(getattr(args, "eval_preset", "none") or "none").strip()
    if preset in ("none", ""):
        return None
    if preset != EVAL_PRESET_VAL_RESULTS_SML_LEVELS:
        raise ValueError(
            f"Unknown eval_preset {preset!r}; use 'none' or {EVAL_PRESET_VAL_RESULTS_SML_LEVELS!r}",
        )

    args.levels = "all"
    args.semantic_mask_thresh = 0.55
    args.class_name_replace_hyphen_with = None
    args.negative_from_other_classes = False
    args.negative_texts = "object,things,stuff,texture"
    args.negative_weight = 0.35
    args.negative_mode = "max"
    args.negative_score_mode = "softmax_pair"
    args.softmax_inv_temp = 10.0
    args.semantic_mask_large_pool = 29
    args.semantic_mask_smooth_pool = 7
    args.heatmap_blur = 3.0
    args.text_template = "a {class_name}"
    args.align_gs_train_frame = True
    args.no_localization = False
    args.traj_format = "c2w"
    args.min_gt_pixels = 1
    args.codebook_size = 64
    args.vq_layer_num = 1
    args.void_class_ids = "0"
    return preset


def parse_levels(spec: str) -> tuple[str, ...]:
    """Parse ``all`` or comma-separated subset of s,m,l (order preserved: s→m→l)."""
    s = str(spec).strip().lower()
    if s == "all":
        return ALL_LEVELS
    parts = [p.strip() for p in s.split(",") if p.strip()]
    bad = [p for p in parts if p not in ALL_LEVELS]
    if bad:
        raise ValueError(f"Invalid --levels entries {bad!r}; use 'all' or comma-separated s,m,l")
    if not parts:
        raise ValueError("Empty --levels")
    return tuple(lvl for lvl in ALL_LEVELS if lvl in parts)


def _num_key(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else -1


def load_frame_sem_pairs(results_habitat: Path) -> list[tuple[int, Path]]:
    sem_dir = results_habitat / "semantic"
    sem_files = {_num_key(p): p for p in sorted(sem_dir.glob("semantic_map_*.npy"), key=_num_key)}
    return sorted(((k, sem_files[k]) for k in sem_files.keys()), key=lambda t: t[0])


def load_name_to_id(info_semantic_path: Path) -> dict[str, int]:
    data = json.loads(info_semantic_path.read_text(encoding="utf-8"))
    m: dict[str, int] = {}
    for c in data["classes"]:
        name = c["name"].strip().lower()
        cid = int(c["id"])
        m[name] = cid
        m[name.replace("-", " ")] = cid
    return m


def load_id_to_canonical_name(info_semantic_path: Path) -> dict[int, str]:
    data = json.loads(info_semantic_path.read_text(encoding="utf-8"))
    return {int(c["id"]): str(c["name"]).strip() for c in data["classes"]}


def resolve_class_id(class_name: str, name_to_id: dict[str, int]) -> int:
    q = class_name.strip().lower()
    for pref in ("a ", "an ", "the "):
        if q.startswith(pref):
            q = q[len(pref) :]
            break
    q = q.strip()
    if q in name_to_id:
        return name_to_id[q]
    qh = q.replace(" ", "-")
    if qh in name_to_id:
        return name_to_id[qh]
    raise ValueError(f"Class not found in info_semantic: {class_name!r}")


def first_c2w_from_traj_file(path: Path) -> np.ndarray:
    arr = np.loadtxt(str(path.expanduser()), dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(4, 4)
    return arr[0].reshape(4, 4)


def poses_from_traj(
    path: Path,
    fmt: str,
    device: torch.device,
    *,
    c2w_train0: np.ndarray | None,
) -> dict[int, torch.Tensor]:
    arr = np.loadtxt(str(path.expanduser()), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != 16:
        raise ValueError(f"Expected 16 floats per line in {path}")
    mats = arr.reshape(-1, 4, 4).astype(np.float64)
    out: dict[int, torch.Tensor] = {}
    for i in range(mats.shape[0]):
        if fmt == "c2w":
            m_np = mats[i]
            if c2w_train0 is not None:
                m_np = qlf.w2c_gaussian_frame_from_replica_c2w(m_np, c2w_train0)
            else:
                m_np = np.linalg.inv(m_np) @ mats[0]
        elif fmt == "w2c":
            m_np = mats[i] @ np.linalg.inv(mats[0])
        else:
            raise ValueError(f"Unknown traj_format: {fmt!r}")
        out[i] = torch.tensor(m_np.astype(np.float32), dtype=torch.float32, device=device)
    return out


def discover_classes_from_semantics(
    sem_paths_by_fid: dict[int, Path],
    frame_ids: list[int],
    id_to_name: dict[int, str],
    *,
    exclude_ids: frozenset[int],
) -> list[str]:
    seen: set[int] = set()
    for fid in frame_ids:
        p = sem_paths_by_fid.get(fid)
        if p is None or not p.is_file():
            continue
        sem = np.load(str(p)).astype(np.int64)
        if sem.ndim == 3:
            sem = sem.squeeze()
        for uid in np.unique(sem):
            ui = int(uid)
            if ui in exclude_ids or ui not in id_to_name:
                continue
            seen.add(ui)
    return sorted({id_to_name[i].strip().lower() for i in seen}, key=lambda x: x)


def slug(name: str) -> str:
    return qlf._safe_filename_fragment(name.replace(" ", "_"), max_len=48)


def fmt_metric(v: float) -> str:
    if v != v:
        return "nan"
    return f"{v:.4f}"


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b > 0 else float("nan")


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return float("nan")
    return float(inter) / float(union)


def gt_aabb_from_mask(mask_hw: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask_hw)
    if ys.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def centroid_in_box(cx: float, cy: float, box: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= cx <= x1 and y0 <= cy <= y1


def format_query_class_name(class_name: str, *, replace_hyphen_with: str | None) -> str:
    cn = class_name
    if replace_hyphen_with is not None:
        cn = cn.replace("-", replace_hyphen_with)
    return cn


def build_text_query(
    class_name: str,
    template: str,
    *,
    replace_hyphen_with: str | None,
) -> str:
    return template.format(
        class_name=format_query_class_name(class_name, replace_hyphen_with=replace_hyphen_with),
    )


def resolve_lang_field_paths(
    result_dir: Path,
    *,
    levels: tuple[str, ...],
    codebook_size: int,
    vq_layer_num: int,
    lang_field_s: str | None = None,
    lang_field_m: str | None = None,
    lang_field_l: str | None = None,
) -> dict[str, Path]:
    explicit = {"s": lang_field_s, "m": lang_field_m, "l": lang_field_l}
    out: dict[str, Path] = {}
    for lvl in levels:
        if explicit[lvl]:
            p = Path(explicit[lvl]).expanduser().resolve()
        else:
            p = (result_dir / f"lang_field_{lvl}k{codebook_size}_l{vq_layer_num}" / "lang_field.pt").resolve()
        out[lvl] = p
    return out


def langsplat_fuse_heatmap(
    heat_hw: np.ndarray,
    *,
    large_pool: int,
    device: torch.device,
) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(heat_hw, dtype=np.float32)).to(device)
    t = t.unsqueeze(0).unsqueeze(0)
    lk = int(large_pool)
    lk = lk if lk % 2 == 1 else lk + 1
    lp = lk // 2
    pooled = F.avg_pool2d(t, kernel_size=lk, stride=1, padding=lp, count_include_pad=False)
    return (0.5 * (pooled + t)).squeeze(0).squeeze(0)


def langsplat_score_map_from_fused(fused_hw: torch.Tensor) -> np.ndarray:
    """Per-pixel relevancy in [0, 1] (LangSplatV2 normalize step, before threshold)."""
    plane = fused_hw
    mn = plane.min()
    span = plane.max() - mn + 1e-9
    out = torch.clamp((plane - mn) / span, 0.0, 1.0)
    return out.detach().cpu().numpy()


def pick_best_level_fused(
    heatmaps: list[np.ndarray],
    level_order: tuple[str, ...],
    *,
    large_pool: int,
    device: torch.device,
) -> tuple[np.ndarray, int, str]:
    """Return normalized score map for the SAM level with highest global max."""
    if len(heatmaps) != len(level_order):
        raise ValueError(f"expected {len(level_order)} heatmaps, got {len(heatmaps)}")
    best_score = -1.0
    best_map: np.ndarray | None = None
    best_idx = 0
    best_lvl = level_order[0]
    for idx, (heat, lvl) in enumerate(zip(heatmaps, level_order)):
        fused = langsplat_fuse_heatmap(heat, large_pool=large_pool, device=device)
        score = float(fused.max().detach().cpu().item())
        if score > best_score:
            best_score = score
            best_map = langsplat_score_map_from_fused(fused)
            best_idx = idx
            best_lvl = lvl
    assert best_map is not None
    return best_map, best_idx, best_lvl


def load_frame_rgb_path(results_habitat: Path, frame_id: int) -> Path | None:
    """``results_habitat/frame{fid:06d}.jpg`` if present."""
    p = results_habitat / f"frame{frame_id:06d}.jpg"
    return p if p.is_file() else None


def load_eval_rendered_rgb_path(
    result_dir: Path,
    frame_id: int,
    *,
    eval_stage: str = "eval_final",
) -> Path | None:
    """``splatam/eval_<stage>/rendered_rgb/gs_{fid:04d}.png`` if present."""
    suffix = eval_stage[5:] if eval_stage.startswith("eval_") else eval_stage
    p = result_dir / "splatam" / f"eval_{suffix}" / "rendered_rgb" / f"gs_{frame_id:04d}.png"
    return p if p.is_file() else None


def load_pseudo_rgb_np(
    *,
    frame_id: int,
    source: str,
    results_habitat: Path,
    result_dir: Path,
    model: qlf.LangSplatam,
    w2c: torch.Tensor,
    H: int,
    W: int,
    eval_stage: str = "eval_final",
) -> np.ndarray | None:
    """
    RGB uint8 array for pseudo-label generation (``miou_p``).

    ``rendered``: SplaTAM RGB at ``w2c`` (same rasterizer as lang-field eval).
    ``eval_cache``: pre-rendered PNG from ``run_nvs_validation`` / ``eval_final``.
    ``habitat_gt``: GT RGB from ``results_habitat/frame*.jpg``.
    """
    src = str(source).strip().lower()
    if src == "rendered":
        bgr = qlf.render_rgb(model, w2c, H, W)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if src == "eval_cache":
        rgb_path = load_eval_rendered_rgb_path(result_dir, frame_id, eval_stage=eval_stage)
    elif src == "habitat_gt":
        rgb_path = load_frame_rgb_path(results_habitat, frame_id)
    else:
        raise ValueError(f"Unknown pseudo_rgb_source {source!r}; use rendered, eval_cache, or habitat_gt")

    if rgb_path is None:
        return None
    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        return None
    rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb_np.shape[:2] != (H, W):
        rgb_np = cv2.resize(rgb_np, (W, H), interpolation=cv2.INTER_LINEAR)
    return rgb_np


def calc_miou_seg(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean IoU over classes present in ``target`` (non-zero), ActiveSem-style."""
    pred_flat = pred.reshape(-1).long()
    target_flat = target.reshape(-1).long().to(pred_flat.device)
    valid_mask = target_flat != 0
    if not bool(valid_mask.any()):
        return 0.0
    pred_v = pred_flat[valid_mask]
    target_v = target_flat[valid_mask]
    classes = torch.unique(torch.cat((pred_v, target_v)))
    classes = classes[classes != 0]
    if classes.numel() == 0:
        return 0.0
    ious: list[torch.Tensor] = []
    for cls in classes:
        pred_cls = pred_v == cls
        true_cls = target_v == cls
        inter = (pred_cls & true_cls).sum().float()
        union = (pred_cls | true_cls).sum().float()
        if union == 0:
            ious.append(torch.tensor(float("nan"), device=pred_flat.device))
        else:
            ious.append(inter / union)
    return float(torch.nanmean(torch.stack(ious)).item())


def post_process_seg(pred_logits: torch.Tensor, target_id: torch.Tensor) -> torch.Tensor:
    """Remap logits to GT-present classes only (``post_precess_seg`` in eval_helper)."""
    orig_shape = target_id.shape
    target_flat = target_id.reshape(-1).to(pred_logits.device)
    candidate_id = torch.unique(target_flat)
    c = pred_logits.shape[-1]
    valid_mask = (candidate_id >= 0) & (candidate_id < c)
    candidate_id = candidate_id[valid_mask]
    if candidate_id.numel() == 0:
        return torch.zeros(orig_shape, device=pred_logits.device, dtype=target_id.dtype)
    post_logits = pred_logits[..., candidate_id]
    closest_indices = torch.argmax(post_logits, dim=-1)
    post_id = candidate_id[closest_indices]
    return post_id.reshape(orig_shape)


def build_id_indexed_logits(
    score_by_class: dict[str, np.ndarray],
    class_name_to_id: dict[str, int],
) -> tuple[torch.Tensor, list[int]]:
    """Stack per-class score maps into ``(H, W, max_id+1)`` logits tensor."""
    if not score_by_class:
        raise ValueError("empty score_by_class")
    class_ids = sorted({int(class_name_to_id[cn]) for cn in score_by_class})
    max_id = max(class_ids)
    any_map = next(iter(score_by_class.values()))
    h, w = any_map.shape[:2]
    logits = torch.zeros((h, w, max_id + 1), dtype=torch.float32)
    for cn, smap in score_by_class.items():
        cid = int(class_name_to_id[cn])
        t = torch.from_numpy(np.asarray(smap, dtype=np.float32))
        if t.shape[:2] != (h, w):
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False,
            ).squeeze()
        logits[..., cid] = torch.maximum(logits[..., cid], t)
    return logits, class_ids


def logits_to_pred_ids(logits: torch.Tensor) -> torch.Tensor:
    """Argmax over class-id channels; ties → smallest id."""
    return logits.argmax(dim=-1).long()


def write_semantic_result_txt(out_dir: Path, metrics: dict[str, float]) -> Path:
    """Write ``semantic_result.txt`` compatible with SemSplaTAM eval."""
    path = out_dir / "semantic_result.txt"
    lines = [f"{k}: {v}\n" for k, v in metrics.items()]
    path.write_text("".join(lines), encoding="utf-8")
    return path


def langsplat_mask_from_fused(
    fused_hw: torch.Tensor,
    *,
    thresh: float,
    smooth_pool: int,
) -> np.ndarray:
    plane = fused_hw
    mn = plane.min()
    span = plane.max() - mn + 1e-9
    out = torch.clamp((plane - mn) / span * 2.0 - 1.0, 0.0, 1.0)
    mask = (out > float(thresh)).to(torch.uint8)
    sk = int(smooth_pool)
    sk = sk if sk % 2 == 1 else sk + 1
    sp = sk // 2
    sm = F.avg_pool2d(
        mask.float().unsqueeze(0).unsqueeze(0),
        kernel_size=sk,
        stride=1,
        padding=sp,
        count_include_pad=False,
    )
    return (sm > 0.5).to(torch.uint8).squeeze(0).squeeze(0).detach().cpu().numpy()


def write_miou_summary(
    out_dir: Path,
    *,
    experiment_name: str,
    summary: dict[str, Any],
    per_pair: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """Write miou_summary.txt and miou_per_class.csv; return paths."""
    class_ious: dict[str, list[float]] = defaultdict(list)
    for p in per_pair:
        if p.get("iou") in ("", None):
            continue
        class_ious[p["class_name"]].append(float(p["iou"]))

    rows = [(c, float(np.mean(vs)), len(vs)) for c, vs in class_ious.items()]
    rows.sort(key=lambda x: x[1], reverse=True)
    miou = float(summary.get("mIoU_mean_over_pairs", float("nan")))

    txt_path = out_dir / "miou_summary.txt"
    csv_path = out_dir / "miou_per_class.csv"

    lines = [
        f"Experiment: {experiment_name}",
        f"semantic_mask_thresh: {summary.get('semantic_mask_thresh', '—')}",
        f"levels: {summary.get('levels', '—')}",
        f"pairs_evaluated: {summary.get('pairs_evaluated', len(per_pair))}",
        f"frames_evaluated: {summary.get('frames_evaluated', '—')}",
        f"classes: {len(rows)}",
        "",
        f"Overall mIoU (mean over pairs): {miou:.6f}",
        f"precision_macro: {summary.get('precision_macro_mean_pairs', float('nan')):.6f}",
        f"recall_macro:    {summary.get('recall_macro_mean_pairs', float('nan')):.6f}",
        f"f1_macro:        {summary.get('f1_macro_mean_pairs', float('nan')):.6f}",
        "",
        "Per-class mIoU (descending):",
        f"{'#':>3}  {'class':24s}  {'mIoU':>10s}  {'n_pairs':>8s}",
        "-" * 52,
    ]
    for i, (c, cm, n) in enumerate(rows, 1):
        lines.append(f"{i:3d}  {c:24s}  {cm:10.6f}  {n:8d}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    import csv

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "class_name", "miou_mean_over_pairs", "n_pairs"])
        for i, (c, cm, n) in enumerate(rows, 1):
            w.writerow([i, c, f"{cm:.6f}", n])

    return txt_path, csv_path


def load_oneformer_replica(
    *,
    device: torch.device,
    class_info_file: Path,
    oneformer_checkpoint: str,
    ade20k_checkpoint: str,
):
    """Load finetuned OneFormer for Replica (same stack as SemSplaTAM)."""
    from transformers import AutoModelForUniversalSegmentation, AutoProcessor

    from src.data.finetune_oneformer_ReplicaV2 import modify_metadata

    processor = AutoProcessor.from_pretrained(ade20k_checkpoint)
    model = AutoModelForUniversalSegmentation.from_pretrained(
        oneformer_checkpoint, is_training=False,
    ).to(device)
    processor.image_processor.num_text = (
        model.config.num_queries - model.config.text_encoder_n_ctx
    )
    modify_metadata(class_info_file=str(class_info_file), processor=processor)
    return processor, model


@torch.inference_mode()
def oneformer_pseudo_for_rgb(
    rgb_np: np.ndarray,
    *,
    processor,
    model,
    device: torch.device,
    num_classes: int = 102,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(H,W)`` pseudo ids and ``(H,W,C)`` logits on CPU."""
    from PIL import Image

    from src.slam.semsplatam.modified_ver.semantic.oneformer import oneformer_segmentation

    pil = Image.fromarray(rgb_np.astype(np.uint8))
    seg_list, log_list = oneformer_segmentation(
        pil, processor, model, device, num_classes=num_classes,
    )
    pseudo = seg_list[0].detach().cpu().long()
    logits = log_list[0].detach().cpu().float()
    return pseudo, logits


@torch.inference_mode()
def sam_clip_pseudo_for_rgb(
    rgb_np: np.ndarray,
    *,
    class_names: list[str],
    text_queries: list[str],
    class_ids: list[int],
    sam_ckpt: Path,
    clip_model: str,
    clip_pretrained: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-pixel pseudo labels via SAM masks + CLIP text matching."""
    import sam_query_match as sqm

    masks_all = sqm.load_sam_generator(sam_ckpt, device)[0].generate(rgb_np)
    masks_all.sort(key=lambda r: -r["area"])
    h, w = rgb_np.shape[:2]
    max_id = max(class_ids) if class_ids else 0
    logits = torch.zeros((h, w, max_id + 1), dtype=torch.float32)
    if not masks_all:
        return torch.zeros((h, w), dtype=torch.long), logits

    clip_model_obj, _, preprocess = sqm.load_clip(clip_model, clip_pretrained, device)
    img_emb = sqm.embed_masks_clip(rgb_np, masks_all, clip_model_obj, preprocess, device)
    import query_language_field as qlf

    text_emb = qlf.encode_query_clip_batch(text_queries, clip_model, clip_pretrained, device)
    sim = torch.matmul(img_emb.float(), text_emb.float().T)
    best_mask_idx = sim.argmax(dim=0).cpu().numpy()

    for ci, cid in enumerate(class_ids):
        mi = int(best_mask_idx[ci])
        score = float(sim[mi, ci].item())
        mask = masks_all[mi]["segmentation"].astype(bool)
        cid_i = int(cid)
        logits[:, :, cid_i][mask] = torch.maximum(
            logits[:, :, cid_i][mask],
            torch.tensor(score, dtype=torch.float32),
        )
    pseudo = logits.argmax(dim=-1).long()
    return pseudo, logits


def compute_frame_semsplatam_metrics(
    *,
    frame_score_maps: dict[int, dict[str, np.ndarray]],
    sem_by_fid: dict[int, np.ndarray],
    class_name_to_id: dict[str, int],
    frame_ids: list[int],
    pseudo_by_fid: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """
    Aggregate per-frame mIoU_g / mIoU_g_curr and optional mIoU_p / mIoU_p_curr.

    ``frame_score_maps[fid][class_name]`` = LangSplat relevancy score map (H,W) in [0,1].
    """
    miou_g_list: list[float] = []
    miou_g_curr_list: list[float] = []
    miou_p_list: list[float] = []
    miou_p_curr_list: list[float] = []
    per_frame: list[dict[str, Any]] = []

    for fid in frame_ids:
        gt_np = sem_by_fid[fid]
        gt = torch.from_numpy(gt_np.astype(np.int64))
        scores = frame_score_maps.get(fid, {})
        if not scores:
            continue

        cn_to_id = {cn: int(class_name_to_id[cn]) for cn in scores}
        logits_g, _ = build_id_indexed_logits(scores, cn_to_id)
        pred_g = logits_to_pred_ids(logits_g)
        miou_g = calc_miou_seg(pred_g, gt)
        pred_g_curr = post_process_seg(logits_g, gt)
        miou_g_curr = calc_miou_seg(pred_g_curr, gt)
        miou_g_list.append(miou_g)
        miou_g_curr_list.append(miou_g_curr)

        row: dict[str, Any] = {
            "frame_id": fid,
            "miou_g": round(miou_g * 100.0, 6),
            "miou_g_curr": round(miou_g_curr * 100.0, 6),
        }

        if pseudo_by_fid is not None and fid in pseudo_by_fid:
            pseudo, pseudo_logits = pseudo_by_fid[fid]
            if pseudo.shape != gt.shape:
                pseudo = torch.nn.functional.interpolate(
                    pseudo.unsqueeze(0).unsqueeze(0).float(),
                    size=gt.shape[:2],
                    mode="nearest",
                ).squeeze().long()
                pseudo_logits = torch.nn.functional.interpolate(
                    pseudo_logits.permute(2, 0, 1).unsqueeze(0),
                    size=gt.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0)
            miou_p = calc_miou_seg(pseudo.long(), gt)
            pred_p_curr = post_process_seg(pseudo_logits, gt)
            miou_p_curr = calc_miou_seg(pred_p_curr, gt)
            miou_p_list.append(miou_p)
            miou_p_curr_list.append(miou_p_curr)
            row["miou_p"] = round(miou_p * 100.0, 6)
            row["miou_p_curr"] = round(miou_p_curr * 100.0, 6)

        per_frame.append(row)

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    metrics = {
        "miou_g": _mean(miou_g_list) * 100.0,
        "miou_g_curr": _mean(miou_g_curr_list) * 100.0,
    }
    if miou_p_list:
        metrics["miou_p"] = _mean(miou_p_list) * 100.0
        metrics["miou_p_curr"] = _mean(miou_p_curr_list) * 100.0
    return metrics, per_frame
