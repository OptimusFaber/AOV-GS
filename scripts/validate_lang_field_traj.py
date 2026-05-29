#!/usr/bin/env python3
"""
Validate language field on full Replica traj + Habitat semantics (LangSplatV2 s/m/l pyramid).

For each (frame × class): render relevancy at selected SAM levels, pick best level
(argmax fused heatmap, as LangSplatV2 ``eval_lerf.py``), binarize, compute IoU vs GT.

Writes ``metrics.json``, ``pairs.csv``, ``miou_summary.txt``, ``miou_per_class.csv``.
With ``--semsplatam_metrics`` also writes ActiveSem-style frame metrics
(``miou_g``, ``miou_g_curr``, ``miou_p``, ``miou_p_curr``) to ``semantic_result.txt``.

By default ``--eval_preset val_results_sml_levels`` matches
``val-results-sml-levels/`` (s/m/l, thresh 0.55, hyphen class names,
default CLIP negatives, ``align_gs_train_frame``).

Example (pair mIoU + frame metrics; pseudo = SAM+CLIP on rendered SplaTAM RGB)::

    python scripts/validate_lang_field_traj.py \\
      --scene office0 \\
      --result_dir results/Replica/office0/ActiveOpenSem/run_0 \\
      --traj_txt data/replica_sim_nvs/office0/traj.txt \\
      --semsplatam_metrics \\
      --pseudo_source sam_clip \\
      --pseudo_rgb_source rendered \\
      --out_dir results/Replica/office0/ActiveOpenSem/run_0/lang_field_traj_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

try:
    from tqdm import tqdm as _tqdm_cls

    tqdm = _tqdm_cls
    _HAVE_TQDM = True
except ImportError:  # pragma: no cover
    _HAVE_TQDM = False

    def tqdm(x: Any, **_: Any):  # type: ignore[misc]
        return x


_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPTS))

import lang_field_eval_utils as lfu  # noqa: E402
import query_language_field as qlf  # noqa: E402


def langsplat_level_segment(
    heatmaps: list[np.ndarray],
    level_order: tuple[str, ...],
    gt_bool: np.ndarray,
    *,
    thresh: float,
    large_pool: int,
    smooth_pool: int,
    device: torch.device,
) -> tuple[np.ndarray, int, str, float, dict[str, float], dict[str, float]]:
    if len(heatmaps) != len(level_order):
        raise ValueError(f"expected {len(level_order)} heatmaps, got {len(heatmaps)}")

    gt = gt_bool.astype(bool)
    iou_lvl: dict[str, float] = {}
    score_lvl: dict[str, float] = {}
    masks: list[np.ndarray] = []

    for heat, lvl in zip(heatmaps, level_order):
        fused = lfu.langsplat_fuse_heatmap(heat, large_pool=large_pool, device=device)
        score_lvl[lvl] = float(fused.max().detach().cpu().item())
        pred_u8 = lfu.langsplat_mask_from_fused(fused, thresh=thresh, smooth_pool=smooth_pool)
        masks.append(pred_u8)
        iou_lvl[lvl] = lfu.binary_iou(pred_u8 > 0, gt)

    chosen_idx = int(np.argmax([score_lvl[lvl] for lvl in level_order]))
    chosen_lvl = level_order[chosen_idx]
    return (
        masks[chosen_idx],
        chosen_idx,
        chosen_lvl,
        float(iou_lvl[chosen_lvl]),
        iou_lvl,
        score_lvl,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Traj-bundle lang-field mIoU (optional s/m/l pyramid).")
    p.add_argument("--scene", required=True)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--lang_field_s", default=None)
    p.add_argument("--lang_field_m", default=None)
    p.add_argument("--lang_field_l", default=None)
    p.add_argument("--codebook_size", type=int, default=64)
    p.add_argument("--vq_layer_num", type=int, default=1)
    p.add_argument(
        "--levels",
        default="all",
        help="SAM levels to use: 'all' or comma-separated subset of s,m,l (e.g. 'l' or 's,m').",
    )
    p.add_argument("--traj_txt", required=True)
    p.add_argument("--out_dir", default=None, help="Default: <result_dir>/lang_field_traj_eval")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--traj_format", choices=("c2w", "w2c"), default="c2w")
    p.add_argument("--align_gs_train_frame", action="store_true")
    p.add_argument("--replica_train_traj", type=Path, default=None)
    p.add_argument("--info_semantic", default=None)
    p.add_argument("--text_template", default="a {class_name}")
    p.add_argument("--class_name_replace_hyphen_with", default=None, metavar="CHAR")
    p.add_argument("--negative_from_other_classes", action="store_true")
    p.add_argument("--void_class_ids", default="0")
    p.add_argument("--min_gt_pixels", type=int, default=1)
    p.add_argument(
        "--eval_preset",
        choices=("none", lfu.EVAL_PRESET_VAL_RESULTS_SML_LEVELS),
        default=lfu.EVAL_PRESET_VAL_RESULTS_SML_LEVELS,
        help="Evaluation defaults; val_results_sml_levels = val-results-sml-levels/ "
        "(s/m/l, thresh 0.55, a desk-organizer queries, default negatives). "
        "Use none to keep CLI defaults only.",
    )
    p.add_argument("--semantic_mask_thresh", type=float, default=0.55)
    p.add_argument("--semantic_mask_large_pool", type=int, default=29)
    p.add_argument("--semantic_mask_smooth_pool", type=int, default=7)
    p.add_argument("--heatmap_blur", type=float, default=3.0)
    p.add_argument("--negative_texts", default="object,things,stuff,texture")
    p.add_argument("--negative_weight", type=float, default=0.35)
    p.add_argument("--negative_mode", choices=("max", "mean"), default="max")
    p.add_argument("--negative_relu_floor", action="store_true")
    p.add_argument("--negative_score_mode", choices=("softmax_pair", "subtract"), default="softmax_pair")
    p.add_argument("--softmax_inv_temp", type=float, default=10.0)
    p.add_argument("--clip_model", default="ViT-B-16")
    p.add_argument("--clip_pretrained", default="laion2b_s34b_b88k")
    p.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for rendering and CLIP (default cuda:0). "
        "With CUDA_VISIBLE_DEVICES=N pass cuda:0 inside the process.",
    )
    p.add_argument("--no_localization", action="store_true")
    p.add_argument("--allow_mixed_paths", action="store_true")
    p.add_argument(
        "--semsplatam_only",
        action="store_true",
        help="Skip open-vocab pair mIoU; only compute --semsplatam_metrics (faster).",
    )
    p.add_argument(
        "--semsplatam_metrics",
        action="store_true",
        help="Also compute ActiveSem-style frame mIoU (miou_g/_curr, miou_p/_curr) "
        "and write semantic_result.txt.",
    )
    p.add_argument(
        "--pseudo_source",
        choices=("none", "oneformer", "sam_clip"),
        default="sam_clip",
        help="Pseudo-label source for miou_p when --semsplatam_metrics (default: sam_clip, "
        "SAM masks + CLIP text queries; oneformer = ActiveSem closed-set baseline).",
    )
    p.add_argument(
        "--pseudo_rgb_source",
        choices=("rendered", "eval_cache", "habitat_gt"),
        default="rendered",
        help="RGB used for pseudo labels: rendered SplaTAM view (default), cached eval_final PNG, "
        "or Habitat GT frame*.jpg.",
    )
    p.add_argument(
        "--pseudo_eval_stage",
        default="eval_final",
        help="SplaTAM eval subdir when --pseudo_rgb_source=eval_cache (default: eval_final).",
    )
    p.add_argument(
        "--class_info_file",
        default=None,
        help="OneFormer class map (default: configs/Replica/<scene>/class_info_file.json).",
    )
    p.add_argument("--oneformer_checkpoint", default="lly00412/oneformer-replica-finetune")
    p.add_argument("--ade20k_checkpoint", default="shi-labs/oneformer_ade20k_swin_large")
    p.add_argument("--sam_ckpt", default="ckpts/sam_vit_b_01ec64.pth")
    p.add_argument(
        "--semsplatam_max_frames",
        type=int,
        default=-1,
        help="Limit frames for semsplatam_metrics / pseudo (<=0 = all eval frames).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    applied_preset = lfu.apply_eval_preset(args)
    if applied_preset:
        print(f"[lang-field-traj] eval_preset={applied_preset}", file=sys.stderr)
    active_levels = lfu.parse_levels(args.levels)
    do_loc = not bool(args.no_localization)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    traj_txt = Path(args.traj_txt).expanduser().resolve()
    if not traj_txt.is_file():
        raise FileNotFoundError(traj_txt)

    bundle_dir = traj_txt.parent
    results_habitat = bundle_dir / "results_habitat"
    if not results_habitat.is_dir():
        raise FileNotFoundError(f"Expected {results_habitat} next to traj.")

    sem_index = dict(lfu.load_frame_sem_pairs(results_habitat))
    traj_arr = np.loadtxt(str(traj_txt), dtype=np.float64)
    if traj_arr.ndim == 1:
        traj_arr = traj_arr[None, :]
    n_pose = traj_arr.shape[0]
    frame_ids_sorted = sorted(set(range(n_pose)) & set(sem_index.keys()))
    if not frame_ids_sorted:
        raise SystemExit(f"No traj rows ∩ semantic_map_* under {results_habitat}")

    result_dir = Path(args.result_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (result_dir / "lang_field_traj_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

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
            raise FileNotFoundError(
                f"Missing lang_field for level {lvl!r}: {pth}\n"
                f"Train with scripts/aov-gs/03_train_gaussian_lang_field_all_levels.sh "
                f"or pass --lang_field_{lvl}",
            )

    if args.checkpoint is None:
        final_dir = result_dir / "splatam" / "final"
        checkpoint = final_dir / "params0.npz"
        if not checkpoint.is_file():
            checkpoint = final_dir / "params.npz"
    else:
        checkpoint = Path(args.checkpoint).expanduser().resolve()

    if args.info_semantic is None:
        scene_prefix = args.scene[:-1]
        scene_idx = args.scene[-1]
        info_semantic = (
            Path(_ROOT) / "data/replica_v1" / f"{scene_prefix}_{scene_idx}" / "habitat" / "info_semantic.json"
        ).resolve()
    else:
        info_semantic = Path(args.info_semantic).expanduser().resolve()

    if not args.allow_mixed_paths:
        bad = []
        for lvl, pth in lang_field_paths.items():
            try:
                pth.resolve().relative_to(result_dir)
            except ValueError:
                bad.append(f"lang_field_{lvl}={pth}")
        try:
            checkpoint.relative_to(result_dir)
        except ValueError:
            bad.append(f"checkpoint={checkpoint}")
        if bad:
            raise ValueError(
                "Mixed-run paths. Fix paths or pass --allow_mixed_paths.\n"
                + "\n".join(f"  - {x}" for x in bad),
            )

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not info_semantic.is_file():
        raise FileNotFoundError(info_semantic)

    exclude_ids = frozenset(int(x.strip()) for x in str(args.void_class_ids).split(",") if x.strip())
    id_to_name = lfu.load_id_to_canonical_name(info_semantic)
    name_to_id = lfu.load_name_to_id(info_semantic)
    class_names = lfu.discover_classes_from_semantics(
        sem_index, frame_ids_sorted, id_to_name, exclude_ids=exclude_ids,
    )
    if not class_names:
        raise SystemExit("No named classes discovered in semantics.")

    train0 = None
    if args.align_gs_train_frame:
        rtp = args.replica_train_traj
        if rtp is None:
            cand = Path(_ROOT / "data" / "Replica" / args.scene / "traj.txt").resolve()
            if cand.is_file():
                rtp = cand
        if rtp is None or not Path(rtp).expanduser().is_file():
            raise SystemExit("--align_gs_train_frame: set --replica_train_traj or data/Replica/<scene>/traj.txt")
        train0 = lfu.first_c2w_from_traj_file(Path(rtp))

    poses = lfu.poses_from_traj(traj_txt, args.traj_format, device, c2w_train0=train0)
    hyphen_repl = args.class_name_replace_hyphen_with
    if hyphen_repl is not None and hyphen_repl == "":
        hyphen_repl = " "
    text_queries = [
        lfu.build_text_query(cn, args.text_template, replace_hyphen_with=hyphen_repl)
        for cn in class_names
    ]

    first_lvl = active_levels[0]
    latent_dim = qlf.infer_latent_dim(lang_field_paths[first_lvl])
    model = qlf.LangSplatam(checkpoint_path=str(checkpoint), latent_dim=latent_dim, device=str(device))
    model.load_lang_field(lang_field_paths[first_lvl])
    if getattr(model, "model_format", "legacy") != "langsplatv2":
        raise SystemExit("LangSplatV2 lang_field checkpoints required (train with 03_train_gaussian_lang_field.sh).")

    H = int(model.params["org_height"])
    W = int(model.params["org_width"])

    neg_emb: torch.Tensor | None = None
    neg_emb_by_class: dict[str, torch.Tensor] = {}
    neg_texts: list[str] = []
    neg_texts_by_class: dict[str, list[str]] = {}
    if args.negative_from_other_classes:
        for cn in class_names:
            neg_qs = [
                lfu.build_text_query(other, args.text_template, replace_hyphen_with=hyphen_repl)
                for other in class_names if other != cn
            ]
            neg_texts_by_class[cn] = neg_qs
            neg_emb_by_class[cn] = qlf.encode_query_clip_batch(
                neg_qs, args.clip_model, args.clip_pretrained, device,
            )
    else:
        neg_texts = [x.strip() for x in args.negative_texts.split(",") if x.strip()]
        if neg_texts:
            neg_emb = qlf.encode_query_clip_batch(neg_texts, args.clip_model, args.clip_pretrained, device)

    text_cache = {
        tq: qlf.encode_query_clip(tq, args.clip_model, args.clip_pretrained, device)
        for tq in set(text_queries)
    }

    sem_by_fid: dict[int, np.ndarray] = {}
    for fid in frame_ids_sorted:
        sem_np = np.load(str(sem_index[fid])).astype(np.int64)
        if sem_np.ndim == 3:
            sem_np = sem_np.squeeze()
        if sem_np.shape[:2] != (H, W):
            sem_np = cv2.resize(sem_np.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        sem_by_fid[fid] = sem_np

    do_semsplatam = bool(args.semsplatam_metrics or args.semsplatam_only)
    if args.semsplatam_only and not args.semsplatam_metrics:
        args.semsplatam_metrics = True

    work_items: list[tuple[int, str, str, int]] = []
    if not args.semsplatam_only:
        for fid in frame_ids_sorted:
            se = sem_by_fid[fid]
            for cn, cq in zip(class_names, text_queries):
                cid_int = lfu.resolve_class_id(cn, name_to_id)
                if int(np.sum(se == int(cid_int))) >= args.min_gt_pixels:
                    work_items.append((fid, cn, cq, int(cid_int)))
    else:
        eval_fids = frame_ids_sorted
        if int(args.semsplatam_max_frames) > 0:
            eval_fids = eval_fids[: int(args.semsplatam_max_frames)]
        for fid in eval_fids:
            se = sem_by_fid[fid]
            for cn, cq in zip(class_names, text_queries):
                cid_int = lfu.resolve_class_id(cn, name_to_id)
                if int(np.sum(se == int(cid_int))) >= args.min_gt_pixels:
                    work_items.append((fid, cn, cq, int(cid_int)))

    n_tasks = len(work_items)
    print(
        f"[lang-field-traj] pairs={n_tasks} frames={len(frame_ids_sorted)} classes={len(class_names)} "
        f"levels={','.join(active_levels)} thresh={args.semantic_mask_thresh}",
        file=sys.stderr,
    )

    rows: list[dict[str, Any]] = []
    miou_vals: list[float] = []
    prec_vals: list[float] = []
    rec_vals: list[float] = []
    f1_vals: list[float] = []
    chosen_lvl_counts = {lvl: 0 for lvl in active_levels}
    tp_all = fp_all = fn_all = 0
    loc_hits = loc_total = 0
    frame_score_maps: dict[int, dict[str, np.ndarray]] = {}

    render_kw_base = dict(
        ae=None,
        negative_weight=float(args.negative_weight),
        negative_mode=str(args.negative_mode),
        negative_relu_floor=bool(args.negative_relu_floor),
        H=H,
        W=W,
        device=device,
        blur_sigma=float(args.heatmap_blur),
        use_v2=True,
        negative_score_mode=str(args.negative_score_mode),
        softmax_inv_temp=float(args.softmax_inv_temp),
    )

    for k, (fid, cn, cq, cid_int) in enumerate(tqdm(work_items, desc="lang-field-traj", unit="pair", file=sys.stderr)):
        w2c = poses[fid]
        gt_bool = sem_by_fid[fid] == int(cid_int)
        n_gt = int(gt_bool.sum())

        clip_q = text_cache[cq]
        pair_neg_emb = neg_emb_by_class.get(cn) if args.negative_from_other_classes else neg_emb
        render_kw = {**render_kw_base, "negative_clip_queries": pair_neg_emb}

        heatmaps: list[np.ndarray] = []
        for lvl in active_levels:
            model.load_lang_field(lang_field_paths[lvl])
            heatmaps.append(
                qlf._render_raw_cosine_map(model, clip_q, w2c=w2c, **render_kw),
            )

        if do_semsplatam:
            score_map, _, _ = lfu.pick_best_level_fused(
                heatmaps,
                active_levels,
                large_pool=int(args.semantic_mask_large_pool),
                device=device,
            )
            frame_score_maps.setdefault(fid, {})[cn] = score_map

        if args.semsplatam_only:
            continue

        pred_u8, chosen_idx, chosen_lvl, iou, iou_lvl, score_lvl = langsplat_level_segment(
            heatmaps,
            active_levels,
            gt_bool,
            thresh=float(args.semantic_mask_thresh),
            large_pool=int(args.semantic_mask_large_pool),
            smooth_pool=int(args.semantic_mask_smooth_pool),
            device=device,
        )
        pred = pred_u8 > 0
        chosen_lvl_counts[chosen_lvl] += 1

        tp = int(np.logical_and(pred, gt_bool).sum())
        fp = int(np.logical_and(pred, ~gt_bool).sum())
        fn = int(np.logical_and(~pred, gt_bool).sum())
        tp_all += tp
        fp_all += fp
        fn_all += fn

        prec = lfu.safe_div(tp, tp + fp)
        rec = lfu.safe_div(tp, tp + fn)
        f1 = lfu.safe_div(2.0 * prec * rec, prec + rec) if prec == prec and rec == rec else float("nan")

        if prec == prec:
            prec_vals.append(prec)
        if rec == rec:
            rec_vals.append(rec)
        if f1 == f1:
            f1_vals.append(f1)
        if iou == iou:
            miou_vals.append(iou)

        hit = False
        if do_loc:
            loc_total += 1
            bb = lfu.gt_aabb_from_mask(gt_bool)
            M = cv2.moments((pred_u8 > 0).astype(np.uint8))
            if bb is not None and M["m00"] > 1e-3:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                hit = lfu.centroid_in_box(cx, cy, bb)
            if hit:
                loc_hits += 1

        row: dict[str, Any] = {
            "frame_id": fid,
            "class_name": cn,
            "text_query": cq,
            "chosen_level": chosen_lvl,
            "chosen_level_idx": chosen_idx,
            "iou": "" if math.isnan(iou) else round(float(iou), 6),
            "precision": "" if math.isnan(prec) else round(float(prec), 6),
            "recall": "" if math.isnan(rec) else round(float(rec), 6),
            "f1": "" if math.isnan(f1) else round(float(f1), 6),
            "gt_pixels": n_gt,
            "pred_pixels": int(pred.sum()),
            "localization_hit": (1 if hit else 0) if do_loc else "",
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        for lvl in active_levels:
            iv = iou_lvl[lvl]
            sv = score_lvl[lvl]
            row[f"iou_{lvl}"] = "" if math.isnan(iv) else round(float(iv), 6)
            row[f"level_score_{lvl}"] = round(float(sv), 6)
        rows.append(row)

    (out_dir / "queries_used.txt").write_text(
        "\n".join(f"{cn}\t{tq}" for cn, tq in zip(class_names, text_queries)) + "\n",
        encoding="utf-8",
    )

    miou = float(np.nanmean(miou_vals)) if miou_vals else float("nan")
    prec_micro = lfu.safe_div(float(tp_all), float(tp_all + fp_all))
    rec_micro = lfu.safe_div(float(tp_all), float(tp_all + fn_all))
    f1_micro = lfu.safe_div(2.0 * prec_micro * rec_micro, prec_micro + rec_micro)

    summary = {
        "scene": args.scene,
        "result_dir": str(result_dir),
        "traj_txt": str(traj_txt),
        "results_habitat": str(results_habitat),
        "eval_preset": applied_preset,
        "levels": list(active_levels),
        "lang_fields": {lvl: str(lang_field_paths[lvl]) for lvl in active_levels},
        "checkpoint": str(checkpoint),
        "langsplat_v2": True,
        "multilevel_mode": "langsplat_eval_lerf_pyramid",
        "aligned_train_frame": bool(args.align_gs_train_frame),
        "text_template": str(args.text_template),
        "class_name_replace_hyphen_with": args.class_name_replace_hyphen_with,
        "negative_from_other_classes": bool(args.negative_from_other_classes),
        "negative_texts": str(args.negative_texts),
        "negative_weight": float(args.negative_weight),
        "negative_mode": str(args.negative_mode),
        "negative_score_mode": str(args.negative_score_mode),
        "softmax_inv_temp": float(args.softmax_inv_temp),
        "semantic_mask_thresh": float(args.semantic_mask_thresh),
        "semantic_mask_large_pool": int(args.semantic_mask_large_pool),
        "semantic_mask_smooth_pool": int(args.semantic_mask_smooth_pool),
        "heatmap_blur": float(args.heatmap_blur),
        "localization_enabled": do_loc,
        "text_queries": text_queries,
        "pairs_evaluated": len(rows),
        "frames_evaluated": len(frame_ids_sorted),
        "classes_discovered": class_names,
        "mIoU_mean_over_pairs": miou,
        "precision_macro_mean_pairs": float(np.mean(prec_vals)) if prec_vals else float("nan"),
        "recall_macro_mean_pairs": float(np.mean(rec_vals)) if rec_vals else float("nan"),
        "f1_macro_mean_pairs": float(np.mean(f1_vals)) if f1_vals else float("nan"),
        "precision_micro_global": prec_micro,
        "recall_micro_global": rec_micro,
        "f1_micro_global": f1_micro,
        "chosen_level_counts": chosen_lvl_counts,
        "localization_hit_rate_pairs": (
            float(loc_hits) / float(loc_total) if do_loc and loc_total > 0 else None
        ),
    }

    json_path = out_dir / "metrics.json"
    csv_path = out_dir / "pairs.csv"
    json_path.write_text(
        json.dumps({"summary": summary, "per_pair": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    headers = list(rows[0].keys()) if rows else ["frame_id", "class_name", "iou"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in headers})

    miou_txt, miou_csv = lfu.write_miou_summary(
        out_dir,
        experiment_name=out_dir.name,
        summary=summary,
        per_pair=rows,
    )

    sem_metrics: dict[str, float] | None = None
    sem_frame_rows: list[dict[str, Any]] = []
    if do_semsplatam:
        sem_frame_ids = sorted(frame_score_maps.keys())
        if int(args.semsplatam_max_frames) > 0:
            sem_frame_ids = sem_frame_ids[: int(args.semsplatam_max_frames)]

        cn_to_id = {cn: int(lfu.resolve_class_id(cn, name_to_id)) for cn in class_names}
        pseudo_by_fid: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None

        if args.pseudo_source != "none":
            print(
                f"[semsplatam_metrics] pseudo_source={args.pseudo_source} "
                f"pseudo_rgb_source={args.pseudo_rgb_source} "
                f"frames={len(sem_frame_ids)}",
                file=sys.stderr,
            )
            pseudo_by_fid = {}
            oneformer_bundle = None
            sam_ckpt: Path | None = None
            if args.pseudo_source == "oneformer":
                if args.class_info_file is None:
                    cif = _ROOT / "configs" / "Replica" / args.scene / "class_info_file.json"
                    if not cif.is_file():
                        cif = _ROOT / "configs" / "Replica" / "office0" / "class_info_file.json"
                else:
                    cif = Path(args.class_info_file).expanduser().resolve()
                oneformer_bundle = lfu.load_oneformer_replica(
                    device=device,
                    class_info_file=cif,
                    oneformer_checkpoint=args.oneformer_checkpoint,
                    ade20k_checkpoint=args.ade20k_checkpoint,
                )
            elif args.pseudo_source == "sam_clip":
                sam_ckpt = Path(args.sam_ckpt).expanduser()
                if not sam_ckpt.is_file():
                    sam_ckpt = (_ROOT / args.sam_ckpt).resolve()
                if not sam_ckpt.is_file():
                    raise FileNotFoundError(sam_ckpt)

            for fid in tqdm(sem_frame_ids, desc="pseudo-labels", unit="frame", file=sys.stderr):
                rgb_np = lfu.load_pseudo_rgb_np(
                    frame_id=fid,
                    source=str(args.pseudo_rgb_source),
                    results_habitat=results_habitat,
                    result_dir=result_dir,
                    model=model,
                    w2c=poses[fid],
                    H=H,
                    W=W,
                    eval_stage=str(args.pseudo_eval_stage),
                )
                if rgb_np is None:
                    continue

                if args.pseudo_source == "oneformer":
                    proc, of_model = oneformer_bundle
                    pseudo, plogits = lfu.oneformer_pseudo_for_rgb(
                        rgb_np, processor=proc, model=of_model, device=device,
                    )
                else:
                    assert sam_ckpt is not None
                    class_ids_int = [cn_to_id[cn] for cn in class_names]
                    pseudo, plogits = lfu.sam_clip_pseudo_for_rgb(
                        rgb_np,
                        class_names=class_names,
                        text_queries=text_queries,
                        class_ids=class_ids_int,
                        sam_ckpt=sam_ckpt,
                        clip_model=args.clip_model,
                        clip_pretrained=args.clip_pretrained,
                        device=device,
                    )
                pseudo_by_fid[fid] = (pseudo, plogits)

        sem_metrics, sem_frame_rows = lfu.compute_frame_semsplatam_metrics(
            frame_score_maps=frame_score_maps,
            sem_by_fid=sem_by_fid,
            class_name_to_id=cn_to_id,
            frame_ids=sem_frame_ids,
            pseudo_by_fid=pseudo_by_fid,
        )
        summary["semsplatam_metrics"] = sem_metrics
        summary["semsplatam_pseudo_source"] = args.pseudo_source
        summary["semsplatam_pseudo_rgb_source"] = args.pseudo_rgb_source
        summary["semsplatam_pseudo_eval_stage"] = (
            str(args.pseudo_eval_stage) if args.pseudo_rgb_source == "eval_cache" else None
        )
        summary["semsplatam_frames"] = len(sem_frame_rows)
        json_path.write_text(
            json.dumps({"summary": summary, "per_pair": rows, "semsplatam_per_frame": sem_frame_rows}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        sem_txt = lfu.write_semantic_result_txt(out_dir, sem_metrics)
        sem_csv = out_dir / "semsplatam_per_frame.csv"
        with sem_csv.open("w", newline="", encoding="utf-8") as f:
            headers = list(sem_frame_rows[0].keys()) if sem_frame_rows else ["frame_id", "miou_g"]
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in sem_frame_rows:
                w.writerow(r)

    print("")
    print(f"mIoU (mean over pairs): {miou:.6f}  (n={len(miou_vals)} pairs, levels={','.join(active_levels)})")
    print(f"  precision_macro={summary['precision_macro_mean_pairs']:.4f}  "
          f"recall_macro={summary['recall_macro_mean_pairs']:.4f}  "
          f"f1_macro={summary['f1_macro_mean_pairs']:.4f}")
    if do_loc and loc_total > 0:
        print(f"  localization_hit_rate={summary['localization_hit_rate_pairs']:.4f}  ({loc_hits}/{loc_total})")
    print(f"Saved metrics  -> {json_path}")
    print(f"Saved pairs    -> {csv_path}")
    print(f"Saved mIoU txt -> {miou_txt}")
    print(f"Saved mIoU csv -> {miou_csv}")
    if sem_metrics is not None:
        print("")
        print("SemSplaTAM-style frame metrics (percent):")
        for k in ("miou_g", "miou_g_curr", "miou_p", "miou_p_curr"):
            if k in sem_metrics:
                print(f"  {k}: {sem_metrics[k]:.4f}")
        print(f"Saved semantic_result.txt -> {out_dir / 'semantic_result.txt'}")
        print(f"Saved per-frame csv     -> {out_dir / 'semsplatam_per_frame.csv'}")


if __name__ == "__main__":
    main()
