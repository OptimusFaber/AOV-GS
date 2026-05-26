#!/usr/bin/env python3
"""Per-class mIoU for SemSplaTAM / ActiveSem (OneFormer closed-set on NVS traj)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))
sys.path.append("third_parties/splatam")

from src.naruto.cfg_loader import load_cfg
from src.slam import init_SLAM_model
from src.slam.semsplatam.modified_ver.splatam.eval_helper import post_precess_seg
from src.slam.semsplatam.modified_ver.splatam.splatam import (
    setup_camera,
    transformed_params2rendervar,
    transformed_params2semrendervar,
)
from src.slam.splatam.eval_helper import transform_to_frame
from src.utils.general_utils import InfoPrinter, fix_random_seed
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from channel_rasterization import GaussianRasterizer as SEMRenderer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-class mIoU for ActiveSem / SemSplaTAM.")
    p.add_argument("--cfg", required=True, help="Config path, e.g. configs/Replica/office0/ActiveSem.py")
    p.add_argument("--result_dir", required=True, help="Run directory with splatam/final/params.npz")
    p.add_argument("--stage", default="final", help="Checkpoint stage folder under splatam/")
    p.add_argument("--step", type=int, default=0, help="0 loads splatam/{stage}/params.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output dir (default: {result_dir}/splatam/eval_{stage}/)",
    )
    return p.parse_args()


def load_class_names(class_info_file: Path) -> dict[int, str]:
    with open(class_info_file, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v["name"] for k, v in raw.items()}


def per_class_iou(pred: torch.Tensor, target: torch.Tensor) -> dict[int, float]:
    """IoU for each class present in target (non-zero)."""
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1).to(pred_flat.device)
    valid = target_flat != 0
    pred_v = pred_flat[valid]
    target_v = target_flat[valid]
    out: dict[int, float] = {}
    for cls in torch.unique(target_v):
        cid = int(cls.item())
        pred_cls = pred_v == cls
        true_cls = target_v == cls
        union = (pred_cls | true_cls).sum().float()
        if union > 0:
            out[cid] = ((pred_cls & true_cls).sum().float() / union).item()
    return out


def write_outputs(
    out_dir: Path,
    iou_g: dict[int, list[float]],
    iou_g_curr: dict[int, list[float]],
    class_names: dict[int, str],
    n_frames: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cid in sorted(set(iou_g.keys()) | set(iou_g_curr.keys())):
        g_vals = iou_g.get(cid, [])
        gc_vals = iou_g_curr.get(cid, [])
        rows.append(
            {
                "class_id": cid,
                "class_name": class_names.get(cid, f"class_{cid}"),
                "miou_g": np.mean(g_vals) if g_vals else float("nan"),
                "miou_g_curr": np.mean(gc_vals) if gc_vals else float("nan"),
                "n_frames_g": len(g_vals),
                "n_frames_g_curr": len(gc_vals),
            }
        )
    rows.sort(key=lambda r: (-(r["miou_g_curr"] if not np.isnan(r["miou_g_curr"]) else -1), r["class_name"]))

    csv_path = out_dir / "miou_per_class.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["class_id", "class_name", "miou_g", "miou_g_curr", "n_frames_g", "n_frames_g_curr"],
        )
        w.writeheader()
        w.writerows(rows)

    valid_g = [r["miou_g"] for r in rows if not np.isnan(r["miou_g"])]
    valid_gc = [r["miou_g_curr"] for r in rows if not np.isnan(r["miou_g_curr"])]

    txt_path = out_dir / "miou_per_class.txt"
    lines = [
        "Experiment: semsplatam_per_class_miou (OneFormer closed-set)",
        f"frames_evaluated: {n_frames}",
        f"classes_with_gt: {len(rows)}",
        "",
        f"Overall mIoU_g (mean over classes, macro): {np.mean(valid_g) * 100:.2f}%",
        f"Overall mIoU_g_curr (macro): {np.mean(valid_gc) * 100:.2f}%",
        "",
        "Per-class mIoU (descending by miou_g_curr):",
        f"{'#':>3}  {'class':<24} {'miou_g':>8} {'miou_g_curr':>12} {'n_frames':>9}",
        "-" * 62,
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"{i:>3}  {r['class_name']:<24} "
            f"{r['miou_g'] * 100:>7.2f}% {r['miou_g_curr'] * 100:>11.2f}% {r['n_frames_g_curr']:>9}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {txt_path}")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    info = InfoPrinter("ActiveSemPerClass")

    class _Args:
        cfg = args.cfg
        result_dir = args.result_dir
        seed = args.seed
        enable_vis = 0
        stage = args.stage
        step = args.step

    main_cfg = load_cfg(_Args())
    main_cfg.dirs.result_dir = os.path.abspath(args.result_dir)
    fix_random_seed(args.seed)

    slam = init_SLAM_model(main_cfg, info, logger=None)
    slam.load_params_by_step(step=args.step, stage=args.stage)

    class_names = load_class_names(Path(main_cfg.slam.class_info_file))
    dataset = slam.dataset_eval
    eval_every = slam.config["eval_every"]
    params = slam.params
    variables = slam.variables
    out_dir = Path(args.out_dir) if args.out_dir else Path(main_cfg.dirs.result_dir) / "splatam" / f"eval_{args.stage}"

    iou_g: dict[int, list[float]] = defaultdict(list)
    iou_g_curr: dict[int, list[float]] = defaultdict(list)
    cam = None
    n_eval = 0

    for time_idx in tqdm(range(len(dataset)), desc="eval frames"):
        if time_idx != 0 and (time_idx + 1) % eval_every != 0:
            continue

        color, _, intrinsics, pose = dataset[time_idx]
        gt_w2c = torch.linalg.inv(pose)
        intrinsics = intrinsics[:3, :3]
        seman_gt = dataset.get_semantic_map(time_idx)[0]

        if cam is None:
            first_frame_w2c = gt_w2c
            n_cls = slam.n_cls
            cam = setup_camera(
                color.shape[1],
                color.shape[0],
                intrinsics.cpu().numpy(),
                first_frame_w2c.detach().cpu().numpy(),
                num_channels=n_cls,
            )

        transformed_gaussians = transform_to_frame(
            params, time_idx, gaussians_grad=False, camera_grad=False, rel_w2c=gt_w2c
        )
        rendervar = transformed_params2rendervar(params, transformed_gaussians)
        _, radius, _ = Renderer(raster_settings=cam)(**rendervar)
        seen = radius > 0

        seman_rendervar = transformed_params2semrendervar(params, variables, transformed_gaussians, seen)
        rastered_seman, _ = SEMRenderer(raster_settings=cam)(**seman_rendervar)
        rastered_seman = torch.nan_to_num(rastered_seman, nan=0.0)
        rastered_seman[rastered_seman < 0] = 0.0
        rastered_seman = rastered_seman.permute(1, 2, 0)
        rastered_cls_ids = rastered_seman.argmax(-1)

        gt = seman_gt.long()
        for cid, val in per_class_iou(rastered_cls_ids, gt).items():
            iou_g[cid].append(val)

        reprocess_ids = post_precess_seg(rastered_seman.clone(), gt)
        for cid, val in per_class_iou(reprocess_ids, gt).items():
            iou_g_curr[cid].append(val)

        n_eval += 1

    write_outputs(out_dir, iou_g, iou_g_curr, class_names, n_eval)


if __name__ == "__main__":
    main()
