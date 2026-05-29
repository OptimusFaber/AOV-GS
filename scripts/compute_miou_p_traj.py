#!/usr/bin/env python3
"""
Compute mIoU_p / mIoU_p(curr) only on Replica traj eval frames.

Pipeline (per frame):
  1. Render RGB from SplaTAM checkpoint at traj pose.
  2. Run SAM + CLIP text matching (open-vocab pseudo labels).
  3. Compare pseudo segmentation vs Habitat GT semantic map.

Does **not** touch language field / pair mIoU (your existing mIoU_g).

Example::

    python scripts/compute_miou_p_traj.py \\
      --scene office0 \\
      --result_dir results/Replica/office0/ActiveGeom/run_0 \\
      --traj_txt /mnt/data/replica_sim_nvs/office0/traj.txt

    python scripts/compute_miou_p_traj.py \\
      --scene office0 \\
      --result_dir results/Replica/office0/ActiveOpenSemHybrid/run_0 \\
      --traj_txt /mnt/data/replica_sim_nvs/office0/traj.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(x, **kwargs):  # type: ignore[misc]
        return x


_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPTS))

import lang_field_eval_utils as lfu  # noqa: E402
import query_language_field as qlf  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Traj mIoU_p via SplaTAM render + SAM+CLIP.")
    p.add_argument("--scene", required=True)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--traj_txt", required=True)
    p.add_argument("--out_dir", default=None, help="Default: <result_dir>/miou_p_traj_eval")
    p.add_argument("--checkpoint", default=None, help="Default: <result_dir>/splatam/final/params.npz")
    p.add_argument("--traj_format", choices=("c2w", "w2c"), default="c2w")
    p.add_argument("--align_gs_train_frame", action="store_true", default=True)
    p.add_argument("--no_align_gs_train_frame", action="store_false", dest="align_gs_train_frame")
    p.add_argument("--replica_train_traj", type=Path, default=None)
    p.add_argument("--info_semantic", default=None)
    p.add_argument("--void_class_ids", default="0")
    p.add_argument("--text_template", default="a {class_name}")
    p.add_argument("--class_name_replace_hyphen_with", default=None)
    p.add_argument("--sam_ckpt", default="ckpts/sam_vit_b_01ec64.pth")
    p.add_argument("--clip_model", default="ViT-B-16")
    p.add_argument("--clip_pretrained", default="laion2b_s34b_b88k")
    p.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for rendering and SAM+CLIP (default cuda:0). "
        "With CUDA_VISIBLE_DEVICES=N pass cuda:0 inside the process.",
    )
    p.add_argument("--max_frames", type=int, default=-1, help="<=0 = all eval frames")
    p.add_argument("--codebook_size", type=int, default=64, help="LangSplatam latent_dim hint for RGB render")
    return p.parse_args()


def resolve_checkpoint(result_dir: Path, checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        ckpt = Path(checkpoint_arg).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        return ckpt
    final_dir = result_dir / "splatam" / "final"
    for name in ("params.npz", "params0.npz"):
        cand = final_dir / name
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"No checkpoint under {final_dir}")


def resolve_info_semantic(scene: str, info_arg: str | None) -> Path:
    if info_arg:
        path = Path(info_arg).expanduser().resolve()
    else:
        scene_prefix = scene[:-1]
        scene_idx = scene[-1]
        path = (_ROOT / "data/replica_v1" / f"{scene_prefix}_{scene_idx}" / "habitat" / "info_semantic.json").resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_gt_semantic(path: Path, H: int, W: int) -> np.ndarray:
    sem = np.load(str(path)).astype(np.int64)
    if sem.ndim == 3:
        sem = sem.squeeze()
    if sem.shape[:2] != (H, W):
        sem = cv2.resize(sem.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int64)
    return sem


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    traj_txt = Path(args.traj_txt).expanduser().resolve()
    if not traj_txt.is_file():
        raise FileNotFoundError(traj_txt)

    result_dir = Path(args.result_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (result_dir / "miou_p_traj_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_dir = traj_txt.parent
    results_habitat = bundle_dir / "results_habitat"
    if not results_habitat.is_dir():
        raise FileNotFoundError(f"Expected {results_habitat} next to traj.")

    sem_index = dict(lfu.load_frame_sem_pairs(results_habitat))
    traj_arr = np.loadtxt(str(traj_txt), dtype=np.float64)
    if traj_arr.ndim == 1:
        traj_arr = traj_arr[None, :]
    frame_ids = sorted(set(range(traj_arr.shape[0])) & set(sem_index.keys()))
    if not frame_ids:
        raise SystemExit(f"No traj rows intersect semantic maps under {results_habitat}")
    if int(args.max_frames) > 0:
        frame_ids = frame_ids[: int(args.max_frames)]

    info_semantic = resolve_info_semantic(args.scene, args.info_semantic)
    exclude_ids = frozenset(int(x.strip()) for x in str(args.void_class_ids).split(",") if x.strip())
    id_to_name = lfu.load_id_to_canonical_name(info_semantic)
    name_to_id = lfu.load_name_to_id(info_semantic)
    class_names = lfu.discover_classes_from_semantics(sem_index, frame_ids, id_to_name, exclude_ids=exclude_ids)
    if not class_names:
        raise SystemExit("No named classes discovered in GT semantics.")

    hyphen_repl = args.class_name_replace_hyphen_with
    if hyphen_repl is not None and hyphen_repl == "":
        hyphen_repl = " "
    text_queries = [
        lfu.build_text_query(cn, args.text_template, replace_hyphen_with=hyphen_repl)
        for cn in class_names
    ]
    class_ids_int = [int(lfu.resolve_class_id(cn, name_to_id)) for cn in class_names]

    train0 = None
    if args.align_gs_train_frame:
        rtp = args.replica_train_traj
        if rtp is None:
            cand = (_ROOT / "data" / "Replica" / args.scene / "traj.txt").resolve()
            if cand.is_file():
                rtp = cand
        if rtp is None or not Path(rtp).expanduser().is_file():
            raise SystemExit("--align_gs_train_frame: set --replica_train_traj or data/Replica/<scene>/traj.txt")
        train0 = lfu.first_c2w_from_traj_file(Path(rtp))

    checkpoint = resolve_checkpoint(result_dir, args.checkpoint)
    model = qlf.LangSplatam(
        checkpoint_path=str(checkpoint),
        latent_dim=int(args.codebook_size),
        device=str(device),
    )
    H = int(model.params["org_height"])
    W = int(model.params["org_width"])
    poses = lfu.poses_from_traj(traj_txt, args.traj_format, device, c2w_train0=train0)

    sam_ckpt = Path(args.sam_ckpt).expanduser()
    if not sam_ckpt.is_file():
        sam_ckpt = (_ROOT / args.sam_ckpt).resolve()
    if not sam_ckpt.is_file():
        raise FileNotFoundError(sam_ckpt)

    miou_p_list: list[float] = []
    miou_p_curr_list: list[float] = []
    per_frame: list[dict[str, Any]] = []

    print(
        f"[miou_p] frames={len(frame_ids)} classes={len(class_names)} "
        f"render={checkpoint.name} sam={sam_ckpt.name}",
        file=sys.stderr,
    )

    for fid in tqdm(frame_ids, desc="miou_p", unit="frame", file=sys.stderr):
        gt = torch.from_numpy(load_gt_semantic(sem_index[fid], H, W))
        bgr = qlf.render_rgb(model, poses[fid], H, W)
        rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        pseudo, pseudo_logits = lfu.sam_clip_pseudo_for_rgb(
            rgb_np,
            class_names=class_names,
            text_queries=text_queries,
            class_ids=class_ids_int,
            sam_ckpt=sam_ckpt,
            clip_model=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            device=device,
        )

        miou_p = lfu.calc_miou_seg(pseudo.long(), gt)
        pred_p_curr = lfu.post_process_seg(pseudo_logits, gt)
        miou_p_curr = lfu.calc_miou_seg(pred_p_curr, gt)
        miou_p_list.append(miou_p)
        miou_p_curr_list.append(miou_p_curr)
        per_frame.append(
            {
                "frame_id": fid,
                "miou_p": round(miou_p * 100.0, 6),
                "miou_p_curr": round(miou_p_curr * 100.0, 6),
            },
        )

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    metrics = {
        "miou_p": _mean(miou_p_list) * 100.0,
        "miou_p_curr": _mean(miou_p_curr_list) * 100.0,
    }

    summary = {
        "scene": args.scene,
        "result_dir": str(result_dir),
        "traj_txt": str(traj_txt),
        "checkpoint": str(checkpoint),
        "pseudo_source": "sam_clip",
        "pseudo_rgb_source": "rendered_splatam",
        "align_gs_train_frame": bool(args.align_gs_train_frame),
        "text_template": args.text_template,
        "frames_evaluated": len(frame_ids),
        "classes": class_names,
        "text_queries": text_queries,
        **metrics,
    }

    json_path = out_dir / "miou_p_metrics.json"
    json_path.write_text(
        json.dumps({"summary": summary, "per_frame": per_frame}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    txt_path = lfu.write_semantic_result_txt(out_dir, metrics)

    csv_path = out_dir / "miou_p_per_frame.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id", "miou_p", "miou_p_curr"])
        w.writeheader()
        w.writerows(per_frame)

    print("")
    print(f"mIoU_p:      {metrics['miou_p']:.4f}%")
    print(f"mIoU_p(curr): {metrics['miou_p_curr']:.4f}%")
    print(f"Saved {json_path}")
    print(f"Saved {txt_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
