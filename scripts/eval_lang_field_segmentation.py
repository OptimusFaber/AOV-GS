#!/usr/bin/env python3
"""
Evaluate segmentation metrics (IoU/mIoU) for a trained Gaussian language field (AOV-GS).

Expected inputs (Replica NVS layout):
  data/replica_sim_nvs/{scene}/results_habitat/
    frameXXXXXX.jpg
    semantic/semantic_map_XXXX.npy
  data/replica_sim_nvs/{scene}/traj.txt

This script:
  1) loads GS checkpoint (params*.npz) + language field (lang_field.pt),
  2) renders per-pixel latent features for poses from traj.txt,
  3) decodes latent -> CLIP(512) with an autoencoder checkpoint,
  4) computes cosine similarity with text queries (one per class),
  5) thresholds similarity maps to binary masks and reports IoU per class.

Outputs:
  {out_dir}/{scene}_seg_metrics.json
  {out_dir}/{scene}_seg_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.slam.langsplatam.langsplatam import LangSplatam  # noqa: E402
from src.semantic.language_autoencoder import Autoencoder  # noqa: E402


def _num_key(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else -1


def _load_traj_txt(path: Path, device: torch.device, fmt: str) -> Dict[int, torch.Tensor]:
    """
    Load traj.txt with 16 floats per line.

    fmt:
      - 'c2w': file contains camera-to-world; we invert to get w2c.
      - 'w2c': file contains world-to-camera already.
    """
    arr = np.loadtxt(str(path), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != 16:
        raise ValueError(f"{path} must contain 16 floats per line.")
    mats = arr.reshape(-1, 4, 4)
    out: Dict[int, torch.Tensor] = {}
    for i in range(mats.shape[0]):
        m = torch.tensor(mats[i], dtype=torch.float32, device=device)
        if fmt == "c2w":
            m = torch.inverse(m)
        out[i] = m
    return out


def _load_name_to_id(info_semantic_path: Path) -> dict[str, int]:
    data = json.loads(info_semantic_path.read_text(encoding="utf-8"))
    m: dict[str, int] = {}
    for c in data["classes"]:
        name = c["name"].strip().lower()
        cid = int(c["id"])
        m[name] = cid
        m[name.replace("-", " ")] = cid
    return m


def _resolve_class_id(class_name: str, name_to_id: dict[str, int]) -> int:
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


def _collect_frame_sem_pairs(results_habitat: Path) -> List[Tuple[int, Path, Path]]:
    frame_files = {_num_key(p): p for p in sorted(results_habitat.glob("frame*.jpg"), key=_num_key)}
    sem_dir = results_habitat / "semantic"
    sem_files = {_num_key(p): p for p in sorted(sem_dir.glob("semantic_map_*.npy"), key=_num_key)}
    common = sorted(set(frame_files.keys()) & set(sem_files.keys()))
    return [(idx, frame_files[idx], sem_files[idx]) for idx in common]


@torch.inference_mode()
def _encode_texts_clip(texts: List[str], clip_model: str, clip_pretrained: str, device: torch.device) -> torch.Tensor:
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(clip_model, pretrained=clip_pretrained, device=device)
    model.eval()
    tok = open_clip.get_tokenizer(clip_model)
    tokens = tok(texts).to(device)
    return F.normalize(model.encode_text(tokens), dim=-1)  # [C, 512]


def _binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    return float(inter) / float(union)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b > 0 else float("nan")


def _load_ae(
    ae_ckpt: Path,
    device: torch.device,
    encoder_dims: List[int] | None,
    decoder_dims: List[int] | None,
) -> Autoencoder:
    # Autoencoder supports explicit dims; keep behavior consistent with query_language_field.py
    ae = Autoencoder(encoder_dims, decoder_dims).to(device)
    state = torch.load(str(ae_ckpt), map_location=device)
    ae.load_state_dict(state)
    ae.eval()
    return ae


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate AOV-GS language-field segmentation IoU/mIoU.")
    p.add_argument("--scene", required=True, help="Replica scene name, e.g. office0")
    p.add_argument("--result_dir", required=True, help="Result dir (used only for default --out_dir).")
    p.add_argument("--lang_field", required=True, help="Path to lang_field.pt")
    p.add_argument("--checkpoint", default=None, help="Path to params0.npz/params.npz. Auto if omitted.")
    p.add_argument("--ae_ckpt", required=True, help="Autoencoder checkpoint (.pth) for decoding latent -> CLIP512.")
    p.add_argument("--encoder_dims", nargs="+", type=int, default=None, help="Optional AE encoder dims override.")
    p.add_argument("--decoder_dims", nargs="+", type=int, default=None, help="Optional AE decoder dims override.")

    p.add_argument("--data_root", default="data/replica_sim_nvs", help="Root for scene NVS data.")
    p.add_argument("--traj_txt", default=None, help="Path to traj.txt. Auto from data_root/scene if omitted.")
    p.add_argument("--traj_format", choices=("c2w", "w2c"), default="c2w")
    p.add_argument(
        "--info_semantic",
        default=None,
        help="Path to info_semantic.json (class name -> id). Default: data/replica_v1/<scene>_/habitat/info_semantic.json",
    )

    p.add_argument("--classes", required=True, help="Comma-separated class names, e.g. 'chair,table,sofa'")
    p.add_argument("--text_template", default="a {class_name}", help="Template for CLIP queries.")

    p.add_argument("--threshold_mode", choices=("fixed", "quantile"), default="quantile")
    p.add_argument("--threshold", type=float, default=0.25, help="Used when threshold_mode=fixed (raw cosine).")
    p.add_argument("--top_percentile", type=float, default=2.0, help="Used when threshold_mode=quantile.")
    p.add_argument("--include_empty_gt", action="store_true", help="Include frames where GT class is absent.")

    p.add_argument("--max_frames", type=int, default=-1, help="<=0 means all available frame/semantic pairs.")
    p.add_argument("--frame_stride", type=int, default=1)

    p.add_argument("--clip_model", default="ViT-B-16")
    p.add_argument("--clip_pretrained", default="laion2b_s34b_b88k")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_dir", default=None, help="Default: {result_dir}/lang_field_seg_eval")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    result_dir = Path(args.result_dir).expanduser().resolve()
    lang_field = Path(args.lang_field).expanduser().resolve()
    ae_ckpt = Path(args.ae_ckpt).expanduser().resolve()

    if args.checkpoint is None:
        final_dir = result_dir / "splatam" / "final"
        ckpt0 = final_dir / "params0.npz"
        ckpt1 = final_dir / "params.npz"
        checkpoint = ckpt0 if ckpt0.exists() else ckpt1
    else:
        checkpoint = Path(args.checkpoint).expanduser().resolve()

    data_root = Path(args.data_root).expanduser().resolve()
    scene_root = data_root / args.scene
    results_habitat = scene_root / "results_habitat"
    traj_txt = Path(args.traj_txt).expanduser().resolve() if args.traj_txt else (scene_root / "traj.txt")

    if args.info_semantic is None:
        scene_prefix = args.scene[:-1]
        scene_idx = args.scene[-1]
        info_semantic = Path(f"data/replica_v1/{scene_prefix}_{scene_idx}/habitat/info_semantic.json").resolve()
    else:
        info_semantic = Path(args.info_semantic).expanduser().resolve()

    if not results_habitat.is_dir():
        raise FileNotFoundError(f"results_habitat not found: {results_habitat}")
    if not traj_txt.is_file():
        raise FileNotFoundError(f"traj.txt not found: {traj_txt}")
    if not info_semantic.is_file():
        raise FileNotFoundError(f"info_semantic.json not found: {info_semantic}")
    if not lang_field.is_file():
        raise FileNotFoundError(f"lang_field.pt not found: {lang_field}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not ae_ckpt.is_file():
        raise FileNotFoundError(f"ae_ckpt not found: {ae_ckpt}")

    name_to_id = _load_name_to_id(info_semantic)
    class_names = [x.strip() for x in args.classes.split(",") if x.strip()]
    if not class_names:
        raise ValueError("No classes provided in --classes.")
    class_ids = [_resolve_class_id(cn, name_to_id) for cn in class_names]
    text_queries = [args.text_template.format(class_name=cn) for cn in class_names]

    # Model + language field
    ckpt_dict = torch.load(str(lang_field), map_location="cpu")
    latent_dim = int(ckpt_dict.get("latent_dim") or ckpt_dict.get("lang_feats").shape[1])
    model = LangSplatam(checkpoint_path=str(checkpoint), latent_dim=latent_dim, device=str(device))
    model.load_lang_field(lang_field)

    # Renderer resolution
    Hm = int(model.params["org_height"])
    Wm = int(model.params["org_width"])

    # AE and CLIP queries
    ae = _load_ae(ae_ckpt, device, args.encoder_dims, args.decoder_dims)
    text_emb_512 = _encode_texts_clip(text_queries, args.clip_model, args.clip_pretrained, device)  # [C,512]

    poses = _load_traj_txt(traj_txt, device=device, fmt=args.traj_format)  # idx -> w2c
    pairs = _collect_frame_sem_pairs(results_habitat)
    if not pairs:
        raise RuntimeError(
            f"No frame/semantic pairs found in {results_habitat}. "
            f"Expected frame*.jpg and semantic/semantic_map_*.npy."
        )
    if args.frame_stride > 1:
        pairs = pairs[:: args.frame_stride]
    num_pairs_available = len(pairs)
    if args.max_frames is not None and int(args.max_frames) > 0:
        pairs = pairs[: args.max_frames]

    per_class_frame_ious: Dict[str, List[float]] = defaultdict(list)
    per_class_inter: Dict[str, int] = defaultdict(int)
    per_class_union: Dict[str, int] = defaultdict(int)
    per_class_pred_pos: Dict[str, int] = defaultdict(int)
    per_class_gt_pos: Dict[str, int] = defaultdict(int)
    skipped_no_pose = 0
    skipped_bad_sem_shape = 0

    # Streaming decode to avoid holding [H*W,512] for large images.
    decode_batch = 4096

    for idx, _rgb_path, sem_path in pairs:
        if idx not in poses:
            skipped_no_pose += 1
            continue

        w2c = poses[idx]
        sem = np.load(str(sem_path)).astype(np.int64)
        Hs, Ws = int(sem.shape[0]), int(sem.shape[1])
        if sem.ndim != 2:
            raise ValueError(f"Semantic map must be 2D int array, got shape {sem.shape} in {sem_path}")

        # Render latent feature map at model resolution, then (if needed) resize to semantic resolution.
        with torch.no_grad():
            lat = model.render_lang(w2c, Hm, Wm)  # [D,Hm,Wm]
            if (Hm != Hs) or (Wm != Ws):
                lat = F.interpolate(lat.unsqueeze(0), size=(Hs, Ws), mode="bilinear", align_corners=False)[0]

            D = int(lat.shape[0])
            flat_lat = lat.permute(1, 2, 0).reshape(-1, D)  # [HW,D]

            sim_flat = torch.empty(
                (flat_lat.shape[0], len(class_names)),
                device=device,
                dtype=torch.float32,
            )
            q = text_emb_512.to(device).to(torch.float32)  # [C,512]
            for s in range(0, flat_lat.shape[0], decode_batch):
                chunk = flat_lat[s : s + decode_batch]
                dec = ae.decode(chunk)  # [B,512]
                dec = F.normalize(dec, p=2, dim=-1).to(torch.float32)
                sim_flat[s : s + dec.shape[0]] = dec @ q.T  # [B,C]
            sim = sim_flat.reshape(Hs, Ws, len(class_names)).cpu().numpy().astype(np.float32)

        if sim.shape[0] != Hs or sim.shape[1] != Ws:
            skipped_bad_sem_shape += 1
            continue

        for ci, (cn, cid) in enumerate(zip(class_names, class_ids)):
            gt = sem == int(cid)
            if (not gt.any()) and (not args.include_empty_gt):
                continue

            s_map = sim[:, :, ci]
            if args.threshold_mode == "fixed":
                pred = s_map >= float(args.threshold)
            else:
                thr = float(np.percentile(s_map, 100.0 - float(args.top_percentile)))
                pred = s_map >= thr

            if not gt.any():
                iou = 0.0 if pred.any() else float("nan")
                inter = 0
                union = int(pred.sum())
            else:
                inter = int(np.logical_and(pred, gt).sum())
                union = int(np.logical_or(pred, gt).sum())
                iou = _binary_iou(pred, gt)

            if not np.isnan(iou):
                per_class_frame_ious[cn].append(float(iou))
            per_class_inter[cn] += inter
            per_class_union[cn] += union
            per_class_pred_pos[cn] += int(pred.sum())
            per_class_gt_pos[cn] += int(gt.sum())

    # Aggregate metrics
    rows = []
    macro_vals = []
    precision_macro_vals = []
    recall_macro_vals = []
    f1_macro_vals = []
    total_inter = 0
    total_union = 0
    total_pred_pos = 0
    total_gt_pos = 0
    for cn in class_names:
        ious = per_class_frame_ious.get(cn, [])
        macro = float(np.mean(ious)) if ious else float("nan")
        tp = int(per_class_inter.get(cn, 0))
        union = int(per_class_union.get(cn, 0))
        pred_pos = int(per_class_pred_pos.get(cn, 0))
        gt_pos = int(per_class_gt_pos.get(cn, 0))
        fp = pred_pos - tp
        fn = gt_pos - tp
        micro = _safe_div(float(tp), float(union))
        precision = _safe_div(float(tp), float(tp + fp))
        recall = _safe_div(float(tp), float(tp + fn))
        if np.isnan(precision) or np.isnan(recall):
            f1 = float("nan")
        else:
            f1 = _safe_div(2.0 * precision * recall, precision + recall)
        if not np.isnan(macro):
            macro_vals.append(macro)
        if not np.isnan(precision):
            precision_macro_vals.append(precision)
        if not np.isnan(recall):
            recall_macro_vals.append(recall)
        if not np.isnan(f1):
            f1_macro_vals.append(f1)
        if union > 0:
            total_inter += tp
            total_union += union
        total_pred_pos += pred_pos
        total_gt_pos += gt_pos
        rows.append(
            {
                "class_name": cn,
                "class_id": _resolve_class_id(cn, name_to_id),
                "n_pairs": len(ious),
                "macro_iou": macro,
                "micro_iou": micro,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "inter": tp,
                "union": union,
            }
        )

    total_fp = total_pred_pos - total_inter
    total_fn = total_gt_pos - total_inter
    precision_micro_global = _safe_div(float(total_inter), float(total_inter + total_fp))
    recall_micro_global = _safe_div(float(total_inter), float(total_inter + total_fn))
    if np.isnan(precision_micro_global) or np.isnan(recall_micro_global):
        f1_micro_global = float("nan")
    else:
        f1_micro_global = _safe_div(2.0 * precision_micro_global * recall_micro_global, precision_micro_global + recall_micro_global)

    result = {
        "scene": args.scene,
        "result_dir": str(result_dir),
        "lang_field": str(lang_field),
        "checkpoint": str(checkpoint),
        "results_habitat": str(results_habitat),
        "traj_txt": str(traj_txt),
        "traj_format": args.traj_format,
        "latent_dim": int(latent_dim),
        "threshold_mode": args.threshold_mode,
        "threshold": args.threshold if args.threshold_mode == "fixed" else None,
        "top_percentile": args.top_percentile if args.threshold_mode == "quantile" else None,
        "classes": class_names,
        "num_frames_available": int(num_pairs_available),
        "num_frames_eval": len(pairs),
        "skipped_no_pose": int(skipped_no_pose),
        "skipped_bad_sem_shape": int(skipped_bad_sem_shape),
        "per_class": rows,
        "mIoU_macro": float(np.mean(macro_vals)) if macro_vals else float("nan"),
        "IoU_micro_global": (float(total_inter) / float(total_union)) if total_union > 0 else float("nan"),
        "precision_macro": float(np.mean(precision_macro_vals)) if precision_macro_vals else float("nan"),
        "recall_macro": float(np.mean(recall_macro_vals)) if recall_macro_vals else float("nan"),
        "f1_macro": float(np.mean(f1_macro_vals)) if f1_macro_vals else float("nan"),
        "precision_micro_global": precision_micro_global,
        "recall_micro_global": recall_micro_global,
        "f1_micro_global": f1_micro_global,
    }

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (result_dir / "lang_field_seg_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.scene}_seg_metrics.json"
    csv_path = out_dir / f"{args.scene}_seg_metrics.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_name",
                "class_id",
                "n_pairs",
                "macro_iou",
                "micro_iou",
                "precision",
                "recall",
                "f1",
                "tp",
                "fp",
                "fn",
                "inter",
                "union",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Frames evaluated: {len(pairs)} / {num_pairs_available} (skipped_no_pose={skipped_no_pose})")
    if len(pairs) < num_pairs_available:
        print("Warning: frame evaluation is limited; use --max_frames -1 for all available frames.")
    if skipped_bad_sem_shape:
        print(f"Skipped due to bad semantic shape/resolution mismatch: {skipped_bad_sem_shape}")
    print(f"mIoU macro: {result['mIoU_macro']:.4f}")
    print(f"IoU micro global: {result['IoU_micro_global']:.4f}")
    print(f"F1 micro global: {result['f1_micro_global']:.4f}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()

