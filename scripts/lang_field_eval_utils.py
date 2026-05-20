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
