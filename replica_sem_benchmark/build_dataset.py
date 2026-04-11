#!/usr/bin/env python3
"""
Собирает локальную копию RGB + semantic в папки:

    replica_sem_benchmark/images/*.jpg
    replica_sem_benchmark/semantic/*.npy

и пишет ``replica_sem_benchmark/manifest.json`` с относительными путями.

Источник по умолчанию: ``data/replica_sim_nvs/<scene>/results_habitat/`` (как в
``scripts/prepare_replica_benchmark.py``).

Ключевая цель сборки:
    - нормальная ориентация RGB (без неожиданных поворотов)
    - разметка semantic соответствует RGB по кадру
    - выборка покрывает разные сцены и (по возможности) все классы, которые вообще
      встречаются в доступных semantic-картах

Запуск из New-Proj::

    python Tests/replica_sem_benchmark/build_dataset.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

_PROJ = Path(__file__).resolve().parents[1]  # New-Proj
_HERE = Path(__file__).resolve().parent

_DEFAULT_SCENES = "office0,office1,office2,office3,office4,room0,room1"


def _num_key(p: Path) -> int:
    import re

    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def _glob_scene_pairs(scene_root: Path) -> list[tuple[Path, Path, int]]:
    """
    Return list of (rgb_path, sem_path, frame_idx) for one scene.
    frame_idx is derived from filename order (stable).
    """
    hab = scene_root / "results_habitat"
    rgbs = sorted(hab.glob("frame*.jpg"), key=_num_key)
    sems = sorted((hab / "semantic").glob("semantic_map_*.npy"), key=_num_key)
    if not rgbs or len(rgbs) != len(sems):
        return []
    out: list[tuple[Path, Path, int]] = []
    for i, (r, s) in enumerate(zip(rgbs, sems)):
        out.append((r.resolve(), s.resolve(), i))
    return out


def _class_set_from_sem(sem_path: Path) -> set[int]:
    import numpy as np

    sem = np.load(str(sem_path))
    u = np.unique(sem.astype("int64"))
    return {int(x) for x in u.tolist() if int(x) > 0}


def _evenly_spaced_idxs(n: int, k: int) -> list[int]:
    if k <= 0 or n <= 0:
        return []
    if k >= n:
        return list(range(n))
    import numpy as np

    return np.linspace(0, n - 1, k, dtype=int).tolist()


def _greedy_cover(
    candidates: list[dict],
    *,
    target_classes: set[int],
    max_select: int,
) -> list[dict]:
    """Greedy set cover: pick frames that add most uncovered classes."""
    uncovered = set(target_classes)
    picked: list[dict] = []
    remaining = candidates[:]
    while uncovered and remaining and len(picked) < max_select:
        best_i = -1
        best_gain = -1
        for i, c in enumerate(remaining):
            gain = len(uncovered.intersection(c["classes"]))
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i < 0 or best_gain <= 0:
            break
        c = remaining.pop(best_i)
        picked.append(c)
        uncovered -= set(c["classes"])
    return picked


def _rm_tree_contents(d: Path) -> None:
    if not d.exists():
        return
    for p in d.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        type=Path,
        default=Path("data/replica_sim_nvs"),
        help="Root with scene folders (default: data/replica_sim_nvs).",
    )
    ap.add_argument(
        "--scenes",
        type=str,
        default=_DEFAULT_SCENES,
        help=f"Scenes to include (comma-separated). Default: {_DEFAULT_SCENES}",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=0,
        help=(
            "How many frames to select total. "
            "0 means 'auto': pick enough to cover all classes, then add a small balanced set."
        ),
    )
    ap.add_argument(
        "--min_per_scene",
        type=int,
        default=10,
        help="Minimum frames per scene for diversity (default: 10).",
    )
    ap.add_argument(
        "--scan_stride",
        type=int,
        default=1,
        help="When scanning semantic maps for class coverage, use every k-th frame (default: 1 = scan all).",
    )
    ap.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG quality for copied images (default: 95).",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing replica_sem_benchmark/images and semantic contents before rebuilding.",
    )
    args = ap.parse_args()

    data_root = (_PROJ / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root.resolve()
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    if not scenes:
        raise SystemExit("No scenes specified")

    img_dir = _HERE / "images"
    sem_dir = _HERE / "semantic"
    img_dir.mkdir(parents=True, exist_ok=True)
    sem_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        _rm_tree_contents(img_dir)
        _rm_tree_contents(sem_dir)

    # 1) Collect all candidate frames across scenes (paired rgb+semantic).
    candidates: list[dict] = []
    for scene in scenes:
        pairs = _glob_scene_pairs(data_root / scene)
        if not pairs:
            print(f"[skip] {scene}: no paired rgb/semantic under {data_root/scene}", file=sys.stderr)
            continue
        # For diversity fill, pre-pick evenly spaced indices.
        idxs_div = set(_evenly_spaced_idxs(len(pairs), max(1, int(args.min_per_scene))))
        stride = max(1, int(args.scan_stride))
        for j, (rgb_p, sem_p, fi) in enumerate(pairs):
            if (j % stride) != 0 and (j not in idxs_div):
                continue
            try:
                cls = _class_set_from_sem(sem_p)
            except Exception as e:
                print(f"[skip] cannot read semantic {sem_p}: {e}", file=sys.stderr)
                continue
            candidates.append(
                {
                    "scene": scene,
                    "frame_index": int(fi),
                    "rgb": rgb_p,
                    "semantic": sem_p,
                    "classes": sorted(cls),
                    "is_diverse": (j in idxs_div),
                }
            )

    if not candidates:
        raise SystemExit(f"No candidates found under {data_root}. Check --data_root/--scenes.")

    # 2) Determine target class universe = union over candidates.
    target_classes: set[int] = set()
    for c in candidates:
        target_classes.update(c["classes"])
    if not target_classes:
        raise SystemExit("No semantic classes found in candidates (all semantic maps empty?)")

    # 3) Selection strategy:
    #    - ensure at least min_per_scene (diversity set)
    #    - greedy cover remaining classes
    #    - if args.n > 0, cap to n and fill with additional diverse frames
    selected: list[dict] = []

    # 3a) diversity baseline: evenly spaced per scene (from candidates marked is_diverse)
    by_scene: dict[str, list[dict]] = {}
    for c in candidates:
        by_scene.setdefault(c["scene"], []).append(c)
    for scene in scenes:
        pool = [c for c in by_scene.get(scene, []) if c["is_diverse"]]
        if not pool:
            continue
        # keep unique by (rgb,semantic)
        seen = set()
        for c in pool:
            key = (str(c["rgb"]), str(c["semantic"]))
            if key in seen:
                continue
            seen.add(key)
            selected.append(c)

    # 3b) greedy class cover, excluding already selected
    sel_keys = {(str(c["rgb"]), str(c["semantic"])) for c in selected}
    remaining = [c for c in candidates if (str(c["rgb"]), str(c["semantic"])) not in sel_keys]
    covered = set()
    for c in selected:
        covered.update(c["classes"])
    need = target_classes - covered
    cap = int(args.n) if int(args.n) > 0 else 10_000_000
    more = _greedy_cover(remaining, target_classes=need, max_select=max(0, cap - len(selected)))
    selected.extend(more)

    # 3c) If args.n is set and we still have room, add more diverse frames.
    if int(args.n) > 0 and len(selected) < int(args.n):
        # Prefer frames with many classes, but keep scene balance
        remaining2 = [c for c in remaining if c not in more]
        remaining2.sort(key=lambda x: (len(x["classes"]), x["scene"]), reverse=True)
        sel_scene_counts = {s: 0 for s in scenes}
        for c in selected:
            sel_scene_counts[c["scene"]] = sel_scene_counts.get(c["scene"], 0) + 1
        for c in remaining2:
            if len(selected) >= int(args.n):
                break
            # weak balancing: prefer scenes with fewer selected
            if sel_scene_counts.get(c["scene"], 0) > (int(args.n) // max(len(scenes), 1) + 2):
                continue
            selected.append(c)
            sel_scene_counts[c["scene"]] = sel_scene_counts.get(c["scene"], 0) + 1

    # 3d) If args.n==0 (auto), keep selection as-is but cap to something reasonable
    #      while preserving class coverage; default target is 100 like old benchmark.
    if int(args.n) == 0:
        target_n = 100
        if len(selected) > target_n:
            # Keep first part (diversity + greedy cover), then truncate.
            selected = selected[:target_n]

    # Report coverage
    final_classes = set()
    for c in selected:
        final_classes.update(c["classes"])
    missing = sorted(target_classes - final_classes)
    if missing:
        print(f"[warn] missing {len(missing)} class ids in selection (increase --n or reduce --scan_stride).", file=sys.stderr)

    # 4) Copy files: RGB is re-encoded (ensures consistent orientation as stored), semantic is copied.
    try:
        import cv2
    except Exception as e:
        print(f"OpenCV required for dataset build: {e}", file=sys.stderr)
        sys.exit(1)

    out_samples: list[dict] = []
    for sid, s in enumerate(selected):
        rgb_src: Path = Path(s["rgb"])
        sem_src: Path = Path(s["semantic"])
        rgb_dst = img_dir / f"{sid:05d}.jpg"
        sem_dst = sem_dir / f"{sid:05d}.npy"

        bgr = cv2.imread(str(rgb_src))
        if bgr is None:
            print(f"[skip] cannot read rgb {rgb_src}", file=sys.stderr)
            continue
        ok = cv2.imwrite(str(rgb_dst), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
        if not ok:
            raise RuntimeError(f"Cannot write image: {rgb_dst}")
        shutil.copy2(sem_src, sem_dst)

        out_samples.append(
            {
                "id": int(sid),
                "scene": s["scene"],
                "frame_index": int(s["frame_index"]),
                "rgb": str(Path("images") / rgb_dst.name),
                "semantic": str(Path("semantic") / sem_dst.name),
            }
        )

    manifest = {
        "version": 1,
        "n_samples": len(out_samples),
        "info_semantic": "data/replica_v1/office_0/habitat/info_semantic.json",
        "note": (
            "Replica semantic benchmark subset. RGB copied with cv2.imwrite (no rotation by default) "
            "to ensure consistent orientation; semantic maps copied 1:1 from source."
        ),
        "samples": out_samples,
    }
    out_path = _HERE / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Built {len(out_samples)} pairs → {img_dir} , {sem_dir}\n"
        f"Scenes: {', '.join(sorted({s['scene'] for s in out_samples}))}\n"
        f"Class coverage: {len(final_classes)}/{len(target_classes)} classes\n"
        f"Wrote {out_path}"
    )


if __name__ == "__main__":
    main()
