#!/usr/bin/env python3
"""
Собирает manifest для бенчмарка: ~N кадров RGB + GT semantic из replica_sim_nvs,
равномерно по сценам (office*/room*).

Запуск из корня New-Proj::

    python scripts/prepare_replica_benchmark.py --n 100 --out data/benchmarks/replica_sem100/manifest.json

Пути в manifest относительно каталога, где лежит manifest (родитель = out.parent),
либо задайте --root для префикса.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _num_key(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def _glob_frames(scene_root: Path) -> tuple[list[Path], list[Path]]:
    hab = scene_root / "results_habitat"
    rgbs = sorted(hab.glob("frame*.jpg"), key=_num_key)
    sems = sorted((hab / "semantic").glob("semantic_map_*.npy"), key=_num_key)
    return list(rgbs), list(sems)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100, help="Всего кадров (по умолчанию 100)")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/benchmarks/replica_sem100/manifest.json"),
        help="Куда записать manifest.json",
    )
    p.add_argument(
        "--data_root",
        type=Path,
        default=Path("data/replica_sim_nvs"),
        help="Корень с подпапками сцен",
    )
    p.add_argument(
        "--scenes",
        type=str,
        default="office0,office1,office2,office3,office4,room0,room1",
        help="Сцены через запятую (room2 пустая — не включать)",
    )
    args = p.parse_args()

    proj = Path(__file__).resolve().parents[1]
    data_root = (proj / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    out_path = (proj / args.out).resolve() if not args.out.is_absolute() else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    n_scenes = len(scenes)
    if n_scenes == 0:
        raise SystemExit("Нет сцен")

    # Распределение кадров: первые (n % k) сцен получают +1 кадр
    n_total = args.n
    base = n_total // n_scenes
    rem = n_total % n_scenes
    per_scene = [base + (1 if i < rem else 0) for i in range(n_scenes)]

    samples: list[dict] = []
    sid = 0

    for scene, k in zip(scenes, per_scene):
        scene_root = data_root / scene
        rgbs, sems = _glob_frames(scene_root)
        if len(rgbs) == 0 or len(rgbs) != len(sems):
            print(f"[skip] {scene}: rgb={len(rgbs)} sem={len(sems)}")
            continue
        n = len(rgbs)
        if k >= n:
            idxs = list(range(n))
        else:
            import numpy as np

            idxs = np.linspace(0, n - 1, k, dtype=int).tolist()
        for fi in idxs:
            rgb_p = rgbs[fi].resolve()
            sem_p = sems[fi].resolve()
            samples.append(
                {
                    "id": sid,
                    "scene": scene,
                    "frame_index": int(fi),
                    "rgb": str(rgb_p),
                    "semantic": str(sem_p),
                }
            )
            sid += 1

    # Пути в manifest — относительно New-Proj (короткие)
    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(proj))
        except ValueError:
            return str(p)

    for s in samples:
        s["rgb"] = rel(Path(s["rgb"]))
        s["semantic"] = rel(Path(s["semantic"]))

    manifest = {
        "version": 1,
        "n_samples": len(samples),
        "info_semantic": "data/replica_v1/office_0/habitat/info_semantic.json",
        "note": "Классы Replica: имена как в info_semantic (например chair, base-cabinet).",
        "samples": samples,
    }
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(samples)} samples → {out_path}")


if __name__ == "__main__":
    main()
