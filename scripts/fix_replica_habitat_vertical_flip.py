#!/usr/bin/env python3
"""
Разовая правка уже сохранённых кадров (legacy): вертикальный flip.

Для новых прогонов используйте ``pinhole_vertical_flip`` в ``configs/Replica/.../habitat.py`` и
``HabitatSim.simulate`` — повторный flip здесь перевернёт кадры обратно.

Обрабатывает в каталоге ``results_habitat`` (или аналоге):
  - ``frame*.jpg``
  - ``depth*.png`` (uint16)
  - ``semantic/semantic_map_*.npy``

Запуск из New-Proj::

    python scripts/fix_replica_habitat_vertical_flip.py \\
        --dir data/replica_sim_nvs/office0/results_habitat

    # или бенч:
    python scripts/fix_replica_habitat_vertical_flip.py \\
        --dir replica_sem_benchmark --benchmark_layout

Повторный запуск снова перевернёт кадры — не запускайте дважды на тех же файлах.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _flip_jpg_png(path: Path) -> None:
    from PIL import Image

    arr = np.array(Image.open(path))
    if arr.ndim < 2:
        return
    out = np.flipud(arr)
    Image.fromarray(out).save(path)


def _flip_npy(path: Path) -> None:
    a = np.load(path)
    np.save(path, np.flipud(a))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Каталог: results_habitat или replica_sem_benchmark при --benchmark_layout",
    )
    p.add_argument(
        "--benchmark_layout",
        action="store_true",
        help="Ожидать подпапки images/*.jpg и semantic/*.npy рядом с manifest",
    )
    args = p.parse_args()
    root = args.dir.resolve()
    if not root.is_dir():
        print(f"Нет каталога: {root}", file=sys.stderr)
        sys.exit(1)

    n_img = n_dep = n_sem = 0
    if args.benchmark_layout:
        img_dir = root / "images"
        sem_dir = root / "semantic"
        if img_dir.is_dir():
            for f in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg")):
                _flip_jpg_png(f)
                n_img += 1
        if sem_dir.is_dir():
            for f in sorted(sem_dir.glob("*.npy")):
                _flip_npy(f)
                n_sem += 1
    else:
        for f in sorted(root.glob("frame*.jpg")):
            _flip_jpg_png(f)
            n_img += 1
        for f in sorted(root.glob("depth*.png")):
            _flip_jpg_png(f)
            n_dep += 1
        sem_dir = root / "semantic"
        if sem_dir.is_dir():
            for f in sorted(sem_dir.glob("semantic_map_*.npy")):
                _flip_npy(f)
                n_sem += 1

    print(f"done  rgb={n_img}  depth={n_dep}  semantic={n_sem}  root={root}")


if __name__ == "__main__":
    main()
