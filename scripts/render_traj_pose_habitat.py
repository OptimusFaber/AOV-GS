#!/usr/bin/env python3
"""
Render one RGB frame from Habitat at a pose decoded from ``replica_sim_nvs/.../traj.txt``.

EN: ``traj.txt`` stores absolute RDF camera-to-world poses in the same world frame as the
training ``data/Replica/<scene>/traj.txt``.  To recover the Habitat/RUB pose, simply negate
columns 1 and 2 (Y and Z axes) of the stored matrix.

RU: Рендер одного кадра Habitat по строке из ``traj.txt``. Позы хранятся в абсолютной системе
координат RDF (та же, что у тренировочного ``traj.txt``).  Для рендера в Habitat нужно инвертировать
оси Y и Z.  Сохраняются два варианта: ``*_raw`` — порядок строк как из симулятора; ``*_dataset`` —
после ``np.flipud``, как ``frame*.jpg`` при генерации.

EN (orientation): ``*_raw`` is the tensor row order from Habitat/OpenGL; many viewers show it
flipped vertically vs. a normal photo. ``*_dataset`` matches ``generate_Replica_NVS_data`` (flipud
when saving). Use ``*_dataset`` to compare with on-disk training RGB — not ``*_raw``.

Example / Пример::

  cd New-Proj
  python scripts/render_traj_pose_habitat.py \\
    --cfg configs/Replica/office0/generate_nvs_data.py \\
    --traj data/replica_sim_nvs/office0/traj.txt \\
    --pose_idx 0 \\
    --out results/debug_habitat_pose.png

Note / Замечание: индекс строки ``traj.txt`` не всегда совпадает с ``frameXXXXXX.jpg``, если при
генерации часть кадров пропускалась (too close).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.naruto.cfg_loader import load_cfg
from src.simulator import init_simulator
from src.utils.general_utils import InfoPrinter, fix_random_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Habitat render from replica_sim_nvs traj.txt pose")
    p.add_argument(
        "--cfg",
        type=str,
        default="configs/Replica/office0/generate_nvs_data.py",
        help="Same config as for generate_Replica_NVS_data (scene, sim, planner).",
    )
    p.add_argument(
        "--traj",
        type=str,
        default="data/replica_sim_nvs/office0/traj.txt",
        help="Path to traj.txt produced by generate_Replica_NVS_data.",
    )
    p.add_argument("--pose_idx", type=int, default=0, help="0-based line index in traj.txt.")
    p.add_argument(
        "--out",
        type=str,
        default="results/habitat_traj_pose_render.png",
        help="Output base path; writes *_raw and *_dataset next to this stem.",
    )
    p.add_argument("--result_dir", type=str, default=None, help="Override cfg.dirs.result_dir.")
    p.add_argument("--seed", type=int, default=None, help="Override cfg.general.seed.")
    return p.parse_args()


def _load_cfg_ns(args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace(
        cfg=args.cfg,
        result_dir=args.result_dir,
        seed=args.seed,
        enable_vis=None,
        use_clip=None,
        stage="final",
        debug=False,
    )
    return ns


def main() -> None:
    args = _parse_args()
    os.chdir(ROOT)
    cfg = load_cfg(_load_cfg_ns(args))
    fix_random_seed(getattr(cfg.general, "seed", 0))

    info_printer = InfoPrinter("RenderTrajHabitat")
    sim = init_simulator(cfg, info_printer)

    traj_path = Path(args.traj)
    if not traj_path.is_file():
        raise FileNotFoundError(traj_path)
    lines = traj_path.read_text().strip().splitlines()
    if args.pose_idx < 0 or args.pose_idx >= len(lines):
        raise IndexError(f"pose_idx={args.pose_idx} out of range [0, {len(lines) - 1}]")

    # traj.txt stores absolute RDF c2w poses (same convention as training traj.txt).
    # Habitat expects RUB: negate columns 1 (Y) and 2 (Z).
    c2w_stored = np.array(list(map(float, lines[args.pose_idx].split())), dtype=np.float64).reshape(4, 4)
    c2w_rub = c2w_stored.copy()
    c2w_rub[:3, 1] *= -1  # RDF → RUB
    c2w_rub[:3, 2] *= -1
    c2w_np = c2w_rub

    out = sim.simulate(c2w_np, no_print=True)
    color = out["color"]
    if color is None:
        raise RuntimeError("Simulator returned no color.")

    h, w = color.shape[:2]
    bgr_raw = (color.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    bgr_raw = cv2.cvtColor(bgr_raw, cv2.COLOR_RGB2BGR)

    bgr_ds = np.flipud(bgr_raw).copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.parent / out_path.stem
    suf = out_path.suffix if out_path.suffix else ".png"
    raw_p = f"{stem}_raw{suf}"
    ds_p = f"{stem}_dataset{suf}"
    cv2.imwrite(str(raw_p), bgr_raw)
    cv2.imwrite(str(ds_p), bgr_ds)

    print(f"Saved (*_raw = sim row order; may look upside-down in viewers): {raw_p}")
    print(f"Saved (*_dataset = flipud, matches frame*.jpg / training): {ds_p}")
    print(f"pose_idx={args.pose_idx}  image size {w}x{h}")


if __name__ == "__main__":
    main()
