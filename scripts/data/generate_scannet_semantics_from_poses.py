#!/usr/bin/env python3
"""
Render semantic maps for existing ScanNet NVS poses.

This script does NOT change RGB/depth/pose. It only adds semantic labels aligned
with current frames/poses:
  data/scannet_sim_nvs/<scene>/results_habitat/semantic/semanticXXXXXX.npy
  data/scannet_sim_nvs/<scene>/semantic/semanticXXXXXX.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.habitat_egl_bootstrap import bootstrap_habitat_egl  # noqa: E402
from src.simulator import init_simulator  # noqa: E402
from src.utils.general_utils import InfoPrinter  # noqa: E402


DEFAULT_SCENES = [
    "scene0000_00", "scene0005_00", "scene0010_00",
    "scene0050_02", "scene0144_01", "scene0221_01", "scene0300_01",
    "scene0354_00", "scene0389_00", "scene0423_02", "scene0427_00",
    "scene0494_00", "scene0616_00", "scene0645_02", "scene0693_00",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ScanNet semantic maps for existing NVS poses.")
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing semantic files.")
    parser.add_argument(
        "--no-semantic-ply",
        action="store_true",
        help="Disable semantic .ply scene_id override (not recommended for ScanNet).",
    )
    return parser.parse_args()


def _make_semantic_ply_habitat_cfg(scene: str) -> Path:
    """Create a temporary habitat config that renders from semantic .ply directly."""
    habitat_cfg = ROOT / "configs" / "ScanNet" / scene / "habitat.py"
    content = habitat_cfg.read_text(encoding="utf-8")
    pattern = r"scene_id\s*=\s*os\.path\.join\([^)]*scannet\.stage_config\.json[^)]*\),"
    replacement = (
        "scene_id=os.path.join("
        "\"data\", \"ScanNet\", \"habitat\", scene_name, f\"{scene_name}_semantic.ply\"),"
    )
    patched, n = re.subn(pattern, replacement, content, count=1)
    if n != 1:
        raise RuntimeError(f"Failed to patch scene_id in {habitat_cfg}")

    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tmp.write(patched)
    tmp.close()
    return Path(tmp.name)


def _pick_scene_cfg(scene: str) -> Path:
    cfg_dir = ROOT / "configs" / "ScanNet" / scene
    for name in ("ActiveOpenSemGeom.py", "ActiveOpenSem.py", "ActiveOpenSem_base.py"):
        path = cfg_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No ActiveOpenSem* config under {cfg_dir}")


def render_scene(scene: str, overwrite: bool, use_semantic_ply: bool) -> None:
    if not bootstrap_habitat_egl(strict=True):
        raise RuntimeError(
            "Habitat EGL bootstrap failed. Run: bash docker/ensure_habitat_egl.sh"
        )

    cfg_path = _pick_scene_cfg(scene)
    cfg = Config.fromfile(str(cfg_path))
    temp_habitat_cfg = None
    if use_semantic_ply:
        temp_habitat_cfg = _make_semantic_ply_habitat_cfg(scene)
        cfg.sim.habitat_cfg = str(temp_habitat_cfg)
    sim = init_simulator(cfg, InfoPrinter("ScanNetSemantic"))

    base = ROOT / "data" / "scannet_sim_nvs" / scene
    pose_dir = base / "pose"
    pose_files = sorted(pose_dir.glob("*.txt"))
    if not pose_files:
        raise RuntimeError(f"No poses found in {pose_dir}")

    sem_dir_1 = base / "results_habitat" / "semantic"
    sem_dir_2 = base / "semantic"
    sem_dir_1.mkdir(parents=True, exist_ok=True)
    sem_dir_2.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    bad_frames = 0
    for i, pose_file in enumerate(pose_files):
        out1 = sem_dir_1 / f"semantic{i:06d}.npy"
        out2 = sem_dir_2 / f"semantic{i:06d}.npy"
        if (not overwrite) and out1.exists() and out2.exists():
            sem_chk = np.load(out1).astype(np.int64)
            if np.any(sem_chk != 0):
                skipped += 1
                continue

        pose = np.loadtxt(pose_file).astype(np.float32)
        sim_out = sim.simulate(pose, return_semantic=True, no_print=True)
        sem = sim_out["seman"]
        if sem is None:
            raise RuntimeError(f"Semantic tensor is None for {scene} frame {i}")
        sem_np = np.rint(sem.detach().cpu().numpy()).astype(np.int32)
        if not np.any(sem_np != 0):
            bad_frames += 1
            if i < 3:
                print(f"[warn] {scene} frame {i}: semantic map is all zeros", file=sys.stderr)
        np.save(out1, sem_np)
        np.save(out2, sem_np)
        saved += 1

    try:
        sim.sim.close()
    except Exception:
        pass
    if temp_habitat_cfg is not None:
        temp_habitat_cfg.unlink(missing_ok=True)

    sample = sem_dir_1 / "semantic000000.npy"
    if sample.is_file():
        uniq = np.unique(np.load(sample).astype(np.int64))
        print(f"{scene}: sample frame0 unique ids (first 20): {uniq[:20].tolist()}")

    if saved > 0 and bad_frames == saved:
        raise RuntimeError(
            f"{scene}: all rendered semantic maps are zero. "
            "Check data/ScanNet/habitat/{scene}/{scene}_semantic.ply and NVS poses."
        )
    print(f"{scene}: poses={len(pose_files)} saved={saved} skipped={skipped} all_zero={bad_frames}")


def main() -> None:
    args = parse_args()
    for scene in args.scenes:
        render_scene(scene, overwrite=args.overwrite, use_semantic_ply=not args.no_semantic_ply)
    print("Done.")


if __name__ == "__main__":
    main()

