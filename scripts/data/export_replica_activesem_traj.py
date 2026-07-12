#!/usr/bin/env python3
"""Export ActiveSem exploration poses → data/replica_activesem_traj/<scene>/traj.txt.

Replica ``traj.txt`` uses camera-to-world **RDF** (see ``PoseLoader.load_Replica_pose``).
``exploration_path_poses.json`` from activesgm is Habitat **RUB** — flip Y/Z columns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _load_poses_json(path: Path) -> list[np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "poses_c2w" in data:
            raw = data["poses_c2w"]
        elif "poses" in data:
            raw = data["poses"]
        else:
            raise ValueError(f"Unrecognized keys in {path}: {list(data.keys())}")
    elif isinstance(data, list):
        raw = data
    else:
        raise ValueError(f"Expected list or dict in {path}")

    return [np.asarray(item, dtype=np.float64).reshape(4, 4) for item in raw]


def _rub_to_rdf(c2w_rub: np.ndarray) -> np.ndarray:
    """Inverse of PoseLoader.load_Replica_pose (RDF → RUB)."""
    out = c2w_rub.copy()
    out[:3, 1] *= -1.0
    out[:3, 2] *= -1.0
    return out


def _write_traj(out_path: Path, poses_rdf: list[np.ndarray]) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(f"{x:.8f}" for x in mat.reshape(-1)) for mat in poses_rdf]
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export ActiveSem traj for Replica passive replay.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--run-tag", default="run_0")
    parser.add_argument("--poses-json", default=None)
    parser.add_argument("--out-traj", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    poses_json = Path(args.poses_json) if args.poses_json else (
        ROOT / "results" / "Replica" / args.scene / "ActiveSem" / args.run_tag / "exploration_path_poses.json"
    )
    out_traj = Path(args.out_traj) if args.out_traj else (
        ROOT / "data" / "replica_activesem_traj" / args.scene / "traj.txt"
    )

    if out_traj.is_file() and not args.force:
        n = sum(1 for ln in out_traj.read_text(encoding="utf-8").splitlines() if ln.strip())
        print(f"SKIP: {out_traj} exists ({n} poses). Use --force to overwrite.")
        return 0

    if not poses_json.is_file():
        print(f"ERROR: missing {poses_json}", file=sys.stderr)
        print("  Run ActiveSem first: results/Replica/<scene>/ActiveSem/run_0/", file=sys.stderr)
        return 1

    poses_rub = _load_poses_json(poses_json)
    if not poses_rub:
        print(f"ERROR: empty poses in {poses_json}", file=sys.stderr)
        return 1

    poses_rdf = [_rub_to_rdf(p) for p in poses_rub]
    n = _write_traj(out_traj, poses_rdf)
    print(f"Wrote {n} poses (RUB→RDF) → {out_traj}")
    print(f"  source: {poses_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
