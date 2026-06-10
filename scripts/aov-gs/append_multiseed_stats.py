#!/usr/bin/env python3
"""Record per-stage NVS metrics from one run; upsert multiseed CSV; optionally delete run dir."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "results" / "Replica" / "multiseed_experiments.csv"

CSV_FIELDS = [
    "Experiment",
    "Seed",
    "Scene",
    "step_stage0",
    "step_stage1",
    "step_refinement",
    "exploration_stage_I",
    "exploration_stage_II",
    "refinement_stage",
    "num_gaussians",
]

KEY_FIELDS = ("Experiment", "Seed", "Scene")


def parse_render(p: Path) -> dict[str, float] | None:
    if not p.exists():
        return None
    out: dict[str, float] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = float(v.strip())
    return out


def get_step(p: Path, stage: int) -> int | None:
    if not p.exists():
        return None
    key = f"exploration_stage_{stage}_step"
    for line in p.read_text(encoding="utf-8").splitlines():
        if key in line:
            return int(line.split(":")[1].strip())
    return None


def get_step_refinement(run_dir: Path) -> int | None:
    p = run_dir / "exploration_path_poses.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        n = int(data["num_poses"])
        return n - 1 if n > 0 else None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def num_gaussians(run_dir: Path) -> int | None:
    try:
        import numpy as np
    except ImportError:
        return None
    for rel in ("splatam/final/params0.npz", "splatam/final/params.npz"):
        p = run_dir / rel
        if not p.exists():
            continue
        data = np.load(str(p))
        if "means3D" in data:
            return int(data["means3D"].shape[0])
    return None


def fmt_stage(m: dict[str, float] | None) -> str:
    if not m:
        return "—"
    return (
        f"PSNR {m['psnr']:.2f} | SSIM {m['ssim']:.3f} | "
        f"LPIPS {m['lpips']:.3f} | L1 {m['l1(cm)']:.2f} cm"
    )


def collect_row(experiment: str, seed: int, scene: str, run_dir: Path) -> dict[str, str]:
    fin = parse_render(run_dir / "splatam/eval_final/render_result.txt")
    if fin is None:
        raise RuntimeError(f"missing eval_final in {run_dir}")

    s0 = parse_render(run_dir / "splatam/eval_exploration_stage_0/render_result.txt")
    s1 = parse_render(run_dir / "splatam/eval_exploration_stage_1/render_result.txt")
    st0 = get_step(run_dir / "splatam/eval_exploration_stage_0/exploration_info.txt", 0)
    st1 = get_step(run_dir / "splatam/eval_exploration_stage_1/exploration_info.txt", 1)
    st_ref = get_step_refinement(run_dir)
    ng = num_gaussians(run_dir)

    return {
        "Experiment": experiment,
        "Seed": str(seed),
        "Scene": scene,
        "step_stage0": "" if st0 is None else str(st0),
        "step_stage1": "" if st1 is None else str(st1),
        "step_refinement": "" if st_ref is None else str(st_ref),
        "exploration_stage_I": fmt_stage(s0),
        "exploration_stage_II": fmt_stage(s1),
        "refinement_stage": fmt_stage(fin),
        "num_gaussians": "" if ng is None else str(ng),
    }


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(row[k] for k in KEY_FIELDS)


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(csv_path.parent), prefix=".multiseed_", suffix=".csv.tmp"
    )
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        tmp_path.replace(csv_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def upsert_row(csv_path: Path, new_row: dict[str, str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = csv_path.with_suffix(".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        rows = load_rows(csv_path)
        key = row_key(new_row)
        rows = [r for r in rows if row_key(r) != key]
        rows.append(new_row)
        rows.sort(key=lambda r: (r["Experiment"], int(r["Seed"]), r["Scene"]))
        write_rows(csv_path, rows)
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def is_recorded(csv_path: Path, experiment: str, seed: int, scene: str) -> bool:
    if not csv_path.exists():
        return False
    target = (experiment, str(seed), scene)
    for row in load_rows(csv_path):
        if row_key(row) == target:
            return True
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--experiment", help="ActiveGeom | ActiveOpenSem")
    p.add_argument("--seed", type=int)
    p.add_argument("--scene")
    p.add_argument("--run-dir", type=Path, help="Path to seed_N run folder")
    p.add_argument(
        "--delete",
        type=int,
        default=1,
        choices=(0, 1),
        help="Remove run-dir after successful record (default 1)",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Exit 0 if (experiment, seed, scene) already in CSV",
    )
    args = p.parse_args()

    if args.check:
        if not args.experiment or args.seed is None or not args.scene:
            print("--check requires --experiment --seed --scene", file=sys.stderr)
            return 2
        return 0 if is_recorded(args.csv, args.experiment, args.seed, args.scene) else 1

    if not args.experiment or args.seed is None or not args.scene or not args.run_dir:
        p.error("--experiment --seed --scene --run-dir are required")

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"run dir not found: {run_dir}", file=sys.stderr)
        return 3

    row = collect_row(args.experiment, args.seed, args.scene, run_dir)
    upsert_row(args.csv, row)
    print(
        f"recorded {args.experiment} scene={args.scene} seed={args.seed} "
        f"→ {args.csv}",
        flush=True,
    )

    if args.delete:
        shutil.rmtree(run_dir)
        print(f"deleted {run_dir}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
