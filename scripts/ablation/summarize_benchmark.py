#!/usr/bin/env python3
"""Generate benchmark tables with timing, VRAM, and gaussian counts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_utils import (
    SCENES_DEFAULT,
    collect_row,
    find_best_run,
    mean_row,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "Replica"

ALGORITHMS = {
    "activesgm": ("ActiveSem", "table_activesgm"),
    "activegeom": ("ActiveGeom", "table_activegeom"),
    "hybrid": ("ActiveOpenSem", "table_hybrid"),
}

CSV_FIELDS = [
    "scene",
    "status",
    "time_spent_h",
    "time_stage0_h",
    "time_stage1_h",
    "time_refinement_h",
    "time_lang_train_h",
    "time_validate_h",
    "vram_slam_peak_mb",
    "vram_lang_peak_mb",
    "num_gaussians",
    "run",
    "step_stage0",
    "step_stage1",
    "step_refinement",
    "exploration_stage_I",
    "exploration_stage_II",
    "refinement_stage",
    "PSNR",
    "SSIM",
    "LPIPS",
    "L1_cm",
    "lang_field_miou_pct",
]

NUMERIC_FOR_MEAN = [
    "time_spent_h",
    "time_stage0_h",
    "time_stage1_h",
    "time_refinement_h",
    "vram_slam_peak_mb",
    "num_gaussians",
    "PSNR",
    "SSIM",
    "LPIPS",
    "L1_cm",
    "lang_field_miou_pct",
]


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize Replica benchmark runs.")
    p.add_argument(
        "--algo",
        choices=list(ALGORITHMS.keys()) + ["all"],
        default="all",
        help="Which algorithm table to build",
    )
    p.add_argument("--scenes", nargs="*", default=SCENES_DEFAULT)
    p.add_argument("--out_dir", type=Path, default=OUT)
    args = p.parse_args()

    algos = ALGORITHMS.keys() if args.algo == "all" else [args.algo]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for slug in algos:
        folder, table_name = ALGORITHMS[slug]
        rows = [collect_row(sc, find_best_run(sc, folder, args.out_dir)) for sc in args.scenes]
        mean = mean_row(rows, NUMERIC_FOR_MEAN)
        for k in CSV_FIELDS:
            mean.setdefault(k, "")
        rows.append(mean)

        csv_path = args.out_dir / f"{table_name}.csv"
        write_csv(csv_path, rows, CSV_FIELDS)
        print(f"Wrote {csv_path} ({sum(1 for r in rows if r['status']=='complete')} complete)")


if __name__ == "__main__":
    main()
