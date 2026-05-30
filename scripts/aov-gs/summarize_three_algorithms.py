#!/usr/bin/env python3
"""Summary tables: ActiveSGM (ActiveSem) vs ActiveGeom vs ActiveOpenSem."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2] / "results" / "Replica"
SCENES = ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]

ALGORITHMS = {
    "ActiveSGM": "ActiveSem",
    "ActiveGeom": "ActiveGeom",
    "ActiveOpenSem": "ActiveOpenSem",
}


def parse_render(p: Path) -> dict[str, float] | None:
    if not p.exists():
        return None
    d: dict[str, float] = {}
    for line in p.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = float(v.strip())
    return d


def get_step(p: Path, stage: int) -> int | None:
    if not p.exists():
        return None
    key = f"exploration_stage_{stage}_step"
    for line in p.read_text().splitlines():
        if key in line:
            return int(line.split(":")[1].strip())
    return None


def get_step_refinement(run_dir: Path) -> int | None:
    """Final planner step (after refinement). Requires exploration_path_poses.json."""
    p = run_dir / "exploration_path_poses.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        n = int(data["num_poses"])
        return n - 1 if n > 0 else None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def find_best_run(scene: str, folder: str) -> Path | None:
    base = ROOT / scene / folder
    if not base.exists():
        return None
    runs = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda x: int(x.name.split("_")[1]),
    )
    for run in reversed(runs):
        if (run / "splatam/eval_final/render_result.txt").exists():
            return run
    return runs[-1] if runs else None


def wall_time_hours(run_dir: Path) -> float | None:
    metrics = run_dir / "ablation_metrics.json"
    if metrics.exists():
        try:
            data = json.loads(metrics.read_text(encoding="utf-8"))
            total_s = data.get("total_wall_s")
            if total_s is None and isinstance(data.get("stages"), dict):
                slam = data["stages"].get("slam", {})
                total_s = slam.get("wall_s")
            if total_s is not None:
                return float(total_s) / 3600.0
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    final = run_dir / "splatam/final/params.npz"
    if not final.exists():
        final = run_dir / "splatam/final/params0.npz"
    if not final.exists():
        return None
    mtimes = [p.stat().st_mtime for p in (run_dir / "splatam").rglob("*") if p.is_file()]
    if not mtimes:
        return None
    return (final.stat().st_mtime - min(mtimes)) / 3600.0


def parse_miou(result_dir: Path) -> float | None:
    p = result_dir / "lang_field_traj_eval/miou_summary.txt"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if line.startswith("Overall mIoU"):
            return float(line.split(":")[1].strip()) * 100
    return None


def fmt_stage(m: dict[str, float] | None) -> str:
    if not m:
        return "—"
    return (
        f"PSNR {m['psnr']:.2f} | SSIM {m['ssim']:.3f} | "
        f"LPIPS {m['lpips']:.3f} | L1 {m['l1(cm)']:.2f} cm"
    )


def collect_row(scene: str, run: Path | None) -> dict:
    if run is None:
        return {
            "scene": scene,
            "status": "missing",
            "time_spent_h": "",
            "run": "",
            "step_stage0": "",
            "step_stage1": "",
            "step_refinement": "",
            "exploration_stage_I": "—",
            "exploration_stage_II": "—",
            "refinement_stage": "—",
            "PSNR": "",
            "SSIM": "",
            "LPIPS": "",
            "L1_cm": "",
            "lang_field_miou_pct": "",
        }

    s0 = parse_render(run / "splatam/eval_exploration_stage_0/render_result.txt")
    s1 = parse_render(run / "splatam/eval_exploration_stage_1/render_result.txt")
    fin = parse_render(run / "splatam/eval_final/render_result.txt")
    st0 = get_step(run / "splatam/eval_exploration_stage_0/exploration_info.txt", 0)
    st1 = get_step(run / "splatam/eval_exploration_stage_1/exploration_info.txt", 1)
    st_ref = get_step_refinement(run)
    t = wall_time_hours(run)
    miou = parse_miou(run)

    status = "complete" if fin else "incomplete"
    row = {
        "scene": scene,
        "status": status,
        "time_spent_h": f"{t:.2f}" if t is not None else "—",
        "run": run.name,
        "step_stage0": st0 if st0 is not None else "",
        "step_stage1": st1 if st1 is not None else "",
        "step_refinement": st_ref if st_ref is not None else "",
        "exploration_stage_I": fmt_stage(s0),
        "exploration_stage_II": fmt_stage(s1),
        "refinement_stage": fmt_stage(fin),
        "PSNR": f"{fin['psnr']:.2f}" if fin else "",
        "SSIM": f"{fin['ssim']:.3f}" if fin else "",
        "LPIPS": f"{fin['lpips']:.3f}" if fin else "",
        "L1_cm": f"{fin['l1(cm)']:.2f}" if fin else "",
        "lang_field_miou_pct": f"{miou:.1f}" if miou is not None else "",
    }
    return row


def mean_row(rows: list[dict]) -> dict:
    complete = [r for r in rows if r["status"] == "complete" and r["PSNR"]]
    if not complete:
        return {"scene": "MEAN", "status": "n/a"}

    def avg(key: str) -> str:
        vals = [float(r[key]) for r in complete if r.get(key)]
        return f"{mean(vals):.3f}" if key == "SSIM" or key == "LPIPS" else f"{mean(vals):.2f}"

    times = [float(r["time_spent_h"]) for r in complete if r["time_spent_h"] not in ("", "—")]
    miou_vals = [float(r["lang_field_miou_pct"]) for r in complete if r["lang_field_miou_pct"]]
    ref_steps = [int(r["step_refinement"]) for r in complete if r.get("step_refinement") not in ("", None)]

    return {
        "scene": "MEAN",
        "status": f"{len(complete)}/{len(rows)} complete",
        "time_spent_h": f"{mean(times):.2f}" if times else "—",
        "run": "",
        "step_stage0": "",
        "step_stage1": "",
        "step_refinement": f"{mean(ref_steps):.0f}" if ref_steps else "",
        "exploration_stage_I": "",
        "exploration_stage_II": "",
        "refinement_stage": "",
        "PSNR": avg("PSNR"),
        "SSIM": avg("SSIM"),
        "LPIPS": avg("LPIPS"),
        "L1_cm": avg("L1_cm"),
        "lang_field_miou_pct": f"{mean(miou_vals):.1f}" if miou_vals else "",
    }


CSV_FIELDS = [
    "scene",
    "status",
    "time_spent_h",
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


def write_txt(path: Path, algo: str, rows: list[dict]) -> None:
    lines = [
        "=" * 100,
        f"Replica NVS summary — {algo}",
        "=" * 100,
        "",
        "NVS metrics from splatam eval on replica_sim_nvs trajectory.",
        "  exploration stage I  → eval_exploration_stage_0",
        "  exploration stage II → eval_exploration_stage_1",
        "  refinement stage     → eval_final (refinement + post-refinement)",
        "  PSNR/SSIM/LPIPS/L1   → eval_final (numeric columns)",
        "  step_stage0/1        → end of exploration stage I / II",
        "  step_refinement      → final planner step (exploration_path_poses num_poses − 1; empty if file missing)",
        "  time_spent_h         → wall time (first splatam artifact → final checkpoint)",
        "  lang_field_miou_pct  → lang_field_traj_eval pair mIoU (Geom/Hybrid, if run)",
        "",
    ]

    hdr = (
        f"{'Scene':<8} {'Time(h)':>7} {'St0':>5} {'St1':>5} {'StR':>5} {'Status':<10} "
        f"{'PSNR':>6} {'SSIM':>6} {'LPIPS':>6} {'L1':>6} {'mIoU%':>6}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rows:
        if r["scene"] == "MEAN":
            lines.append("-" * len(hdr))
        miou = r.get("lang_field_miou_pct") or "—"
        lines.append(
            f"{r['scene']:<8} {str(r['time_spent_h']):>7} "
            f"{str(r['step_stage0']):>5} {str(r['step_stage1']):>5} "
            f"{str(r.get('step_refinement', '')):>5} {r['status']:<10} "
            f"{str(r['PSNR']):>6} {str(r['SSIM']):>6} {str(r['LPIPS']):>6} "
            f"{str(r['L1_cm']):>6} {str(miou):>6}"
        )

    lines.append("")
    lines.append("--- Per-stage metrics (full) ---")
    for r in rows:
        if r["scene"] == "MEAN":
            continue
        lines.append(f"\n[{r['scene']}] run={r['run']} status={r['status']}")
        lines.append(f"  Stage I  : {r['exploration_stage_I']}")
        lines.append(f"  Stage II : {r['exploration_stage_II']}")
        lines.append(f"  Refine   : {r['refinement_stage']}")
    lines.append("")
    lines.append("=" * 100)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, list[dict]] = {}

    for algo, folder in ALGORITHMS.items():
        rows = [collect_row(sc, find_best_run(sc, folder)) for sc in SCENES]
        rows.append(mean_row(rows))
        all_rows[algo] = rows

        slug = algo.lower().replace(" ", "_")
        write_csv(ROOT / f"table_{slug}.csv", rows)
        write_txt(ROOT / f"table_{slug}.txt", algo, rows)
        print(f"Wrote table_{slug}.csv / .txt")

    # compact cross-algorithm comparison (eval_final only)
    cmp_path = ROOT / "table_comparison_eval_final.csv"
    with cmp_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scene",
                "ActiveSGM_time_h",
                "ActiveSGM_PSNR",
                "ActiveGeom_time_h",
                "ActiveGeom_PSNR",
                "Hybrid_time_h",
                "Hybrid_PSNR",
                "Hybrid_mIoU_pct",
            ]
        )
        for i, sc in enumerate(SCENES):
            row = [sc]
            for algo in ALGORITHMS:
                r = all_rows[algo][i]
                row.extend([r["time_spent_h"], r["PSNR"]])
            row.append(all_rows["Hybrid"][i]["lang_field_miou_pct"])
            w.writerow(row)
    print(f"Wrote {cmp_path}")


if __name__ == "__main__":
    main()
