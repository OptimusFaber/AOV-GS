#!/usr/bin/env python3
"""Shared helpers for ablation benchmarks: timing, VRAM, gaussian count."""
from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any

SCENES_DEFAULT = [
    "office0", "office1", "office2", "office3", "office4",
    "room0", "room1", "room2",
]

ROOT = Path(__file__).resolve().parents[2]
RESULTS_REPLICA = ROOT / "results" / "Replica"


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


def find_best_run(scene: str, folder: str, results_root: Path | None = None) -> Path | None:
    base = (results_root or RESULTS_REPLICA) / scene / folder
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


def count_gaussians(run_dir: Path) -> int | None:
    for rel in ("splatam/final/params.npz", "splatam/final/params0.npz"):
        p = run_dir / rel
        if not p.exists():
            continue
        try:
            import numpy as np

            data = np.load(str(p))
            if "means3D" in data:
                return int(data["means3D"].shape[0])
            if "means3D" in data.files:
                return int(data["means3D"].shape[0])
        except Exception:
            pass
    return None


def load_ablation_metrics(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "ablation_metrics.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def vram_peak_from_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    vals: list[float] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[,\s]+", line)
        for part in parts:
            try:
                vals.append(float(part))
            except ValueError:
                continue
    return max(vals) if vals else None


def stage_wall_hours(run_dir: Path) -> dict[str, float | None]:
    """Estimate per-stage wall time from checkpoint mtimes (fallback)."""
    splatam = run_dir / "splatam"
    if not splatam.exists():
        return {"stage0_h": None, "stage1_h": None, "refinement_h": None, "total_h": None}

    def mtime(p: Path) -> float | None:
        return p.stat().st_mtime if p.exists() else None

    all_files = [p for p in splatam.rglob("*") if p.is_file()]
    if not all_files:
        return {"stage0_h": None, "stage1_h": None, "refinement_h": None, "total_h": None}
    t0 = min(p.stat().st_mtime for p in all_files)

    ckpt = {
        "stage0": mtime(splatam / "exploration_stage_0/params.npz"),
        "stage1": mtime(splatam / "exploration_stage_1/params.npz"),
        "final": mtime(splatam / "final/params.npz") or mtime(splatam / "final/params0.npz"),
    }

    def h(a: float | None, b: float | None) -> float | None:
        if a is None or b is None or b <= a:
            return None
        return (b - a) / 3600.0

    total = h(t0, ckpt["final"])
    return {
        "stage0_h": h(t0, ckpt["stage0"]),
        "stage1_h": h(ckpt["stage0"], ckpt["stage1"]),
        "refinement_h": h(ckpt["stage1"], ckpt["final"]),
        "total_h": total,
    }


def merge_timing_metrics(run_dir: Path) -> dict[str, Any]:
    abl = load_ablation_metrics(run_dir)
    stages = abl.get("stages", {})
    fallback = stage_wall_hours(run_dir)

    def pick(key: str, fb_key: str) -> str:
        if key in stages and stages[key].get("wall_s") is not None:
            return f"{float(stages[key]['wall_s']) / 3600.0:.2f}"
        fb = fallback.get(fb_key)
        return f"{fb:.2f}" if fb is not None else "—"

    def vram_pick(key: str) -> str:
        if key in stages and stages[key].get("vram_peak_mb") is not None:
            return f"{float(stages[key]['vram_peak_mb']):.0f}"
        log = run_dir / f"vram_{key}.log"
        peak = vram_peak_from_log(log)
        return f"{peak:.0f}" if peak is not None else "—"

    total_h = abl.get("total_wall_s")
    if total_h is None:
        total_h = fallback.get("total_h")
        total_str = f"{total_h:.2f}" if total_h is not None else "—"
    else:
        total_str = f"{float(total_h) / 3600.0:.2f}"

    ng = abl.get("num_gaussians") or count_gaussians(run_dir)
    return {
        "time_total_h": total_str,
        "time_stage0_h": pick("slam", "stage0_h"),
        "time_stage1_h": pick("slam_stage1", "stage1_h"),
        "time_refinement_h": pick("slam_refinement", "refinement_h"),
        "time_lang_train_h": pick("lang_field", "refinement_h") if "lang_field" in stages else "—",
        "time_validate_h": pick("validate", "refinement_h") if "validate" in stages else "—",
        "vram_slam_peak_mb": vram_pick("slam"),
        "vram_lang_peak_mb": vram_pick("lang_field"),
        "num_gaussians": str(ng) if ng is not None else "—",
    }


def parse_miou(result_dir: Path) -> float | None:
    for sub in ("lang_field_traj_eval/miou_summary.txt", "validate/miou_summary.txt"):
        p = result_dir / sub
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
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


def collect_row(scene: str, run: Path | None, extra_fields: list[str] | None = None) -> dict:
    base = {
        "scene": scene,
        "status": "missing",
        "time_spent_h": "",
        "time_stage0_h": "",
        "time_stage1_h": "",
        "time_refinement_h": "",
        "time_lang_train_h": "",
        "time_validate_h": "",
        "vram_slam_peak_mb": "",
        "vram_lang_peak_mb": "",
        "num_gaussians": "",
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
    if extra_fields:
        for f in extra_fields:
            base.setdefault(f, "")

    if run is None:
        return base

    s0 = parse_render(run / "splatam/eval_exploration_stage_0/render_result.txt")
    s1 = parse_render(run / "splatam/eval_exploration_stage_1/render_result.txt")
    fin = parse_render(run / "splatam/eval_final/render_result.txt")
    timing = merge_timing_metrics(run)
    miou = parse_miou(run)

    row = {
        **base,
        "scene": scene,
        "status": "complete" if fin else "incomplete",
        "time_spent_h": timing["time_total_h"],
        "time_stage0_h": timing["time_stage0_h"],
        "time_stage1_h": timing["time_stage1_h"],
        "time_refinement_h": timing["time_refinement_h"],
        "time_lang_train_h": timing["time_lang_train_h"],
        "time_validate_h": timing["time_validate_h"],
        "vram_slam_peak_mb": timing["vram_slam_peak_mb"],
        "vram_lang_peak_mb": timing["vram_lang_peak_mb"],
        "num_gaussians": timing["num_gaussians"],
        "run": run.name,
        "step_stage0": get_step(run / "splatam/eval_exploration_stage_0/exploration_info.txt", 0) or "",
        "step_stage1": get_step(run / "splatam/eval_exploration_stage_1/exploration_info.txt", 1) or "",
        "step_refinement": get_step_refinement(run) or "",
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


def mean_row(rows: list[dict], numeric_keys: list[str]) -> dict:
    complete = [r for r in rows if r["status"] == "complete" and r.get("PSNR")]
    out: dict[str, Any] = {"scene": "MEAN", "status": f"{len(complete)}/{len(rows)} complete"}
    if not complete:
        return out

    for key in numeric_keys:
        vals = []
        for r in complete:
            v = r.get(key, "")
            if v in ("", "—"):
                continue
            try:
                vals.append(float(v))
            except ValueError:
                continue
        if not vals:
            out[key] = ""
        elif key == "num_gaussians":
            out[key] = f"{mean(vals):.0f}"
        elif key in ("SSIM", "LPIPS"):
            out[key] = f"{mean(vals):.3f}"
        else:
            out[key] = f"{mean(vals):.2f}"
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def available_gpus(min_free_mb: int = 4000) -> list[str]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ["cuda:0"]

    gpus: list[str] = []
    for line in out.strip().splitlines():
        idx, free = [x.strip() for x in line.split(",")]
        if int(free) >= min_free_mb:
            gpus.append(f"cuda:{idx}")
    return gpus or ["cuda:0"]


def patch_hybrid_k(scene: str, k: int, root: Path | None = None) -> Path:
    cfg_path = (root or ROOT) / "configs" / "Replica" / scene / "ActiveOpenSemHybrid.py"
    text = cfg_path.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r"max_semantic_candidates\s*=\s*\d+",
        f"max_semantic_candidates={k}",
        text,
        count=1,
    )
    if n == 0:
        raise RuntimeError(f"max_semantic_candidates not found in {cfg_path}")
    cfg_path.write_text(new_text, encoding="utf-8")
    return cfg_path
