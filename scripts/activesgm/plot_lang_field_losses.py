#!/usr/bin/env python3
"""
Plot and summarize language-field training losses for a RESULT_DIR.

Reads per-run files like:
  RESULT_DIR/lang_field_s4/loss_4s.txt
  RESULT_DIR/lang_field_m64/loss_64m.txt

Each loss file is expected to be tab-separated with a header:
  iter  total  l1  cos  lr  lambda_l1  lambda_cos

Outputs (default):
  RESULT_DIR/lang_field_loss_plots/
    - summary.csv
    - loss_total_grid.png
    - loss_components_grid.png
    - per_run/loss_4s.png (etc.)
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LOSS_RE = re.compile(r"loss_(\d+)([sml])\.txt$")


@dataclass(frozen=True)
class LossSeries:
    D: int
    level: str
    path: Path
    it: List[int]
    total: List[float]
    l1: List[float]
    cos: List[float]
    lr: List[float]


def _read_loss_file(path: Path) -> LossSeries:
    m = LOSS_RE.search(path.name)
    if not m:
        raise ValueError(f"Unrecognized loss filename: {path.name}")
    D = int(m.group(1))
    level = m.group(2)

    it: List[int] = []
    total: List[float] = []
    l1: List[float] = []
    cos: List[float] = []
    lr: List[float] = []

    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        cols = {name: idx for idx, name in enumerate(header)}
        required = ["iter", "total", "l1", "cos", "lr"]
        for r in required:
            if r not in cols:
                raise ValueError(f"{path}: missing column '{r}' (got {header})")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            it.append(int(float(parts[cols["iter"]])))
            total.append(float(parts[cols["total"]]))
            l1.append(float(parts[cols["l1"]]))
            cos.append(float(parts[cols["cos"]]))
            lr.append(float(parts[cols["lr"]]))

    return LossSeries(D=D, level=level, path=path, it=it, total=total, l1=l1, cos=cos, lr=lr)


def _sort_key(s: LossSeries) -> Tuple[int, str]:
    order = {"s": 0, "m": 1, "l": 2}
    return (s.D, order.get(s.level, 99))


def _ensure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401


def _plot_per_run(series: LossSeries, out_png: Path) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    ax.plot(series.it, series.total, label="total", linewidth=2.0)
    ax.plot(series.it, series.l1, label="l1", linewidth=1.5, alpha=0.9)
    ax.plot(series.it, series.cos, label="cos", linewidth=1.5, alpha=0.9)
    ax.set_title(f"loss_{series.D}{series.level}  ({series.path.parent.name})")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def _grid_dims(series_all: List[LossSeries]) -> Tuple[List[int], List[str]]:
    Ds = sorted({s.D for s in series_all})
    levels = ["s", "m", "l"]
    return Ds, levels


def _series_map(series_all: List[LossSeries]) -> Dict[Tuple[int, str], LossSeries]:
    return {(s.D, s.level): s for s in series_all}


def _plot_grid_total(series_all: List[LossSeries], out_png: Path) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    Ds, levels = _grid_dims(series_all)
    smap = _series_map(series_all)

    nrows = len(Ds)
    ncols = len(levels)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.6 * ncols, 2.6 * nrows), dpi=160, squeeze=False)

    for ri, D in enumerate(Ds):
        for ci, L in enumerate(levels):
            ax = axes[ri][ci]
            s = smap.get((D, L))
            if s is None or len(s.it) == 0:
                ax.set_axis_off()
                continue
            ax.plot(s.it, s.total, linewidth=1.8)
            ax.set_title(f"{D}{L} total")
            ax.grid(True, alpha=0.25)
            if ri == nrows - 1:
                ax.set_xlabel("iter")
            if ci == 0:
                ax.set_ylabel("loss")

    fig.suptitle("Language field training loss (total)", y=1.01)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _plot_grid_components(series_all: List[LossSeries], out_png: Path) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    Ds, levels = _grid_dims(series_all)
    smap = _series_map(series_all)

    nrows = len(Ds)
    ncols = len(levels)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.6 * ncols, 2.6 * nrows), dpi=160, squeeze=False)

    for ri, D in enumerate(Ds):
        for ci, L in enumerate(levels):
            ax = axes[ri][ci]
            s = smap.get((D, L))
            if s is None or len(s.it) == 0:
                ax.set_axis_off()
                continue
            ax.plot(s.it, s.l1, label="l1", linewidth=1.5)
            ax.plot(s.it, s.cos, label="cos", linewidth=1.5)
            ax.set_title(f"{D}{L} components")
            ax.grid(True, alpha=0.25)
            if ri == nrows - 1:
                ax.set_xlabel("iter")
            if ci == 0:
                ax.set_ylabel("loss")
            if ri == 0 and ci == ncols - 1:
                ax.legend(loc="best")

    fig.suptitle("Language field training loss components", y=1.01)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _write_summary_csv(series_all: List[LossSeries], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "D",
                "level",
                "loss_file",
                "n_points",
                "iter_last",
                "total_last",
                "total_best",
                "l1_last",
                "cos_last",
                "lr_last",
            ]
        )
        for s in sorted(series_all, key=_sort_key):
            if not s.it:
                continue
            w.writerow(
                [
                    s.D,
                    s.level,
                    str(s.path),
                    len(s.it),
                    s.it[-1],
                    s.total[-1],
                    min(s.total),
                    s.l1[-1],
                    s.cos[-1],
                    s.lr[-1],
                ]
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot language field loss curves for a RESULT_DIR.")
    p.add_argument("--result_dir", required=True, help=".../results/.../run_0")
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: RESULT_DIR/lang_field_loss_plots)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir).expanduser().resolve()
    if not result_dir.is_dir():
        raise FileNotFoundError(result_dir)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (result_dir / "lang_field_loss_plots")
    per_run_dir = out_dir / "per_run"

    loss_paths = sorted(result_dir.glob("lang_field_*/loss_*.txt"))
    if not loss_paths:
        raise SystemExit(f"No loss files found under {result_dir}/lang_field_*/loss_*.txt")

    series_all: List[LossSeries] = []
    for lp in loss_paths:
        try:
            series_all.append(_read_loss_file(lp))
        except Exception as e:
            print(f"[skip] {lp}: {e}")

    if not series_all:
        raise SystemExit("No valid loss series parsed.")

    # Per-run plots
    for s in series_all:
        _plot_per_run(s, per_run_dir / f"loss_{s.D}{s.level}.png")

    # Grid plots
    _plot_grid_total(series_all, out_dir / "loss_total_grid.png")
    _plot_grid_components(series_all, out_dir / "loss_components_grid.png")

    # Summary
    _write_summary_csv(series_all, out_dir / "summary.csv")

    print(f"Done. Wrote plots + summary to: {out_dir}")


if __name__ == "__main__":
    main()

