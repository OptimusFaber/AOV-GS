#!/usr/bin/env python3
"""
Compare rough OpenCLIP VRAM estimates (from ``clip_model_catalog``) to your GPU.

Uses ``nvidia-smi`` for total VRAM if available, else ``torch.cuda``. Does **not**
load every model — estimates are static heuristics for ``--vram_limit_gb`` in the benchmark.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from clip_model_catalog import CLIP_VRAM_ESTIMATES_GB, clip_configs_for_eval


def _gpu_total_mib() -> tuple[str | None, int | None]:
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        line = out.splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            name = parts[0]
            m = re.search(r"(\d+)", parts[1])
            if m:
                return name, int(m.group(1))
    except Exception:
        pass
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--headroom-gb",
        type=float,
        default=1.5,
        help="Treat (total_GB - headroom) as safe CLIP-only budget (SAM unloaded).",
    )
    ap.add_argument(
        "--vram-limit",
        type=float,
        default=None,
        help="Override budget (GB); default: min(14, total - headroom).",
    )
    args = ap.parse_args()

    name, mib = _gpu_total_mib()
    total_gb = (mib / 1024.0) if mib else None

    try:
        import torch

        if torch.cuda.is_available():
            dev = torch.cuda.get_device_properties(0)
            if total_gb is None:
                total_gb = dev.total_memory / (1024**3)
                name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    if total_gb is None:
        print("No GPU / nvidia-smi — use a machine with CUDA for meaningful numbers.")
        total_gb = 16.0
        name = "(assumed)"

    budget = args.vram_limit if args.vram_limit is not None else min(14.0, total_gb - args.headroom_gb)

    print(f"GPU: {name}")
    print(f"Total VRAM: {total_gb:.1f} GB")
    print(f"Benchmark-style CLIP budget (≈ min(14, total−{args.headroom_gb})): {budget:.1f} GB  "
          f"(override with --vram-limit)")
    print()

    pairs = clip_configs_for_eval()
    rows = []
    for model_name, pre in pairs:
        est = CLIP_VRAM_ESTIMATES_GB.get(model_name, 1.0)
        ok = est <= budget
        label = f"{model_name} / {pre[:24]}"
        rows.append((label, est, "OK" if ok else "SKIP est>limit"))

    w = min(72, max(len(r[0]) for r in rows) if rows else 40)
    for lab, est, st in sorted(rows, key=lambda x: (-x[1], x[0])):
        print(f"  {lab:<{w}}  ~{est:.2f} GB  {st}")

    n_ok = sum(1 for _, est, _ in rows if est <= budget)
    print()
    print(f"Fit estimate (≤ {budget:.1f} GB): {n_ok}/{len(rows)} models")


if __name__ == "__main__":
    main()
