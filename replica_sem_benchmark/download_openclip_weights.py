#!/usr/bin/env python3
"""
Download / warm OpenCLIP weights into the same cache layout as the benchmark:

* ``HF_HOME`` / ``HF_HUB_CACHE`` → Hub-hosted weights (Laion, etc.)
* ``OPENCLIP_CACHE`` legacy tree (optional)

Default root (override with ``ROOT=`` env): ``/mnt/data/model-ckpts``

Usage::

    ROOT=/mnt/data/model-ckpts python replica_sem_benchmark/download_openclip_weights.py
    python replica_sem_benchmark/download_openclip_weights.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from clip_model_catalog import CLIP_DOWNLOAD_PAIRS


def _setup_cache_env(root: Path) -> None:
    """Match ``eval_clip_sam_systematic`` / ``download_model_ckpts.sh`` layout."""
    open_clip = root / "clip" / "open_clip"
    # Same as ``DEFAULT_MODEL_CKPTS_ROOT/clip/huggingface_hub`` in eval (Hub root).
    hf_home = root / "clip" / "huggingface_hub"
    open_clip.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    (hf_home / "hub").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OPENCLIP_CACHE", str(open_clip.resolve()))
    os.environ.setdefault("HF_HOME", str(hf_home.resolve()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Print pairs only.")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("ROOT", "/mnt/data/model-ckpts")),
        help="Checkpoint root (default: ROOT env or /mnt/data/model-ckpts).",
    )
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    _setup_cache_env(root)

    pairs = CLIP_DOWNLOAD_PAIRS
    print(f"ROOT={root}")
    print(f"OPENCLIP_CACHE={os.environ['OPENCLIP_CACHE']}")
    print(f"HF_HOME={os.environ['HF_HOME']}")
    print(f"Pairs: {len(pairs)}")
    if args.dry_run:
        for m, p in pairs:
            print(f"  {m} / {p}")
        return

    import open_clip

    device = "cpu"
    ok, bad = 0, 0
    for m, p in pairs:
        try:
            open_clip.create_model_and_transforms(m, pretrained=p, device=device)
            print("OK", m, p)
            ok += 1
        except Exception as e:
            print("SKIP", m, p, "->", e)
            bad += 1
    print(f"Done: {ok} ok, {bad} skipped")


if __name__ == "__main__":
    main()
