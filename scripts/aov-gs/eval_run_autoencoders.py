#!/usr/bin/env python3
"""
Evaluate language autoencoder checkpoints for a given RESULT_DIR.

This is a small wrapper around `scripts/eval_language_autoencoder.py` that:
- uses RESULT_DIR/language_features as the original 512d features
- optionally uses RESULT_DIR/language_features_dim{D} for pixel-IoU (if present)
- finds AE checkpoints under AOV-GS/ckpt/<scene_name>/*.pth
- reconstructs encoder_dims/decoder_dims from the checkpoint's state_dict shapes
- runs evaluation with user-provided text queries and writes per-ckpt metrics files

Example:
  python scripts/aov-gs/eval_run_autoencoders.py \
    --result_dir results/Replica/office0/ActiveOpenSem/run_0 \
    --scene_name office0 \
    --queries "a sofa" "a table" "a window" "the chair" \
    --feature_level 1 \
    --top_pct 10 \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Allow importing from repo root when executed from anywhere
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_language_autoencoder import eval_ae  # noqa: E402


def _infer_mlp_dims_from_state_dict(state: Dict[str, torch.Tensor], prefix: str) -> List[int]:
    """
    Infer hidden dims list for Autoencoder by reading Linear weights in order.

    Autoencoder stores layers as ModuleList with interleaved BN/ReLU.
    We extract Linear layers by looking for 2D weight tensors under:
      - f"{prefix}.<idx>.weight"
    and sort by <idx>.
    """
    pat = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.weight$")
    items: List[Tuple[int, torch.Tensor]] = []
    for k, v in state.items():
        m = pat.match(k)
        if not m:
            continue
        if not isinstance(v, torch.Tensor) or v.ndim != 2:
            continue  # skip BatchNorm1d (1D) etc.
        items.append((int(m.group(1)), v))
    items.sort(key=lambda t: t[0])
    if not items:
        raise ValueError(f"Could not infer dims: no Linear weights found under '{prefix}.*.weight'")
    # Each Linear weight is [out_features, in_features]
    return [int(w.shape[0]) for _, w in items]


def _load_state_dict(path: Path, device: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=torch.device(device))
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        return obj  # state_dict
    raise ValueError(f"Unexpected checkpoint format at {path} (expected a raw state_dict dict)")


def _default_scene_name_from_result_dir(result_dir: Path) -> Optional[str]:
    # results/Replica/<scene>/...
    parts = result_dir.as_posix().split("/")
    if "Replica" in parts:
        i = parts.index("Replica")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate AE checkpoints for a RESULT_DIR.")
    p.add_argument(
        "--result_dir",
        required=True,
        help="Path like results/Replica/office0/ActiveOpenSem/run_0",
    )
    p.add_argument(
        "--scene_name",
        default=None,
        help="Name of ckpt subdir under ./ckpt/ (default: infer from RESULT_DIR, e.g. office0)",
    )
    p.add_argument(
        "--queries",
        nargs="+",
        default=["a sofa", "a table", "a window", "the chair"],
        help="Text queries to evaluate",
    )
    p.add_argument("--feature_level", type=int, default=1, help="0=default, 1=s, 2=m, 3=l")
    p.add_argument("--top_pct", type=float, default=10.0)
    p.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for AE evaluation (default cuda:0).",
    )
    p.add_argument(
        "--ckpt_glob",
        default=None,
        help='Optional glob relative to repo root, e.g. "ckpt/office0/*ckpt.pth". '
        "If omitted, uses ckpt/<scene_name>/*.pth",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Where to write metrics (default: RESULT_DIR/ae_eval)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = (REPO_ROOT / args.result_dir).resolve() if not os.path.isabs(args.result_dir) else Path(args.result_dir).resolve()
    if not result_dir.exists():
        raise FileNotFoundError(f"RESULT_DIR not found: {result_dir}")

    scene_name = args.scene_name or _default_scene_name_from_result_dir(result_dir) or "office0"

    features_orig = result_dir / "language_features"
    if not features_orig.is_dir():
        raise FileNotFoundError(f"Missing original features dir: {features_orig}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (result_dir / "ae_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_glob = args.ckpt_glob or f"ckpt/{scene_name}/**/*.pth"
    ckpt_paths = sorted((REPO_ROOT).glob(ckpt_glob))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found with glob: {REPO_ROOT / ckpt_glob}")

    print("==============================================")
    print("AE eval wrapper")
    print(f"RESULT_DIR   : {result_dir}")
    print(f"scene_name   : {scene_name}")
    print(f"features_orig: {features_orig}")
    print(f"ckpt_glob    : {ckpt_glob}")
    print(f"out_dir      : {out_dir}")
    print(f"queries      : {args.queries}")
    print("==============================================")

    failures = 0
    for ckpt in ckpt_paths:
        ckpt = ckpt.resolve()
        try:
            state = _load_state_dict(ckpt, device=args.device)
            encoder_dims = _infer_mlp_dims_from_state_dict(state, "encoder")
            decoder_dims = _infer_mlp_dims_from_state_dict(state, "decoder")
            latent_dim = int(encoder_dims[-1])

            features_enc = result_dir / f"language_features_dim{latent_dim}"
            features_enc_arg = str(features_enc) if features_enc.is_dir() else None

            metrics_path = out_dir / f"metrics_{scene_name}_{latent_dim}d_{ckpt.stem}.txt"

            ns = argparse.Namespace(
                features_orig=str(features_orig),
                features_enc=features_enc_arg,
                ae_ckpt=str(ckpt),
                queries=args.queries,
                feature_level=args.feature_level,
                top_pct=args.top_pct,
                encoder_dims=encoder_dims,
                decoder_dims=decoder_dims,
                clip_model="ViT-B-16",
                clip_pretrained="laion2b_s34b_b88k",
                device=args.device,
                out_metrics=str(metrics_path),
            )

            print("")
            print(f"[ckpt] {ckpt}")
            print(f"  inferred latent_dim : {latent_dim}")
            print(f"  encoder_dims        : {encoder_dims}")
            print(f"  decoder_dims        : {decoder_dims}")
            print(f"  features_enc        : {features_enc_arg if features_enc_arg else '(none; IoU skipped)'}")
            print(f"  out_metrics         : {metrics_path}")

            eval_ae(ns)
        except Exception as e:
            failures += 1
            print("")
            print(f"[FAIL] {ckpt}: {e}")

    if failures:
        raise SystemExit(f"Done with {failures} failures (see logs above).")
    print("\nAll checkpoints evaluated successfully.")


if __name__ == "__main__":
    main()

