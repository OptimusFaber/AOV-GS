#!/usr/bin/env python3
"""
Batch-evaluate trained Gaussian language fields (lang_field_*) on a set of text queries.

For each lang_field directory under RESULT_DIR (or an explicit list), this script:
  - loads lang_field.pt to infer latent_dim and checkpoint_path
  - selects a compatible AE checkpoint from ckpt/<scene_name>/*.pth (matching latent_dim)
  - runs `scripts/query_language_field.py` for each query
  - collects the written *_metrics.json into a single CSV summary

Outputs (default):
  RESULT_DIR/lang_field_query_eval/
    runs/<lang_field_name>/<query_slug>_metrics.json (+ renders from query_language_field)
    summary.csv

Example:
  python scripts/activesgm/eval_run_lang_fields.py \
    --result_dir results/Replica/office0/ActiveOpenVocab/run_0 \
    --scene_name office0 \
    --queries "a sofa" "a table" "a window" "the chair" \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "q"


def _infer_latent_from_ae_ckpt(ae_ckpt: Path, device: str) -> int:
    state = torch.load(str(ae_ckpt), map_location=torch.device(device))
    if not isinstance(state, dict):
        raise ValueError(f"{ae_ckpt} is not a state_dict dict")
    # Autoencoder.encoder is ModuleList of Linear/BN/ReLU/Linear...; last Linear out_features = latent_dim
    # Take the max encoder.*.weight 2D keys by index.
    pat = re.compile(r"^encoder\.(\d+)\.weight$")
    items: List[Tuple[int, torch.Tensor]] = []
    for k, v in state.items():
        m = pat.match(k)
        if m and isinstance(v, torch.Tensor) and v.ndim == 2:
            items.append((int(m.group(1)), v))
    if not items:
        raise ValueError(f"Could not infer latent_dim from {ae_ckpt} (no encoder.*.weight 2D)")
    items.sort(key=lambda t: t[0])
    return int(items[-1][1].shape[0])


def _pick_ae_ckpt(scene_dir: Path, latent_dim: int, device: str) -> Optional[Path]:
    """
    Pick an AE checkpoint under ckpt/<scene_name>/ that matches latent_dim.
    Preference: best_ckpt.pth if it matches; otherwise first matching *.pth by name order.
    """
    if not scene_dir.is_dir():
        return None
    # New layout: ckpt/<scene>/<latent_dim>/*.pth (but keep compatibility).
    preferred = scene_dir / str(latent_dim) / "best_ckpt.pth"
    if preferred.exists():
        return preferred
    candidates = sorted(scene_dir.rglob("*.pth"))
    if not candidates:
        return None
    # Prefer best_ckpt.pth if compatible
    best = scene_dir / "best_ckpt.pth"
    if best.exists():
        try:
            if _infer_latent_from_ae_ckpt(best, device) == latent_dim:
                return best
        except Exception:
            pass
    for p in candidates:
        try:
            if _infer_latent_from_ae_ckpt(p, device) == latent_dim:
                return p
        except Exception:
            continue
    return None


def _load_lang_field_meta(lang_field_pt: Path) -> Dict:
    ckpt = torch.load(str(lang_field_pt), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"{lang_field_pt} must be a dict checkpoint")
    latent_dim = int(ckpt.get("latent_dim") or ckpt["lang_feats"].shape[1])
    checkpoint_path = ckpt.get("checkpoint_path", None)
    level = ckpt.get("level", None)
    return {"latent_dim": latent_dim, "checkpoint_path": checkpoint_path, "level": level}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-evaluate lang_field_* dirs with fixed queries.")
    p.add_argument("--result_dir", required=True)
    p.add_argument("--scene_name", required=True)
    p.add_argument("--queries", nargs="+", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--poses", default=None, help="Optional keyframe_poses.json; omit to use auto-poses.")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--top_percentile", type=float, default=2.0)
    p.add_argument("--dbscan_eps", type=float, default=0.15)
    p.add_argument("--dbscan_min", type=int, default=30)
    p.add_argument("--top_k_views", type=int, default=3)
    p.add_argument(
        "--pose_select",
        choices=("relevancy", "centroid"),
        default="centroid",
        help="Pose selection mode passed to query_language_field.py. "
        "centroid is much cheaper and avoids per-pose full-frame cosine renders.",
    )
    p.add_argument(
        "--lang_fields",
        nargs="*",
        default=None,
        help="Optional explicit list of lang_field directories (relative to RESULT_DIR or absolute). "
        "If omitted, uses RESULT_DIR/lang_field_*/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = (REPO_ROOT / result_dir).resolve()
    if not result_dir.is_dir():
        raise FileNotFoundError(result_dir)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (result_dir / "lang_field_query_eval")
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    ckpt_root = Path(os.environ.get("ACTIVESGM_CKPT_ROOT", str(REPO_ROOT / "ckpt"))).expanduser()
    ckpt_scene_dir = ckpt_root / args.scene_name

    # Determine which lang_field dirs to evaluate
    if args.lang_fields:
        lang_dirs = []
        for x in args.lang_fields:
            p = Path(x)
            if not p.is_absolute():
                p = (result_dir / p).resolve()
            lang_dirs.append(p)
    else:
        lang_dirs = sorted(result_dir.glob("lang_field_*"))

    # Filter to those that have lang_field.pt
    lang_dirs = [d for d in lang_dirs if (d / "lang_field.pt").exists()]
    if not lang_dirs:
        raise SystemExit("No lang_field.pt found (expected RESULT_DIR/lang_field_*/lang_field.pt).")

    rows: List[Dict[str, object]] = []

    for ld in lang_dirs:
        meta = _load_lang_field_meta(ld / "lang_field.pt")
        latent_dim = int(meta["latent_dim"])
        ckpt_path = meta["checkpoint_path"]
        if not ckpt_path:
            # fallback to RESULT_DIR/splatam/final params
            final_dir = result_dir / "splatam" / "final"
            ckpt0 = final_dir / "params0.npz"
            ckpt1 = final_dir / "params.npz"
            ckpt_path = str(ckpt0 if ckpt0.exists() else ckpt1)

        ae_ckpt = _pick_ae_ckpt(ckpt_scene_dir, latent_dim, args.device)
        if ae_ckpt is None:
            # Record as skipped
            rows.append(
                {
                    "lang_field_dir": str(ld),
                    "latent_dim": latent_dim,
                    "level": meta.get("level"),
                    "query": "",
                    "status": "SKIP_no_ae_ckpt_for_dim",
                    "ae_ckpt": "",
                }
            )
            continue

        for q in args.queries:
            qslug = _slug(q)
            run_out = runs_dir / ld.name / qslug
            run_out.parent.mkdir(parents=True, exist_ok=True)

            out_prefix = str(run_out)  # query_language_field will add suffixes and write <name>_metrics.json in same dir

            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "query_language_field.py"),
                "--checkpoint",
                str(ckpt_path),
                "--lang_field",
                str(ld / "lang_field.pt"),
                "--text",
                q,
                "--ae_ckpt",
                str(ae_ckpt),
                "--device",
                args.device,
                "--pose_select",
                args.pose_select,
                "--top_percentile",
                str(args.top_percentile),
                "--dbscan_eps",
                str(args.dbscan_eps),
                "--dbscan_min",
                str(args.dbscan_min),
                "--top_k_views",
                str(args.top_k_views),
                "--out",
                out_prefix,
            ]
            if args.poses:
                cmd += ["--poses", args.poses]

            print(f"\n==> {ld.name} | D={latent_dim} | query={q!r}")
            ec = subprocess.call(cmd)

            metrics_path = (run_out.parent / f"{run_out.name}_metrics.json")
            if ec != 0:
                rows.append(
                    {
                        "lang_field_dir": str(ld),
                        "latent_dim": latent_dim,
                        "level": meta.get("level"),
                        "query": q,
                        "status": f"FAIL_exit_{ec}",
                        "ae_ckpt": str(ae_ckpt),
                        "metrics_json": str(metrics_path) if metrics_path.exists() else "",
                    }
                )
                continue

            if not metrics_path.exists():
                rows.append(
                    {
                        "lang_field_dir": str(ld),
                        "latent_dim": latent_dim,
                        "level": meta.get("level"),
                        "query": q,
                        "status": "OK_no_metrics_json",
                        "ae_ckpt": str(ae_ckpt),
                        "metrics_json": "",
                    }
                )
                continue

            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            top = (m.get("clusters") or [{}])[0]
            rows.append(
                {
                    "lang_field_dir": str(ld),
                    "latent_dim": latent_dim,
                    "level": meta.get("level"),
                    "query": q,
                    "status": "OK",
                    "ae_ckpt": str(ae_ckpt),
                    "clusters_found": m.get("clusters_found"),
                    "top_total_score": top.get("total_score"),
                    "top_size": top.get("size"),
                    "top_frame_id": top.get("frame_id"),
                    "metrics_json": str(metrics_path),
                }
            )

    # Write summary CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nDone. Summary → {csv_path}")


if __name__ == "__main__":
    main()

