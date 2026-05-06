#!/usr/bin/env python3
"""
Simulate a trajectory and overlay a text-query heatmap on rendered RGB.

Inputs:
  - --checkpoint : params*.npz (SplaTAM/ActiveSGM checkpoint with Gaussians)
  - --lang_field : lang_field.pt (per-Gaussian latent features)
  - --text       : CLIP text query
  - --ae_ckpt    : autoencoder checkpoint (state_dict) used to decode latents -> 512d

Trajectory source:
  - Uses checkpoint field ``gt_w2c_all_frames`` if present (saved by SLAM loop).
    Otherwise, pass --poses with keyframe_poses.json.

Outputs:
  - <out_dir>/frames/frame_000000.png ... (BGR images)
  - <out_dir>/video.mp4 (optional, if --write_video)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT))

from src.slam.langsplatam.langsplatam import LangSplatam  # noqa: E402


def _load_ae(ae_ckpt: Path, device: torch.device, encoder_dims=None, decoder_dims=None):
+    from src.semantic.language_autoencoder import Autoencoder
+
+    if encoder_dims is None and decoder_dims is None:
+        ae = Autoencoder().to(device)
+    else:
+        if encoder_dims is None or decoder_dims is None:
+            raise ValueError("Provide both --encoder_dims and --decoder_dims, or neither.")
+        ae = Autoencoder(list(encoder_dims), list(decoder_dims)).to(device)
+    ae.load_state_dict(torch.load(str(ae_ckpt), map_location=device))
+    ae.eval()
+    return ae


def _encode_query_clip(text: str, clip_model: str, clip_pretrained: str, device: torch.device) -> torch.Tensor:
    """Unit-norm CLIP 512d text embedding."""
    import open_clip

    clip, _, _ = open_clip.create_model_and_transforms(
        clip_model, pretrained=clip_pretrained, device=device
    )
    clip.eval()
    tok = open_clip.get_tokenizer(clip_model)
    with torch.no_grad():
        emb = clip.encode_text(tok([text]).to(device))[0]
        return F.normalize(emb, dim=-1)


def _render_rgb(model: LangSplatam, w2c: torch.Tensor, H: int, W: int) -> np.ndarray:
    """Render RGB as uint8 BGR image."""
    sys.path.insert(0, str(REPO_ROOT / "third_parties" / "splatam"))
    from utils.slam_helpers import transformed_params2rendervar
    from diff_gaussian_rasterization import GaussianRasterizer as Renderer

    cam = model._setup_camera(H, W, w2c)
    tr = model._transform_gaussians(w2c)
    rv = transformed_params2rendervar(model.params, tr)
    with torch.no_grad():
        rgb, _, _ = Renderer(raster_settings=cam)(**rv)
    img = rgb.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _normalize_heatmap(sim_np: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    lo = float(np.percentile(sim_np, p_low))
    hi = float(np.percentile(sim_np, p_high))
    if hi <= lo + 1e-8:
        return np.zeros_like(sim_np, dtype=np.float32)
    return np.clip((sim_np - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _render_relevancy_map(
    model: LangSplatam,
    ae,
    clip_query_512: torch.Tensor,  # [512] unit-norm
    w2c: torch.Tensor,
    H: int,
    W: int,
    *,
    blur_sigma: float,
    p_low: float,
    p_high: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render latent field [D,H,W], decode -> 512, compute cosine to CLIP query.
    Returns (jet_bgr_uint8, sim_raw_float32).
    """
    with torch.no_grad():
        rendered = model.render_lang(w2c, H, W)  # [D,H,W]
    D = int(rendered.shape[0])
    r_flat = rendered.permute(1, 2, 0).reshape(-1, D)

    batch = 4096
    sim_flat = torch.empty((r_flat.shape[0],), device=r_flat.device, dtype=torch.float32)
    q = clip_query_512.to(r_flat.device).to(torch.float32)
    with torch.no_grad():
        for i in range(0, r_flat.shape[0], batch):
            c = ae.decode(r_flat[i : i + batch])
            c = F.normalize(c, p=2, dim=-1).to(torch.float32)
            sim_flat[i : i + c.shape[0]] = (c @ q)

    sim = sim_flat.reshape(H, W).cpu().numpy().astype(np.float32)
    if blur_sigma > 0.0:
        ksize = int(blur_sigma * 6) | 1
        sim = cv2.GaussianBlur(sim, (ksize, ksize), blur_sigma)

    norm = _normalize_heatmap(sim, p_low=p_low, p_high=p_high)
    jet = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return jet, sim


def _load_poses_json(path: Path, device: torch.device) -> Dict[int, torch.Tensor]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): torch.tensor(v, dtype=torch.float32, device=device) for k, v in raw.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to params*.npz")
    p.add_argument("--lang_field", required=True, help="Path to lang_field.pt")
    p.add_argument("--text", required=True, help="Text query, e.g. 'a chair'")
    p.add_argument("--ae_ckpt", required=True, help="AE checkpoint (state_dict .pth)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--stride", type=int, default=10, help="Take every Nth pose from trajectory")
    p.add_argument("--max_frames", type=int, default=600, help="Max frames to render")
    p.add_argument("--write_video", action="store_true", help="Write out_dir/video.mp4")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--heatmap_alpha", type=float, default=0.45, help="Overlay alpha for jet heatmap")
    p.add_argument("--heatmap_blur", type=float, default=3.0, help="Gaussian blur sigma on similarity map")
    p.add_argument("--heatmap_p_low", type=float, default=8.0)
    p.add_argument("--heatmap_p_high", type=float, default=98.0)
    p.add_argument("--clip_model", default="ViT-B-16")
    p.add_argument("--clip_pretrained", default="laion2b_s34b_b88k")
    p.add_argument("--poses", default=None, help="Optional keyframe_poses.json (if no gt_w2c_all_frames in npz)")
    p.add_argument("--encoder_dims", nargs="+", type=int, default=None)
    p.add_argument("--decoder_dims", nargs="+", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    out_dir = Path(args.out_dir).expanduser().resolve()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Load scene + language field
    model = LangSplatam(checkpoint_path=str(Path(args.checkpoint).expanduser()), latent_dim=3, device=str(device))
    model.load_lang_field(Path(args.lang_field).expanduser())

    H = int(model.params["org_height"])
    W = int(model.params["org_width"])
    latent_dim = int(model.lang_feats.shape[1])
    print(f"Gaussians: {model.lang_feats.shape[0]}  latent_dim={latent_dim}  Res={H}x{W}")

    # AE + CLIP query
    ae = _load_ae(
        Path(args.ae_ckpt).expanduser(),
        device,
        encoder_dims=args.encoder_dims,
        decoder_dims=args.decoder_dims,
    )
    clip_q = _encode_query_clip(args.text, args.clip_model, args.clip_pretrained, device)

    # Poses
    poses_seq = None
    if "gt_w2c_all_frames" in model.params and isinstance(model.params["gt_w2c_all_frames"], np.ndarray):
        poses_seq = model.params["gt_w2c_all_frames"]
        # poses_seq is [T,4,4] numpy, already w2c in SLAM code
        print(f"Trajectory: gt_w2c_all_frames (T={poses_seq.shape[0]}) from checkpoint")
        pose_ids = list(range(poses_seq.shape[0]))
        pose_dict: Optional[Dict[int, torch.Tensor]] = None
    else:
        if not args.poses:
            raise SystemExit(
                "Checkpoint has no 'gt_w2c_all_frames'. Provide --poses keyframe_poses.json."
            )
        pose_dict = _load_poses_json(Path(args.poses).expanduser(), device)
        pose_ids = sorted(pose_dict.keys())
        print(f"Trajectory: {len(pose_ids)} poses from {Path(args.poses).expanduser().resolve()}")

    # Video writer
    vw = None
    if args.write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out_dir / "video.mp4"), fourcc, int(args.fps), (W, H), True)
        if not vw.isOpened():
            raise RuntimeError("Could not open video writer. Try a different codec/container.")

    n_written = 0
    for k, pid in enumerate(pose_ids[:: max(1, int(args.stride))]):
        if n_written >= int(args.max_frames):
            break
        if poses_seq is not None:
            w2c = torch.tensor(poses_seq[pid], dtype=torch.float32, device=device)
        else:
            assert pose_dict is not None
            w2c = pose_dict[int(pid)]

        rgb = _render_rgb(model, w2c, H, W)
        jet, _sim = _render_relevancy_map(
            model,
            ae,
            clip_q,
            w2c,
            H,
            W,
            blur_sigma=float(args.heatmap_blur),
            p_low=float(args.heatmap_p_low),
            p_high=float(args.heatmap_p_high),
        )
        overlay = cv2.addWeighted(rgb, 1.0 - float(args.heatmap_alpha), jet, float(args.heatmap_alpha), 0.0)

        label = f'{args.text} | frame={int(pid)}'
        cv2.putText(
            overlay,
            label,
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        out_path = frames_dir / f"frame_{n_written:06d}.png"
        cv2.imwrite(str(out_path), overlay)
        if vw is not None:
            vw.write(overlay)

        if n_written % 20 == 0:
            print(f"[{n_written:04d}] pose_id={int(pid)} -> {out_path.name}")
        n_written += 1

    if vw is not None:
        vw.release()
        print(f"Video -> {(out_dir / 'video.mp4').resolve()}")
    print(f"Frames -> {frames_dir.resolve()}  (count={n_written})")


if __name__ == "__main__":
    main()

