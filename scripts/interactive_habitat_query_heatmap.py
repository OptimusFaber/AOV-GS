#!/usr/bin/env python3
"""
Interactive Habitat navigation with live Gaussian query heatmap.

This script:
  1) loads a Habitat scene config (for simulator + camera setup),
  2) loads Gaussian checkpoint (params*.npz) + language field (lang_field.pt),
  3) encodes a text query with CLIP,
  4) lets you navigate with keyboard while rendering query heatmap each frame.

Default display mode is overlay on scene RGB. You can toggle heatmap-only mode.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

# OpenCV Qt backend in some conda builds misses bundled fonts.
# Point Qt to a system font dir to suppress QFontDatabase warnings.
if "QT_QPA_FONTDIR" not in os.environ:
    _sys_font_dir = "/usr/share/fonts/truetype/dejavu"
    if os.path.isdir(_sys_font_dir):
        os.environ["QT_QPA_FONTDIR"] = _sys_font_dir

import cv2
import mmengine
import numpy as np
import quaternion
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.slam.langsplatam.langsplatam import LangSplatam  # noqa: E402
from src.simulator.habitat_utils import make_configuration  # noqa: E402


def _load_ae(ae_ckpt: Path, device: torch.device, encoder_dims=None, decoder_dims=None):
    from src.semantic.language_autoencoder import Autoencoder

    if encoder_dims is None and decoder_dims is None:
        ae = Autoencoder().to(device)
    else:
        if encoder_dims is None or decoder_dims is None:
            raise ValueError("Provide both --encoder_dims and --decoder_dims, or neither.")
        ae = Autoencoder(list(encoder_dims), list(decoder_dims)).to(device)
    ae.load_state_dict(torch.load(str(ae_ckpt), map_location=device))
    ae.eval()
    return ae


def _encode_query_clip(text: str, clip_model: str, clip_pretrained: str, device: torch.device) -> torch.Tensor:
    import open_clip

    clip, _, _ = open_clip.create_model_and_transforms(
        clip_model, pretrained=clip_pretrained, device=device
    )
    clip.eval()
    tok = open_clip.get_tokenizer(clip_model)
    with torch.no_grad():
        emb = clip.encode_text(tok([text]).to(device))[0]
        return F.normalize(emb, dim=-1)


def _normalize_heatmap(sim_np: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    lo = float(np.percentile(sim_np, p_low))
    hi = float(np.percentile(sim_np, p_high))
    if hi <= lo + 1e-8:
        return np.zeros_like(sim_np, dtype=np.float32)
    return np.clip((sim_np - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _render_relevancy_jet(
    model: LangSplatam,
    ae,
    clip_query_512: torch.Tensor,
    w2c_rel: torch.Tensor,
    H: int,
    W: int,
    *,
    blur_sigma: float,
    p_low: float,
    p_high: float,
) -> np.ndarray:
    with torch.no_grad():
        rendered = model.render_lang(w2c_rel, H, W)  # [D,H,W]
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
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)


def _render_rgb(model: LangSplatam, w2c_rel: torch.Tensor, H: int, W: int) -> np.ndarray:
    sys.path.insert(0, str(REPO_ROOT / "third_parties" / "splatam"))
    from utils.slam_helpers import transformed_params2rendervar
    from diff_gaussian_rasterization import GaussianRasterizer as Renderer

    cam = model._setup_camera(H, W, w2c_rel)
    tr = model._transform_gaussians(w2c_rel)
    rv = transformed_params2rendervar(model.params, tr)
    with torch.no_grad():
        rgb, _, _ = Renderer(raster_settings=cam)(**rv)
    img = rgb.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _infer_latent_dim(lang_field_pt: Path) -> int:
    """Infer latent_dim from lang_field.pt checkpoint."""
    ckpt = torch.load(str(lang_field_pt), map_location="cpu")
    if isinstance(ckpt, dict) and "latent_dim" in ckpt:
        return int(ckpt["latent_dim"])
    lf = ckpt.get("lang_feats") if isinstance(ckpt, dict) else None
    if lf is None:
        raise ValueError(f"{lang_field_pt} must contain 'latent_dim' or 'lang_feats'.")
    return int(lf.shape[1])


def _pose_from_state(state) -> np.ndarray:
    """Habitat AgentState -> c2w matrix in RUB."""
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = quaternion.as_rotation_matrix(state.rotation)
    c2w[:3, 3] = np.asarray(state.position, dtype=np.float64)
    return c2w


def _rub_to_rdf_c2w(c2w_rub: np.ndarray) -> np.ndarray:
    """Convert camera-to-world RUB -> RDF (same rule as main activesgm loop)."""
    out = c2w_rub.copy()
    out[:3, 1] *= -1.0
    out[:3, 2] *= -1.0
    return out


def _state_from_pose(sim, c2w_rub: np.ndarray):
    st = sim.agents[0].get_state()
    st.position = c2w_rub[:3, 3]
    st.rotation = quaternion.from_rotation_matrix(c2w_rub[:3, :3])
    sim.agents[0].set_state(st)


def _apply_local_motion(c2w_rub: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """Move in camera-local coordinates (RUB/OpenCV)."""
    out = c2w_rub.copy()
    R = out[:3, :3]
    t = out[:3, 3]
    t = t + R @ np.array([dx, dy, dz], dtype=np.float64)
    out[:3, 3] = t
    return out


def _apply_yaw_pitch(c2w_rub: np.ndarray, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Rotate camera orientation in local camera frame."""
    out = c2w_rub.copy()
    yaw = np.deg2rad(float(yaw_deg))
    pitch = np.deg2rad(float(pitch_deg))

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    # Local-camera rotations: yaw around camera Y (up), pitch around camera X (right).
    R_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    R_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    dR = R_yaw @ R_pitch
    out[:3, :3] = out[:3, :3] @ dR
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--habitat_cfg", required=True, help="Path to habitat.py config")
    p.add_argument("--checkpoint", required=True, help="Path to params*.npz")
    p.add_argument("--lang_field", required=True, help="Path to lang_field.pt")
    p.add_argument("--ae_ckpt", required=True, help="Path to AE checkpoint (.pth)")
    p.add_argument("--text", required=True, help='Text query, e.g. "a chair"')
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--clip_model", default="ViT-B-16")
    p.add_argument("--clip_pretrained", default="laion2b_s34b_b88k")
    p.add_argument("--encoder_dims", nargs="+", type=int, default=None)
    p.add_argument("--decoder_dims", nargs="+", type=int, default=None)
    p.add_argument("--heatmap_blur", type=float, default=3.0)
    p.add_argument("--heatmap_p_low", type=float, default=8.0)
    p.add_argument("--heatmap_p_high", type=float, default=98.0)
    p.add_argument("--overlay_alpha", type=float, default=0.45, help="Only used in overlay mode")
    p.add_argument("--move_step", type=float, default=0.08, help="Translation step in meters")
    p.add_argument("--turn_deg", type=float, default=4.0, help="Yaw step in degrees")
    p.add_argument("--pitch_deg", type=float, default=3.0, help="Pitch step in degrees")
    p.add_argument("--window", default="Habitat Query Heatmap")
    return p.parse_args()


def _print_controls(move_step: float, yaw_deg: float, pitch_deg: float) -> None:
    print("Controls:")
    print("  move: W/S forward/back, A/D strafe, SPACE/C (or Q/E) up/down")
    print("  rotate: arrows (left/right yaw, up/down pitch), or J/L/I/K")
    print("  speed: +/- move step, [/] yaw step, ;/' pitch step")
    print("  display: O toggle overlay<->heatmap")
    print("  misc: R reset pose, H print controls, ESC quit")
    print(
        f"  current: move_step={move_step:.3f}m, yaw_step={yaw_deg:.2f}deg, "
        f"pitch_step={pitch_deg:.2f}deg"
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # ---- Habitat init ----
    hcfg = mmengine.Config.fromfile(args.habitat_cfg)
    sim_cfg = make_configuration(hcfg)
    import habitat_sim

    sim = habitat_sim.Simulator(sim_cfg)
    agent_state = habitat_sim.agent.AgentState()
    agent_state.position = np.asarray(hcfg.agent.position, dtype=np.float64)
    agent_state.rotation = quaternion.from_rotation_vector(np.asarray(hcfg.agent.rotation, dtype=np.float64))
    sim.initialize_agent(0, agent_state)

    # ---- Gaussian + language field ----
    lang_field_path = Path(args.lang_field).expanduser()
    latent_dim = _infer_latent_dim(lang_field_path)
    model = LangSplatam(
        checkpoint_path=str(Path(args.checkpoint).expanduser()),
        latent_dim=latent_dim,
        device=str(device),
    )
    model.load_lang_field(lang_field_path)

    H = int(model.params["org_height"])
    W = int(model.params["org_width"])

    ae = _load_ae(
        Path(args.ae_ckpt).expanduser(),
        device,
        encoder_dims=args.encoder_dims,
        decoder_dims=args.decoder_dims,
    )
    clip_query = _encode_query_clip(
        args.text, args.clip_model, args.clip_pretrained, device
    )

    # Initial pose in simulator (RUB) and SLAM frame anchor (RDF)
    c2w_rub = _pose_from_state(sim.agents[0].get_state())
    c2w_rdf_init = _rub_to_rdf_c2w(c2w_rub)

    show_overlay = True  # default: overlay on rendered scene
    move_step = float(args.move_step)
    yaw_step = float(args.turn_deg)
    pitch_step = float(args.pitch_deg)
    _print_controls(move_step, yaw_step, pitch_step)

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window, W, H)

    while True:
        # Keep simulator agent synchronized with current interactive pose.
        _state_from_pose(sim, c2w_rub)

        # Convert to SLAM-relative w2c for GS rendering.
        c2w_rdf = _rub_to_rdf_c2w(c2w_rub)
        c2w_rel = np.linalg.inv(c2w_rdf_init) @ c2w_rdf
        w2c_rel = torch.tensor(np.linalg.inv(c2w_rel), dtype=torch.float32, device=device)

        jet = _render_relevancy_jet(
            model,
            ae,
            clip_query,
            w2c_rel,
            H,
            W,
            blur_sigma=float(args.heatmap_blur),
            p_low=float(args.heatmap_p_low),
            p_high=float(args.heatmap_p_high),
        )

        if show_overlay:
            rgb = _render_rgb(model, w2c_rel, H, W)
            vis = cv2.addWeighted(
                rgb,
                1.0 - float(args.overlay_alpha),
                jet,
                float(args.overlay_alpha),
                0.0,
            )
        else:
            vis = jet

        mode = "overlay" if show_overlay else "heatmap"
        txt = f'{args.text} | mode={mode} | step={move_step:.2f}m'
        cv2.putText(vis, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(args.window, vis)

        key = cv2.waitKeyEx(1)
        if key == 27:  # ESC
            break
        key8 = key & 0xFF
        # Arrow keycodes in common OpenCV builds: left=81 right=83 up=82 down=84.
        if key8 in (ord("w"), ord("W")):
            c2w_rub = _apply_local_motion(c2w_rub, 0.0, 0.0, -move_step)
        elif key8 in (ord("s"), ord("S")):
            c2w_rub = _apply_local_motion(c2w_rub, 0.0, 0.0, move_step)
        elif key8 in (ord("a"), ord("A")):
            c2w_rub = _apply_local_motion(c2w_rub, -move_step, 0.0, 0.0)
        elif key8 in (ord("d"), ord("D")):
            c2w_rub = _apply_local_motion(c2w_rub, move_step, 0.0, 0.0)
        elif key8 in (ord(" "), ord("q"), ord("Q")):
            c2w_rub = _apply_local_motion(c2w_rub, 0.0, move_step, 0.0)
        elif key8 in (ord("c"), ord("C"), ord("e"), ord("E")):
            c2w_rub = _apply_local_motion(c2w_rub, 0.0, -move_step, 0.0)
        elif key8 in (81, ord("j"), ord("J")):
            c2w_rub = _apply_yaw_pitch(c2w_rub, yaw_deg=yaw_step, pitch_deg=0.0)
        elif key8 in (83, ord("l"), ord("L")):
            c2w_rub = _apply_yaw_pitch(c2w_rub, yaw_deg=-yaw_step, pitch_deg=0.0)
        elif key8 in (82, ord("i"), ord("I")):
            c2w_rub = _apply_yaw_pitch(c2w_rub, yaw_deg=0.0, pitch_deg=pitch_step)
        elif key8 in (84, ord("k"), ord("K")):
            c2w_rub = _apply_yaw_pitch(c2w_rub, yaw_deg=0.0, pitch_deg=-pitch_step)
        elif key8 in (ord("o"), ord("O")):
            show_overlay = not show_overlay
        elif key8 in (ord("r"), ord("R")):
            c2w_rub = _pose_from_state(sim.agents[0].initial_state)
        elif key8 in (ord("+"), ord("=")):
            move_step = min(2.0, move_step * 1.15)
            print(f"move_step={move_step:.3f}m")
        elif key8 in (ord("-"), ord("_")):
            move_step = max(0.005, move_step / 1.15)
            print(f"move_step={move_step:.3f}m")
        elif key8 == ord("["):
            yaw_step = max(0.2, yaw_step / 1.15)
            print(f"yaw_step={yaw_step:.2f}deg")
        elif key8 == ord("]"):
            yaw_step = min(45.0, yaw_step * 1.15)
            print(f"yaw_step={yaw_step:.2f}deg")
        elif key8 == ord(";"):
            pitch_step = max(0.2, pitch_step / 1.15)
            print(f"pitch_step={pitch_step:.2f}deg")
        elif key8 == ord("'"):
            pitch_step = min(30.0, pitch_step * 1.15)
            print(f"pitch_step={pitch_step:.2f}deg")
        elif key8 in (ord("h"), ord("H")):
            _print_controls(move_step, yaw_step, pitch_step)

    cv2.destroyAllWindows()
    sim.close()


if __name__ == "__main__":
    main()

