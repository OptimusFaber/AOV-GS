"""
LangSplatam – offline language-field fine-tuning on top of a frozen 3D
Gaussian Splatting scene produced by ActiveSGM.

The class:
  1. Loads an existing SplaTAM checkpoint (``params*.npz``).
  2. Adds a learnable ``lang_feats`` [N, latent_dim] attribute per Gaussian.
  3. Renders language features with ``diff_gaussian_rasterization`` (3 ch / pass).
     For ``latent_dim > 3`` we run ``ceil(latent_dim / 3)`` passes and concatenate
     — no optional CUDA extension required.
  4. Optimises ``lang_feats`` against per-frame CLIP-encoded target maps.

Rendering (any latent_dim)
----------------------------
The bundled RGB rasteriser outputs 3 channels.  For D-dimensional language
features we slice ``lang_feats[:, c:c+3]`` (padding the last chunk to 3 with
zeros), render each chunk, and stack the first ``n`` channels of every pass.
Gradients flow correctly into the corresponding columns of ``lang_feats``.

Language loss
-------------
L = λ_l1 · L1(rendered, target) + λ_cos · (1 − cos(rendered, target))
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy renderer import helpers
# ---------------------------------------------------------------------------

def _get_rgb_renderer():
    """Return the standard diff_gaussian_rasterization Renderer."""
    from diff_gaussian_rasterization import GaussianRasterizer as Renderer
    return Renderer


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _build_rendervar_lang(params: Dict, transformed: Dict, lang_feats: torch.Tensor) -> Dict:
    """Construct a render-var dict where colors = lang_feats."""
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']
    return {
        'means3D': transformed['means3D'],
        'colors_precomp': lang_feats,
        'rotations': F.normalize(transformed['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0,
    }


def _load_checkpoint(npz_path: str) -> Dict[str, torch.Tensor]:
    """Load ``params*.npz`` → dict of CUDA float tensors."""
    raw = dict(np.load(npz_path, allow_pickle=True))
    params = {}
    skip_keys = {'gt_w2c_all_frames', 'keyframe_time_indices',
                 'seman_cls_ids', 'org_width', 'org_height',
                 'intrinsics', 'w2c'}
    for k, v in raw.items():
        if k in skip_keys:
            params[k] = v  # keep as numpy
        else:
            try:
                params[k] = torch.tensor(v).cuda().float()
            except Exception:
                params[k] = v
    return params


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LangSplatam:
    """
    Offline language-field trainer built on top of a frozen SplaTAM scene.

    Typical usage
    -------------
    model = LangSplatam(
        checkpoint_path = "results/room0/splatam/final/params0.npz",
        intrinsics_path = None,          # auto-read from checkpoint
        latent_dim      = 3,
        device          = "cuda:0",
    )
    model.train_language_field(
        frames_dir            = Path("results/room0/keyframes"),
        language_features_dir = Path("results/room0/language_features"),
        poses_file            = Path("results/room0/keyframe_poses.json"),
        autoencoder_ckpt      = Path("results/room0/autoencoder/ae_p.pt"),
        level                 = "p",    # SAM level to use
        num_iters             = 2000,
        lr                    = 1e-3,
        output_dir            = Path("results/room0/lang_field"),
    )
    """

    def __init__(
        self,
        checkpoint_path: str,
        latent_dim: int = 3,
        device: str = "cuda:0",
        render_checkpoint: str = "auto",
    ) -> None:
        self.device = torch.device(device)
        self.latent_dim = latent_dim
        self.checkpoint_path = checkpoint_path
        # ``auto``: checkpoint only for latent_dim==64 (saves VRAM on multi-pass).
        # ``on``: always checkpoint each 3-ch pass (low VRAM, slower).
        # ``off``: one autograd graph for all passes concatenated (``склейка``; fast, high VRAM).
        if render_checkpoint not in ("auto", "on", "off"):
            raise ValueError("render_checkpoint must be 'auto', 'on', or 'off'")
        self._render_checkpoint_mode = render_checkpoint

        # Load frozen GS params
        self.params = _load_checkpoint(checkpoint_path)
        self._freeze_base_params()

        # Camera intrinsics (read from checkpoint)
        intr = self.params.get('intrinsics', None)
        if intr is not None and isinstance(intr, np.ndarray):
            self.intrinsics = torch.from_numpy(intr).float().to(self.device)
        else:
            self.intrinsics = None

        # First frame w2c (read from checkpoint)
        w2c = self.params.get('w2c', None)
        if w2c is not None and isinstance(w2c, np.ndarray):
            self.first_frame_w2c = torch.from_numpy(w2c).float().to(self.device)
        else:
            self.first_frame_w2c = torch.eye(4, device=self.device)

        # Initialise language features.
        # Unit-norm init is critical: targets from AE.encode() live on the unit sphere.
        # Near-zero init (randn*0.01) makes rendered features ≈0, causing cosine_similarity(≈0, target)
        # to be numerically unstable and the cosine loss to stay pinned at ~1.0 throughout training.
        N = self.params['means3D'].shape[0]
        self.lang_feats = nn.Parameter(
            F.normalize(torch.randn(N, latent_dim, device=self.device), dim=-1)
        )
        logger.info(
            "LangSplatam: %d Gaussians loaded, latent_dim=%d, render_checkpoint=%s",
            N, latent_dim, self._render_checkpoint_mode,
        )

    def _use_render_checkpoint(self) -> bool:
        """Whether to wrap each 3-channel raster pass in gradient checkpointing."""
        D = self.latent_dim
        if D <= 3:
            return False
        mode = self._render_checkpoint_mode
        if mode == "on":
            return True
        if mode == "off":
            return False
        # auto: enable for 64-D only (historical default that avoids OOM on ~3M Gaussians)
        return D == 64

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _freeze_base_params(self) -> None:
        """Detach all base GS params so gradients do not flow into them."""
        for k, v in self.params.items():
            if isinstance(v, torch.Tensor):
                self.params[k] = v.detach()

    def _setup_camera(self, H: int, W: int, w2c: torch.Tensor):
        """Build a raster_settings camera object (standard 3-channel renderer)."""
        sys.path.insert(0, "third_parties/splatam")
        from utils.recon_helpers import setup_camera as _setup_camera

        assert self.intrinsics is not None, "Intrinsics not found in checkpoint."
        intr_np = self.intrinsics.cpu().numpy()
        w2c_np = w2c.detach().cpu().numpy()
        return _setup_camera(W, H, intr_np, w2c_np)

    def _transform_gaussians(self, w2c_rel: torch.Tensor) -> Dict:
        """
        Transform Gaussian centres from world to camera frame.
        Simplified version of SplaTAM's ``transform_to_frame``.
        """
        sys.path.insert(0, "third_parties/splatam")
        from utils.slam_helpers import transform_to_frame

        # Build a minimal params dict with camera pose
        from utils.slam_helpers import matrix_to_quaternion
        rot = w2c_rel[:3, :3].unsqueeze(0)
        quat = matrix_to_quaternion(rot)
        tran = w2c_rel[:3, 3]

        params_tmp = dict(self.params)
        params_tmp['cam_unnorm_rots'] = quat.unsqueeze(-1)
        params_tmp['cam_trans'] = tran.unsqueeze(-1)

        transformed = transform_to_frame(params_tmp, 0,
                                         gaussians_grad=False,
                                         camera_grad=False)
        return transformed

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_lang_chunk_from_feats(
        self,
        lang_feats: torch.Tensor,
        c0: int,
        transformed: Dict,
        renderer,
    ) -> torch.Tensor:
        """One 3-ch rasterizer pass; first ``n`` channels are the latent slice."""
        D = self.latent_dim
        c1 = min(c0 + 3, D)
        n = c1 - c0
        chunk = lang_feats[:, c0:c1]
        if n < 3:
            pad = torch.zeros(
                chunk.shape[0], 3 - n,
                device=chunk.device, dtype=chunk.dtype,
            )
            chunk = torch.cat([chunk, pad], dim=1)
        rendervar = _build_rendervar_lang(self.params, transformed, chunk)
        out, _, _ = renderer(**rendervar)  # [3, H, W]
        return out[:n]

    def render_lang(
        self,
        w2c_rel: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Render language feature map at the given camera pose.

        Parameters
        ----------
        w2c_rel : [4, 4]  world-to-camera transform (relative to first frame)
        H, W    : image resolution

        Returns
        -------
        rendered : [latent_dim, H, W] rendered language features
        """
        cam = self._setup_camera(H, W, w2c_rel)
        transformed = self._transform_gaussians(w2c_rel)
        Renderer = _get_rgb_renderer()
        renderer = Renderer(raster_settings=cam)

        D = self.latent_dim
        if D == 3:
            rendervar = _build_rendervar_lang(self.params, transformed, self.lang_feats)
            rendered, _, _ = renderer(**rendervar)
            return rendered

        # Multi-pass RGB: ``diff_gaussian_rasterization`` is 3-channel only; optional
        # ``channel_rasterization`` often lacks a built ``_C`` extension on user machines.
        #
        # Without checkpoint (``склейка`` графа): memory grows ~linearly with P = ceil(D/3)
        # raster passes.  With checkpoint: ~one pass worth of activations (see --render_checkpoint).
        chunks: List[torch.Tensor] = []
        use_ckpt = self._use_render_checkpoint()
        lf = self.lang_feats
        for c0 in range(0, D, 3):
            if use_ckpt:
                out = checkpoint(
                    lambda lf_in, c0_=c0: self._render_lang_chunk_from_feats(
                        lf_in, c0_, transformed, renderer
                    ),
                    lf,
                    use_reentrant=False,
                )
            else:
                out = self._render_lang_chunk_from_feats(
                    lf, c0, transformed, renderer
                )
            chunks.append(out)

        return torch.cat(chunks, dim=0)  # [D, H, W]

    # ------------------------------------------------------------------
    # Language loss
    # ------------------------------------------------------------------

    @staticmethod
    def language_loss(
        rendered: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lambda_l1: float = 1.0,
        lambda_cos: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute language field loss between rendered and target latent maps.

        Rendered features are L2-normalised per-pixel before computing losses,
        because targets are unit-norm vectors produced by Autoencoder.encode().
        Without this normalisation, alpha-composited rendered features have
        magnitude << 1, creating a magnitude-vs-direction conflict between the
        L1 and cosine terms that prevents the cosine loss from converging.

        Parameters
        ----------
        rendered : [D, H, W]
        target   : [D, H, W]  unit-norm per pixel (from AE encoder)
        mask     : [1, H, W] optional binary mask (1 = valid pixel)

        Returns
        -------
        total, l1_loss, cos_loss : scalar tensors
        """
        # Normalise rendered per-pixel along the feature dimension so that both
        # losses operate on the unit sphere (same space as the targets).
        # eps prevents division by zero for background pixels.
        rendered_n = F.normalize(rendered, p=2, dim=0)  # [D, H, W]

        r_flat = rendered_n.permute(1, 2, 0).reshape(-1, rendered_n.shape[0])  # [HW, D]
        t_flat = target.permute(1, 2, 0).reshape(-1, target.shape[0])          # [HW, D]

        if mask is not None:
            m_flat = mask.permute(1, 2, 0).reshape(-1).bool()  # [HW]
            r_sel = r_flat[m_flat]
            t_sel = t_flat[m_flat]
        else:
            r_sel = r_flat
            t_sel = t_flat

        # Guard: empty mask (no valid pixels in this frame) → return zero loss
        # to avoid NaN from mean() on empty tensors.
        # Must go through the graph (rendered.sum()*0) so .backward() works.
        if r_sel.shape[0] == 0:
            zero = rendered.sum() * 0.0
            return zero, zero.detach(), zero.detach()

        l1 = F.l1_loss(r_sel, t_sel)
        cos = F.cosine_similarity(r_sel, t_sel, dim=-1)
        cos_loss = (1.0 - cos).mean()
        total = lambda_l1 * l1 + lambda_cos * cos_loss
        return total, l1, cos_loss

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train_language_field(
        self,
        language_features_dir: Path,
        poses_file: Path,
        level: str = "s",
        num_iters: int = 30000,
        lr: float = 1e-3,
        lambda_l1: float = 1.0,
        lambda_cos: float = 1.0,
        output_dir: Optional[Path] = None,
        log_every: int = 500,
    ) -> None:
        """
        Fine-tune ``lang_feats`` против сжатых CLIP-карт из language_features_dim3/.

        Parameters
        ----------
        language_features_dir : папка language_features_dim3/ (после train_language_autoencoder.py)
                                 содержит {frame_id:06d}_s.npy (4,H,W) и _f.npy (N,3)
        poses_file   : JSON {str(frame_id): [[4×4 w2c matrix]]}
        level        : 'default'|'s'|'m'|'l'  — уровень SAM (как в LangSplat --feature_level)
        num_iters    : число шагов оптимизации (в LangSplat 30000)
        output_dir   : куда сохранить lang_field.pt
        """
        # ----- Load poses -----
        with open(str(poses_file), 'r') as f:
            poses_raw = json.load(f)
        # poses_raw: {str(frame_id): [[4×4 w2c]]}
        poses: Dict[int, torch.Tensor] = {}
        for fid, mat in poses_raw.items():
            poses[int(fid)] = torch.tensor(mat, dtype=torch.float32, device=self.device)

        from src.semantic.sam_clip_extractor import load_frame_features

        # feature_level: 0=default,1=s,2=m,3=l  (LangSplat convention)
        _level_to_int = {'default': 0, 's': 1, 'm': 2, 'l': 3}
        feature_level = _level_to_int.get(level, 1)

        # ----- Collect valid frame IDs -----
        # LangSplat format: {frame_id:06d}_s.npy + {frame_id:06d}_f.npy
        # они находятся в language_features_dim3/ (после автоэнкодера)
        valid_ids: List[int] = []
        for fid in poses:
            s_path = language_features_dir / f"{fid:06d}_s.npy"
            f_path = language_features_dir / f"{fid:06d}_f.npy"
            if s_path.exists() and f_path.exists():
                valid_ids.append(fid)
        if not valid_ids:
            raise FileNotFoundError(
                f"No language features found in {language_features_dir}. "
                f"Run train_language_autoencoder.py first (creates language_features_dim3/)."
            )
        logger.info("Training language field: %d frames, %d iters.", len(valid_ids), num_iters)

        # ----- Get image resolution from first frame -----
        seg0 = np.load(str(language_features_dir / f"{valid_ids[0]:06d}_s.npy"))  # (4, H, W)
        H, W = seg0.shape[1], seg0.shape[2]
        logger.info("Target resolution: H=%d  W=%d", H, W)

        # ----- Optimiser -----
        # Keep LR constant for 80% of training so the optimizer can keep making
        # meaningful progress; apply a short cosine-anneal only in the final 20%
        # to stabilise convergence.  Both CosineAnnealingLR(T_max=num_iters) and
        # ExponentialLR(γ→0.1) kill the LR too early (by 50% of training the LR
        # is already 3-6× smaller than the start, causing premature plateau).
        import math
        optimizer = torch.optim.Adam([self.lang_feats], lr=lr)
        warmup_end   = max(1, num_iters // 20)          # first 5 %  – warmup
        constant_end = max(warmup_end + 1,
                          int(num_iters * 0.80))         # 5 %–80 %   – constant LR
        def _lr_lambda(it: int) -> float:
            if it <= warmup_end:                          # linear warm-up
                return it / warmup_end
            if it <= constant_end:                        # constant
                return 1.0
            # cosine decay from 1.0 → 0.1 over the last 20 %
            progress = (it - constant_end) / max(1, num_iters - constant_end)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

        # ----- Loss log file (every ``log_every`` iters) -----
        loss_log_path: Optional[Path] = None
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            loss_log_path = output_dir / f"loss_{self.latent_dim}{level}.txt"
            with open(loss_log_path, "w", encoding="utf-8") as _lf:
                _lf.write(
                    "iter\ttotal\tl1\tcos\tlr\tlambda_l1\tlambda_cos\n"
                )
            logger.info("Loss log → %s", loss_log_path)

        # ----- Training loop -----
        import random
        losses: List[float] = []
        l1_losses: List[float] = []
        cos_losses: List[float] = []

        # Best-checkpoint tracking.
        # The per-iteration loss has high variance (one random frame per step),
        # so the last iterate is often not the best one.  We track the smoothed
        # loss over the last `log_every` iterations and save whenever it improves.
        best_avg_loss: float = float("inf")
        best_lang_feats: Optional[torch.Tensor] = None

        for it in range(1, num_iters + 1):
            fid = random.choice(valid_ids)
            w2c_rel = poses[fid]  # [4,4]

            # Загружаем пиксельную карту сжатых фич (D=3) — аналог get_language_feature()
            # language_features_dim3/ содержит _f.npy (N, 3) после автоэнкодера
            feat_np, valid_np = load_frame_features(language_features_dir, fid, feature_level)
            # feat_np: (3, H, W) float32, valid_np: (1, H, W) bool
            target = torch.from_numpy(feat_np).to(self.device)        # [3, H, W]
            valid_mask = torch.from_numpy(valid_np.astype(np.float32)).to(self.device)  # [1,H,W]

            # Render
            rendered = self.render_lang(w2c_rel, H, W)  # [D,H,W]

            # Loss (returns total, l1, cos separately for diagnostics)
            loss, l1_val, cos_val = self.language_loss(
                rendered, target, mask=valid_mask,
                lambda_l1=lambda_l1, lambda_cos=lambda_cos,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Clip gradients: sparse updates (1.8M Gaussians, random frames)
            # can produce large spikes that destabilise training.
            torch.nn.utils.clip_grad_norm_([self.lang_feats], max_norm=1.0)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            l1_losses.append(l1_val.item())
            cos_losses.append(cos_val.item())

            if it % log_every == 0:
                avg     = sum(losses[-log_every:])     / log_every
                avg_l1  = sum(l1_losses[-log_every:])  / log_every
                avg_cos = sum(cos_losses[-log_every:])  / log_every
                cur_lr  = optimizer.param_groups[0]['lr']

                best_marker = ""
                if avg < best_avg_loss:
                    best_avg_loss = avg
                    best_lang_feats = self.lang_feats.detach().cpu().clone()
                    best_marker = "  ← best"

                msg = (f"Iter {it:5d}/{num_iters}  "
                       f"loss={avg:.4f}  l1={avg_l1:.4f}  cos={avg_cos:.4f}  "
                       f"lr={cur_lr:.2e}{best_marker}")
                print(msg, flush=True)
                logger.info("%s", msg)

                if loss_log_path is not None:
                    with open(loss_log_path, "a", encoding="utf-8") as _lf:
                        _lf.write(
                            f"{it}\t{avg:.6f}\t{avg_l1:.6f}\t{avg_cos:.6f}\t"
                            f"{cur_lr:.8e}\t{lambda_l1}\t{lambda_cos}\n"
                        )

        # Restore the best weights found during training before saving.
        # The final iterate is often not the best due to high per-step variance
        # (only one random frame is used per gradient update).
        if best_lang_feats is not None:
            with torch.no_grad():
                self.lang_feats.copy_(best_lang_feats.to(self.device))
            logger.info("Restored best lang_feats (avg loss=%.4f)", best_avg_loss)

        # ----- Save -----
        if output_dir is not None:
            output_dir = Path(output_dir)
            save_path = output_dir / "lang_field.pt"
            torch.save(
                {
                    "lang_feats": self.lang_feats.detach().cpu(),
                    "latent_dim": self.latent_dim,
                    "checkpoint_path": self.checkpoint_path,
                    "level": level,
                },
                str(save_path),
            )
            logger.info("Language field saved → %s", save_path)

    # ------------------------------------------------------------------
    # Open-vocabulary query (inference)
    # ------------------------------------------------------------------

    def load_lang_field(self, lang_field_pt: Path) -> None:
        """
        Load ``lang_field.pt`` produced by ``train_language_field.py`` and set
        ``self.lang_feats`` accordingly.
        """
        ckpt = torch.load(str(lang_field_pt), map_location="cpu")
        lf = ckpt.get("lang_feats", None)
        if lf is None:
            raise ValueError(f"{lang_field_pt} does not contain 'lang_feats'.")
        lf = lf.to(self.device)
        if lf.shape != self.lang_feats.shape:
            raise ValueError(
                f"lang_feats shape mismatch: file {tuple(lf.shape)} vs model {tuple(self.lang_feats.shape)}"
            )
        with torch.no_grad():
            self.lang_feats.copy_(lf)
        logger.info("Loaded language field → %s", lang_field_pt)

    @torch.no_grad()
    def query_text(
        self,
        text: str,
        w2c_rel: torch.Tensor,
        H: int,
        W: int,
        ae_ckpt: Path,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "laion2b_s34b_b88k",
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Render a relevancy map for a text query.

        Returns
        -------
        relevancy : [H, W] float tensor in [0, 1]
        """
        dev = torch.device(device or str(self.device))

        import open_clip
        from src.semantic.language_autoencoder import Autoencoder

        # Encode text via CLIP (512d)
        clip, _, _ = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained, device=dev
        )
        clip.eval()
        tokenizer = open_clip.get_tokenizer(clip_model)
        tokens = tokenizer([text]).to(dev)
        text_emb = F.normalize(clip.encode_text(tokens)[0], dim=-1)  # [512]

        # Encode text to 3D latent space using LangSplat-style AE checkpoint
        # (best_ckpt.pth is a state_dict for Autoencoder)
        ae = Autoencoder().to(dev)
        state = torch.load(str(ae_ckpt), map_location=dev)
        ae.load_state_dict(state)
        ae.eval()
        text_latent = ae.encode(text_emb.unsqueeze(0))[0]  # [latent_dim==3]

        # Render language features
        rendered = self.render_lang(w2c_rel, H, W)  # [D, H, W]

        # Compute cosine similarity
        r_flat = rendered.permute(1, 2, 0).reshape(-1, self.latent_dim)  # [HW, D]
        text_latent_exp = text_latent.unsqueeze(0).expand_as(r_flat)
        sim = F.cosine_similarity(r_flat, text_latent_exp, dim=-1)  # [HW]
        relevancy = sim.reshape(H, W)
        relevancy = (relevancy - relevancy.min()) / (relevancy.max() - relevancy.min() + 1e-8)
        return relevancy
