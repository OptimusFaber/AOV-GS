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

Rendering (geometry, SplaTAM-AOV ``eval`` convention)
------------------------------------------------------
Feature / base-GS rasterisation matches ``src/slam/splatam/eval_helper.eval``:
Gaussians are transformed with ``rel_w2c = gt_w2c(view k)``, while the
rasteriser camera projection uses **checkpoint** ``intrinsics`` and
``first_frame_w2c`` (frame-0 projector), **not** the view ``w2c``.
This differs from naive one-matrix setups (camera and Gaussians share the same
``w2c``), where NVS previews would drift from ``rendered_rgb/gs_*.png``.

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
try:
    from sklearn.cluster import MiniBatchKMeans
except Exception:  # pragma: no cover
    MiniBatchKMeans = None

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
        vq_layer_num: int = 1,
        codebook_size: int = 64,
        topk: int = 4,
    ) -> None:
        self.device = torch.device(device)
        self.latent_dim = latent_dim
        self.checkpoint_path = checkpoint_path
        self.vq_layer_num = int(vq_layer_num)
        self.codebook_size = int(codebook_size)
        self.topk = int(topk)
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

        # Legacy LangSplat-style features (kept for backward compatibility).
        N = self.params['means3D'].shape[0]
        self.lang_feats = nn.Parameter(
            F.normalize(torch.randn(N, latent_dim, device=self.device), dim=-1)
        )
        # LangSplatV2-style learnable logits and codebooks.
        self.language_feature_logits: Optional[nn.Parameter] = None
        self.language_feature_codebooks: Optional[nn.Parameter] = None
        self.model_format = "legacy"
        logger.info(
            "LangSplatam: %d Gaussians loaded, latent_dim=%d, vq_layer_num=%d, codebook_size=%d, topk=%d, render_checkpoint=%s",
            N, latent_dim, self.vq_layer_num, self.codebook_size, self.topk, self._render_checkpoint_mode,
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

    def _ensure_v2_parameters(self) -> None:
        """Create LangSplatV2 parameters if missing."""
        if self.language_feature_logits is not None and self.language_feature_codebooks is not None:
            return
        n_gaussians = self.params['means3D'].shape[0]
        self.language_feature_logits = nn.Parameter(
            torch.zeros(
                n_gaussians,
                self.vq_layer_num * self.codebook_size,
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.language_feature_codebooks = nn.Parameter(
            torch.randn(
                self.vq_layer_num,
                self.codebook_size,
                512,
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.model_format = "langsplatv2"

    def _fit_v2_codebooks(
        self,
        language_features_dir: Path,
        max_init_features: int = 200_000,
        random_state: int = 0,
    ) -> None:
        """
        Initialise codebooks with residual MiniBatchKMeans on raw CLIP features.
        """
        if MiniBatchKMeans is None:
            raise ImportError(
                "scikit-learn is required for LangSplatV2 codebook initialisation. "
                "Install it with `pip install scikit-learn`."
            )
        self._ensure_v2_parameters()
        feature_files = sorted(language_features_dir.glob("*_f.npy"))
        if not feature_files:
            raise FileNotFoundError(f"No *_f.npy files in {language_features_dir}")
        feats = []
        for fp in feature_files:
            arr = np.load(str(fp))
            if arr.size == 0:
                continue
            if arr.ndim != 2 or arr.shape[1] != 512:
                raise ValueError(
                    f"Expected CLIP features with shape [N,512], got {arr.shape} in {fp}"
                )
            feats.append(arr.astype(np.float32))
        if not feats:
            raise ValueError("No non-empty feature files found for codebook initialization.")
        data = np.concatenate(feats, axis=0)
        if data.shape[0] > max_init_features:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(data.shape[0], size=max_init_features, replace=False)
            data = data[idx]

        residuals = data
        init_codebooks = []
        for level in range(self.vq_layer_num):
            km = MiniBatchKMeans(
                n_clusters=self.codebook_size,
                batch_size=4096,
                random_state=random_state + level,
                n_init="auto",
            )
            km.fit(residuals)
            centers = km.cluster_centers_.astype(np.float32)
            init_codebooks.append(centers)
            labels = km.predict(residuals)
            residuals = residuals - centers[labels]
        init_codebooks_t = torch.from_numpy(np.stack(init_codebooks, axis=0)).to(self.device)
        with torch.no_grad():
            self.language_feature_codebooks.copy_(init_codebooks_t)

    def _logits_to_sparse_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert per-Gaussian logits to top-k sparse coefficients for each VQ level.
        """
        per_level = []
        for level in range(self.vq_layer_num):
            start = level * self.codebook_size
            end = (level + 1) * self.codebook_size
            level_logits = logits[:, start:end]
            probs = torch.softmax(level_logits, dim=1)
            k = min(self.topk, self.codebook_size)
            values, indices = torch.topk(probs, k=k, dim=1)
            sparse = torch.zeros_like(probs)
            sparse.scatter_(1, indices, values)
            sparse = sparse / (sparse.sum(dim=1, keepdim=True) + 1e-10)
            per_level.append(sparse)
        return torch.cat(per_level, dim=1)

    def _reconstruct_clip_feature_map(
        self, language_feature_weight_map: torch.Tensor, upto_layer: Optional[int] = None
    ) -> torch.Tensor:
        """
        Reconstruct a 512D feature map from rendered weight maps.
        """
        assert self.language_feature_codebooks is not None, "V2 codebooks are not initialised."
        d, h, w = language_feature_weight_map.shape
        weights = language_feature_weight_map.view(d, -1)
        if upto_layer is None:
            upto_layer = self.vq_layer_num - 1
        layers = []
        for level in range(upto_layer + 1):
            start = level * self.codebook_size
            end = (level + 1) * self.codebook_size
            layer_feat = self.language_feature_codebooks[level].T @ weights[start:end]
            layer_feat = layer_feat.view(512, h, w)
            if level > 0:
                layer_feat = layer_feat + layers[-1].detach()
            layers.append(layer_feat)
        return layers[-1]

    @torch.no_grad()
    def decode_gaussian_clip_v2(self, start: int, end: int) -> torch.Tensor:
        """
        Decode per-Gaussian CLIP vectors from V2 logits + codebooks (no rendering).

        Parameters
        ----------
        start, end : slice indices along Gaussian dimension (half-open ``end``).

        Returns
        -------
        Tensor [end - start, 512] on ``self.device``.
        """
        assert self.language_feature_logits is not None
        assert self.language_feature_codebooks is not None
        logits = self.language_feature_logits[start:end]
        w = self._logits_to_sparse_weights(logits)
        out = torch.zeros((w.shape[0], 512), device=w.device, dtype=w.dtype)
        for level in range(self.vq_layer_num):
            sl = slice(level * self.codebook_size, (level + 1) * self.codebook_size)
            wl = w[:, sl]
            Cl = self.language_feature_codebooks[level]
            out = out + wl @ Cl
        return out

    def _setup_camera(self, H: int, W: int, w2c: torch.Tensor):
        """Build SplaTAM raster_settings (``utils.recon_helpers.setup_camera``)."""
        sys.path.insert(0, "third_parties/splatam")
        from utils.recon_helpers import setup_camera as _setup_camera

        assert self.intrinsics is not None, "Intrinsics not found in checkpoint."
        intr_np = self.intrinsics.cpu().numpy().copy()
        # If training/rendering at a reduced resolution, scale intrinsics accordingly.
        org_w = int(self.params.get("org_width", W))
        org_h = int(self.params.get("org_height", H))
        sx = float(W) / float(max(org_w, 1))
        sy = float(H) / float(max(org_h, 1))
        intr_np[0, 0] *= sx
        intr_np[1, 1] *= sy
        intr_np[0, 2] *= sx
        intr_np[1, 2] *= sy
        w2c_np = w2c.detach().cpu().numpy()
        return _setup_camera(W, H, intr_np, w2c_np)

    def _eval_style_transformed_gaussians(self, gt_w2c_k: torch.Tensor) -> Dict:
        """
        World→camera Gaussian transform consistent with ``eval_helper.eval``.

        Uses ``eval_helper.transform_to_frame(..., rel_w2c=gt_w2c_k)``.
        Raster projection must still be built via ``_setup_camera(H, W, self.first_frame_w2c)``.
        """
        from src.slam.splatam.eval_helper import transform_to_frame as splat_eval_t2f

        m = gt_w2c_k.to(device=self.device, dtype=torch.float32)
        # time_idx unused when ``rel_w2c`` is set; kept for API parity with eval().
        return splat_eval_t2f(self.params, 0, False, False, rel_w2c=m)

    def _transform_gaussians(self, w2c_rel: torch.Tensor) -> Dict:
        """
        Transform Gaussian centres from world to camera frame.
        Simplified version of SplaTAM's ``transform_to_frame``.

        .. note::
           For parity with SplaTAM-AOV NVS metrics (``eval`` / ``gs_*.png``), rendering
           should use ``_eval_style_transformed_gaussians`` + ``first_frame_w2c``
           camera — see ``_render_feature_map``. This helper remains for callers that
           intentionally mimic legacy single-``w2c`` setups.
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

    def _render_chunk_from_feats(
        self,
        feats: torch.Tensor,
        feat_dim: int,
        c0: int,
        transformed: Dict,
        renderer,
    ) -> torch.Tensor:
        """One 3-ch rasterizer pass; first ``n`` channels are the latent slice."""
        D = feat_dim
        c1 = min(c0 + 3, D)
        n = c1 - c0
        chunk = feats[:, c0:c1]
        if n < 3:
            pad = torch.zeros(
                chunk.shape[0], 3 - n,
                device=chunk.device, dtype=chunk.dtype,
            )
            chunk = torch.cat([chunk, pad], dim=1)
        rendervar = _build_rendervar_lang(self.params, transformed, chunk)
        out, _, _ = renderer(**rendervar)  # [3, H, W]
        return out[:n]

    def _render_feature_map(
        self,
        feats: torch.Tensor,
        feat_dim: int,
        w2c_rel: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Render an arbitrary per-Gaussian feature tensor [N, feat_dim].

        ``w2c_rel`` is the **Gaussian** world→camera matrix for view *k*
        (same as ``eval``'s ``gt_w2c``); the raster intrinsics/projector use
        ``first_frame_w2c`` saved in the SplaTAM checkpoint — matching
        ``eval_helper.eval``.
        """
        cam = self._setup_camera(H, W, self.first_frame_w2c)
        transformed = self._eval_style_transformed_gaussians(w2c_rel)
        Renderer = _get_rgb_renderer()
        renderer = Renderer(raster_settings=cam)

        if feat_dim == 3:
            rendervar = _build_rendervar_lang(self.params, transformed, feats)
            rendered, _, _ = renderer(**rendervar)
            return rendered

        chunks: List[torch.Tensor] = []
        use_ckpt = self._use_render_checkpoint()
        for c0 in range(0, feat_dim, 3):
            if use_ckpt:
                out = checkpoint(
                    lambda feats_in, c0_=c0: self._render_chunk_from_feats(
                        feats_in, feat_dim, c0_, transformed, renderer
                    ),
                    feats,
                    use_reentrant=False,
                )
            else:
                out = self._render_chunk_from_feats(
                    feats, feat_dim, c0, transformed, renderer
                )
            chunks.append(out)
        return torch.cat(chunks, dim=0)

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
        w2c_rel : [4, 4]
            Dataset / NVS ``gt_w2c`` used to move Gaussians into camera *k*.
            Raster projection stays at checkpoint ``first_frame_w2c`` (SplaTAM-AOV eval).
        H, W    : image resolution

        Returns
        -------
        rendered : [latent_dim, H, W] rendered language features
        """
        return self._render_feature_map(self.lang_feats, self.latent_dim, w2c_rel, H, W)

    def render_v2_weight_map(
        self,
        w2c_rel: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Render LangSplatV2 sparse coefficients (not 512D CLIP features).
        Returns [L*K, H, W].
        """
        assert self.language_feature_logits is not None, "V2 logits are not initialised."
        sparse_weights = self._logits_to_sparse_weights(self.language_feature_logits)
        return self._render_feature_map(
            sparse_weights,
            self.vq_layer_num * self.codebook_size,
            w2c_rel,
            H,
            W,
        )

    def render_v2_clip_feature_map(
        self,
        w2c_rel: torch.Tensor,
        H: int,
        W: int,
        upto_layer: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Render and decode LangSplatV2 CLIP feature map [512, H, W].
        """
        weight_map = self.render_v2_weight_map(w2c_rel, H, W)
        return self._reconstruct_clip_feature_map(weight_map, upto_layer=upto_layer)

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
        max_strip_pixels: int = 49152,
        pixel_chunk: int = 16384,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute language field loss between rendered and target latent maps.

        Rendered features are L2-normalised per-pixel before computing losses,
        because targets are unit-norm vectors produced by Autoencoder.encode().
        Without this normalisation, alpha-composited rendered features have
        magnitude << 1, creating a magnitude-vs-direction conflict between the
        L1 and cosine terms that prevents the cosine loss from converging.

        Memory: avoids allocating ``[H*W, D]`` (OOM on large frames / V2).  Loss is
        accumulated over horizontal strips and pixel chunks.

        Parameters
        ----------
        rendered : [D, H, W]
        target   : [D, H, W]  unit-norm per pixel (from AE encoder)
        mask     : [1, H, W] optional binary mask (1 = valid pixel)

        Returns
        -------
        total, l1_loss, cos_loss : scalar tensors
        """
        rendered_n = F.normalize(rendered, p=2, dim=0)  # [D, H, W]
        D, H, W = rendered_n.shape[0], rendered_n.shape[1], rendered_n.shape[2]

        strip_h = max(1, min(H, max(1, max_strip_pixels // max(W, 1))))

        l1_sum = rendered.sum() * 0.0
        cos_sum = rendered.sum() * 0.0
        n_pixels = 0

        for y0 in range(0, H, strip_h):
            y1 = min(H, y0 + strip_h)
            rn = rendered_n[:, y0:y1, :]
            tn = target[:, y0:y1, :]

            if mask is not None:
                mk = mask[:, y0:y1, :].squeeze(0).bool()
                if not mk.any():
                    continue
                r_strip = rn[:, mk].transpose(0, 1).contiguous()
                t_strip = tn[:, mk].transpose(0, 1).contiguous()
            else:
                r_strip = rn.permute(1, 2, 0).reshape(-1, D).contiguous()
                t_strip = tn.permute(1, 2, 0).reshape(-1, D).contiguous()

            npix = r_strip.shape[0]
            if npix == 0:
                continue

            for s in range(0, npix, pixel_chunk):
                rr = r_strip[s : s + pixel_chunk]
                tt = t_strip[s : s + pixel_chunk]
                l1_sum = l1_sum + F.l1_loss(rr, tt, reduction="sum")
                cos_sum = cos_sum + F.cosine_similarity(rr, tt, dim=-1).sum()
                n_pixels += rr.shape[0]

        if n_pixels == 0:
            zero = rendered.sum() * 0.0
            return zero, zero.detach(), zero.detach()

        l1 = l1_sum / (n_pixels * D)
        cos_loss = 1.0 - cos_sum / n_pixels
        total = lambda_l1 * l1 + lambda_cos * cos_loss
        return total, l1, cos_loss

    def _serialize_lang_field_v2(
        self,
        level: str,
        logits_cpu: torch.Tensor,
        codebooks_cpu: torch.Tensor,
        *,
        best_avg_loss: Optional[float] = None,
        best_iteration: Optional[int] = None,
    ) -> dict:
        payload: dict = {
            "format": "langsplatv2",
            "language_feature_logits": logits_cpu,
            "language_feature_codebooks": codebooks_cpu,
            "vq_layer_num": self.vq_layer_num,
            "codebook_size": self.codebook_size,
            "topk": self.topk,
            "latent_dim": self.vq_layer_num * self.codebook_size,
            "checkpoint_path": self.checkpoint_path,
            "level": level,
        }
        if best_avg_loss is not None and best_iteration is not None:
            payload["best_avg_loss"] = float(best_avg_loss)
            payload["best_iteration"] = int(best_iteration)
        return payload

    def _serialize_lang_field_legacy(
        self,
        level: str,
        lang_feats_cpu: torch.Tensor,
        *,
        best_avg_loss: Optional[float] = None,
        best_iteration: Optional[int] = None,
    ) -> dict:
        payload: dict = {
            "format": "legacy",
            "lang_feats": lang_feats_cpu,
            "latent_dim": self.latent_dim,
            "checkpoint_path": self.checkpoint_path,
            "level": level,
        }
        if best_avg_loss is not None and best_iteration is not None:
            payload["best_avg_loss"] = float(best_avg_loss)
            payload["best_iteration"] = int(best_iteration)
        return payload

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
        use_langsplat_v2: bool = True,
        max_init_features: int = 200_000,
        train_downscale: float = 1.0,
    ) -> None:
        """Train language field in legacy or LangSplatV2 mode."""
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
        # Raw SAM+CLIP format: {frame_id:06d}_s.npy + {frame_id:06d}_f.npy.
        valid_ids: List[int] = []
        for fid in poses:
            s_path = language_features_dir / f"{fid:06d}_s.npy"
            f_path = language_features_dir / f"{fid:06d}_f.npy"
            if s_path.exists() and f_path.exists():
                valid_ids.append(fid)
        if not valid_ids:
            raise FileNotFoundError(
                f"No language features found in {language_features_dir}. "
                f"Expected raw SAM+CLIP exports with *_s.npy and *_f.npy."
            )
        logger.info(
            "Training language field (%s): %d frames, %d iters.",
            "LangSplatV2" if use_langsplat_v2 else "legacy",
            len(valid_ids),
            num_iters,
        )

        # ----- Get image resolution from first frame -----
        seg0 = np.load(str(language_features_dir / f"{valid_ids[0]:06d}_s.npy"))  # (4, H, W)
        H, W = seg0.shape[1], seg0.shape[2]
        logger.info("Target resolution: H=%d  W=%d", H, W)
        if train_downscale <= 0.0 or train_downscale > 1.0:
            raise ValueError(f"train_downscale must be in (0,1], got {train_downscale}")
        H_train = max(1, int(round(H * train_downscale)))
        W_train = max(1, int(round(W * train_downscale)))
        logger.info(
            "Train render resolution: H=%d W=%d (scale=%.3f)",
            H_train,
            W_train,
            train_downscale,
        )

        # ----- Optimiser -----
        import math
        if use_langsplat_v2:
            self._fit_v2_codebooks(
                language_features_dir=language_features_dir,
                max_init_features=max_init_features,
            )
            assert self.language_feature_logits is not None
            assert self.language_feature_codebooks is not None
            optim_params = [self.language_feature_logits, self.language_feature_codebooks]
            feat_dim = self.vq_layer_num * self.codebook_size
        else:
            optim_params = [self.lang_feats]
            feat_dim = self.latent_dim
            self.model_format = "legacy"
        optimizer = torch.optim.Adam(optim_params, lr=lr)
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
            suffix_dim = feat_dim if use_langsplat_v2 else self.latent_dim
            loss_log_path = output_dir / f"loss_{suffix_dim}{level}.txt"
            with open(loss_log_path, "w", encoding="utf-8") as _lf:
                _lf.write(
                    "iter\ttotal\tl1\tcos\tlr\tlambda_l1\tlambda_cos\n"
                )
            logger.info("Loss log → %s", loss_log_path)
            logger.info(
                "best.pt is written whenever the %d-iter rolling avg improves (same cadence as logging).",
                log_every,
            )

        # ----- Training loop -----
        import random
        losses: List[float] = []
        l1_losses: List[float] = []
        cos_losses: List[float] = []

        # Best-checkpoint tracking.
        # The per-iteration loss has high variance (one random frame per step),
        # so the last iterate is often not the best one.  We track the smoothed
        # loss over the last `log_every` iterations; on improvement we refresh
        # ``best_lang_feats`` and write ``best.pt`` under ``output_dir`` (same schema
        # as ``lang_field.pt`` plus ``best_avg_loss`` / ``best_iteration``).
        best_avg_loss: float = float("inf")
        best_lang_feats = None

        for it in range(1, num_iters + 1):
            fid = random.choice(valid_ids)
            w2c_rel = poses[fid]  # [4,4]

            # Load raw CLIP target map [512,H,W].
            feat_np, valid_np = load_frame_features(language_features_dir, fid, feature_level)
            target = torch.from_numpy(feat_np).to(self.device)        # [D, H, W]
            valid_mask = torch.from_numpy(valid_np.astype(np.float32)).to(self.device)  # [1,H,W]
            if H_train != H or W_train != W:
                target = F.interpolate(
                    target.unsqueeze(0),
                    size=(H_train, W_train),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                valid_mask = F.interpolate(
                    valid_mask.unsqueeze(0),
                    size=(H_train, W_train),
                    mode="nearest",
                ).squeeze(0)
            if use_langsplat_v2 and target.shape[0] != 512:
                raise ValueError(
                    "LangSplatV2 mode expects raw CLIP features with dimension 512. "
                    f"Got {target.shape[0]}. Use language_features/, not language_features_dim*."
                )

            if use_langsplat_v2:
                assert self.language_feature_codebooks is not None
                layer_idx = min(int(it / 10000 * self.vq_layer_num), self.vq_layer_num - 1)
                rendered = self.render_v2_weight_map(w2c_rel, H_train, W_train)
                rendered = self._reconstruct_clip_feature_map(rendered, upto_layer=layer_idx)
            else:
                rendered = self.render_lang(w2c_rel, H_train, W_train)

            # Loss (returns total, l1, cos separately for diagnostics)
            loss, l1_val, cos_val = self.language_loss(
                rendered, target, mask=valid_mask,
                lambda_l1=lambda_l1, lambda_cos=lambda_cos,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Clip gradients: sparse updates (1.8M Gaussians, random frames)
            # can produce large spikes that destabilise training.
            torch.nn.utils.clip_grad_norm_(optim_params, max_norm=1.0)
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
                    if use_langsplat_v2:
                        assert self.language_feature_logits is not None
                        assert self.language_feature_codebooks is not None
                        best_lang_feats = (
                            self.language_feature_logits.detach().cpu().clone(),
                            self.language_feature_codebooks.detach().cpu().clone(),
                        )
                    else:
                        best_lang_feats = self.lang_feats.detach().cpu().clone()
                    best_marker = "  ← best"
                    if output_dir is not None:
                        best_path = Path(output_dir) / "best.pt"
                        if use_langsplat_v2:
                            assert best_lang_feats is not None
                            bl, bc = best_lang_feats
                            b_payload = self._serialize_lang_field_v2(
                                level,
                                bl,
                                bc,
                                best_avg_loss=best_avg_loss,
                                best_iteration=it,
                            )
                        else:
                            assert best_lang_feats is not None
                            b_payload = self._serialize_lang_field_legacy(
                                level,
                                best_lang_feats,
                                best_avg_loss=best_avg_loss,
                                best_iteration=it,
                            )
                        torch.save(b_payload, str(best_path))
                        logger.info(
                            "best.pt updated → %s  (rolling avg=%.6f @ iter %d)",
                            best_path,
                            best_avg_loss,
                            it,
                        )

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
                if use_langsplat_v2:
                    assert self.language_feature_logits is not None
                    assert self.language_feature_codebooks is not None
                    best_logits, best_codebooks = best_lang_feats
                    self.language_feature_logits.copy_(best_logits.to(self.device))
                    self.language_feature_codebooks.copy_(best_codebooks.to(self.device))
                else:
                    self.lang_feats.copy_(best_lang_feats.to(self.device))
            logger.info("Restored best language parameters (avg loss=%.4f)", best_avg_loss)

        # ----- Save final lang_field.pt (best weights restored above) -----
        if output_dir is not None:
            output_dir = Path(output_dir)
            save_path = output_dir / "lang_field.pt"
            if use_langsplat_v2:
                assert self.language_feature_logits is not None
                assert self.language_feature_codebooks is not None
                payload = self._serialize_lang_field_v2(
                    level,
                    self.language_feature_logits.detach().cpu(),
                    self.language_feature_codebooks.detach().cpu(),
                )
            else:
                payload = self._serialize_lang_field_legacy(
                    level,
                    self.lang_feats.detach().cpu(),
                )
            torch.save(payload, str(save_path))
            logger.info("Language field saved → %s", save_path)

    # ------------------------------------------------------------------
    # Open-vocabulary query (inference)
    # ------------------------------------------------------------------

    def load_lang_field(self, lang_field_pt: Path) -> None:
        """
        Load ``lang_field.pt`` produced by ``train_language_field.py`` and set
        model parameters accordingly.
        """
        ckpt = torch.load(str(lang_field_pt), map_location="cpu")
        fmt = ckpt.get("format", "legacy")
        if fmt == "langsplatv2" or "language_feature_logits" in ckpt:
            logits = ckpt.get("language_feature_logits", None)
            codebooks = ckpt.get("language_feature_codebooks", None)
            if logits is None or codebooks is None:
                raise ValueError(f"{lang_field_pt} is missing V2 tensors.")
            self.vq_layer_num = int(ckpt.get("vq_layer_num", codebooks.shape[0]))
            self.codebook_size = int(ckpt.get("codebook_size", codebooks.shape[1]))
            self.topk = int(ckpt.get("topk", self.topk))
            self._ensure_v2_parameters()
            assert self.language_feature_logits is not None
            assert self.language_feature_codebooks is not None
            logits = logits.to(self.device)
            codebooks = codebooks.to(self.device)
            if logits.shape != self.language_feature_logits.shape:
                raise ValueError(
                    f"V2 logits shape mismatch: file {tuple(logits.shape)} vs model {tuple(self.language_feature_logits.shape)}"
                )
            if codebooks.shape != self.language_feature_codebooks.shape:
                raise ValueError(
                    f"V2 codebook shape mismatch: file {tuple(codebooks.shape)} vs model {tuple(self.language_feature_codebooks.shape)}"
                )
            with torch.no_grad():
                self.language_feature_logits.copy_(logits)
                self.language_feature_codebooks.copy_(codebooks)
            self.model_format = "langsplatv2"
        else:
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
            self.model_format = "legacy"
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

        if self.model_format == "langsplatv2":
            rendered = self.render_v2_clip_feature_map(w2c_rel, H, W)
            rendered = F.normalize(rendered, p=2, dim=0)
            r_flat = rendered.permute(1, 2, 0).reshape(-1, 512)
            text_exp = text_emb.unsqueeze(0).expand_as(r_flat)
            sim = F.cosine_similarity(r_flat, text_exp, dim=-1)
        else:
            # Encode text to latent space using legacy LangSplat AE.
            ae = Autoencoder().to(dev)
            state = torch.load(str(ae_ckpt), map_location=dev)
            ae.load_state_dict(state)
            ae.eval()
            text_latent = ae.encode(text_emb.unsqueeze(0))[0]
            rendered = self.render_lang(w2c_rel, H, W)
            r_flat = rendered.permute(1, 2, 0).reshape(-1, self.latent_dim)
            text_latent_exp = text_latent.unsqueeze(0).expand_as(r_flat)
            sim = F.cosine_similarity(r_flat, text_latent_exp, dim=-1)
        relevancy = sim.reshape(H, W)
        relevancy = (relevancy - relevancy.min()) / (relevancy.max() - relevancy.min() + 1e-8)
        return relevancy
