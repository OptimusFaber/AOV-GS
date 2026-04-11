"""
Open-vocabulary CLIP encoder.

Wraps open_clip to provide image and text embeddings for
goal-directed navigation queries.

The encoder is intentionally lazy-loaded so that the model
is only placed on the GPU when first used, keeping the
default memory footprint identical to the original ActiveSGM.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """Thin, stateless wrapper around open_clip.

    Parameters
    ----------
    model_name:
        open_clip model identifier, e.g. ``"ViT-B-32"``.
    pretrained:
        Pre-trained weights tag, e.g. ``"openai"`` or ``"laion2b_s34b_b79k"``.
    device:
        Target device string (``"cuda:0"``, ``"cuda:1"``, ``"cpu"``…).
        Defaults to ``"cuda:1"`` when two GPUs are available (so CLIP runs
        alongside OneFormer on the second GPU), otherwise falls back to
        ``"cuda:0"`` or ``"cpu"``.  The user can override this freely via
        the ``clip.device`` config key.
    """

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Return *device* if it exists, otherwise fall back gracefully."""
        if not device.startswith("cuda"):
            return device
        try:
            idx = int(device.split(":")[-1]) if ":" in device else 0
            if idx < torch.cuda.device_count():
                return device
        except (ValueError, RuntimeError):
            pass
        # Requested CUDA device is not available – fall back
        if torch.cuda.is_available():
            fallback = "cuda:0"
        else:
            fallback = "cpu"
        logger.warning(
            "Requested CLIP device %r is not available "
            "(found %d CUDA device(s)). Falling back to %r.",
            device,
            torch.cuda.device_count(),
            fallback,
        )
        return fallback

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda:1",
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = self._resolve_device(device)

        self._model = None
        self._preprocess = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open_clip is required for the CLIP encoder. "
                "Install it with: pip install open_clip_torch"
            ) from exc

        logger.info(
            "Loading CLIP model %s (%s) onto %s…",
            self.model_name,
            self.pretrained,
            self.device,
        )
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        logger.info("CLIP model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_image(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Return a normalised L2 image embedding of shape ``[D]``.

        Parameters
        ----------
        image:
            A PIL image, an ``np.ndarray`` of shape ``[H, W, 3]`` (uint8),
            or a ``torch.Tensor`` of shape ``[3, H, W]`` with values in
            ``[0, 1]``.
        """
        self._ensure_loaded()
        pil = self._to_pil(image)
        tensor = self._preprocess(pil).unsqueeze(0).to(self.device)
        emb = self._model.encode_image(tensor)
        return F.normalize(emb[0], dim=-1)

    @torch.no_grad()
    def encode_images_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, torch.Tensor]],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Encode a list of images in batches.

        Returns
        -------
        torch.Tensor of shape ``[N, D]``, L2-normalised.
        """
        self._ensure_loaded()
        all_embs: List[torch.Tensor] = []
        for start in range(0, len(images), batch_size):
            batch_pils = [self._to_pil(img) for img in images[start : start + batch_size]]
            tensors = torch.stack(
                [self._preprocess(p) for p in batch_pils]
            ).to(self.device)
            embs = self._model.encode_image(tensors)
            all_embs.append(F.normalize(embs, dim=-1))
        return torch.cat(all_embs, dim=0)

    @torch.no_grad()
    def encode_text(self, query: str) -> torch.Tensor:
        """Return a normalised L2 text embedding of shape ``[D]``.

        Parameters
        ----------
        query:
            Free-form text query, e.g. ``"a white chair"`` or
            ``"a kitchen counter"``.
        """
        self._ensure_loaded()
        tokens = self._tokenizer([query]).to(self.device)
        emb = self._model.encode_text(tokens)
        return F.normalize(emb[0], dim=-1)

    @torch.no_grad()
    def encode_texts_batch(self, queries: List[str]) -> torch.Tensor:
        """Encode a list of text queries.

        Returns
        -------
        torch.Tensor of shape ``[N, D]``, L2-normalised.
        """
        self._ensure_loaded()
        tokens = self._tokenizer(queries).to(self.device)
        embs = self._model.encode_text(tokens)
        return F.normalize(embs, dim=-1)

    # ------------------------------------------------------------------
    # Similarity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between two L2-normalised embeddings.

        Parameters
        ----------
        a:
            ``[D]`` or ``[N, D]`` embedding(s).
        b:
            ``[D]`` embedding (broadcast over ``a`` if ``a`` is batched).

        Returns
        -------
        Scalar or ``[N]`` tensor of similarities in ``[-1, 1]``.
        """
        if a.dim() == 1:
            return (a * b).sum()
        return (a * b.unsqueeze(0)).sum(dim=-1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pil(
        image: Union[Image.Image, np.ndarray, torch.Tensor],
    ) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, torch.Tensor):
            # [3, H, W] float [0,1]  →  PIL
            arr = (image.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(image)
        raise TypeError(f"Unsupported image type: {type(image)}")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embeddings."""
        self._ensure_loaded()
        return self._model.visual.output_dim
