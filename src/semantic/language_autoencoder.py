"""
Autoencoder to compress CLIP 512d → latent_dim d (LangSplat-compatible).

Default configuration (64d):
  Encoder: 512 → 256 → 128 → 64  (Linear + BN + ReLU + L2-norm)
  Decoder:  64 → 128 → 256 → 512  (Linear + ReLU + L2-norm)

LangSplat-compatible variant (3d):
  Encoder: 512 → 256 → 128 → 64 → 32 → 3
  Decoder:   3 → 16  → 32  → 64 → 128 → 256 → 256 → 512

Loss: L2 + 0.001 * cosine  (as in LangSplat autoencoder/train.py)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    """
    Autoencoder for CLIP features: 512d → latent_dim d → 512d.

    Parameters
    ----------
    encoder_hidden_dims : list of int
        Encoder layer dims (last = latent_dim).
        Default [256, 128, 64] → latent 64d.
    decoder_hidden_dims : list of int
        Decoder layer dims (last must be 512).
        Default [128, 256, 512].
    """

    def __init__(
        self,
        encoder_hidden_dims: list = None,
        decoder_hidden_dims: list = None,
    ) -> None:
        super().__init__()
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [256, 128, 64]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [128, 256, 512]

        encoder_layers = []
        for i in range(len(encoder_hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(512, encoder_hidden_dims[i]))
            else:
                encoder_layers.append(nn.BatchNorm1d(encoder_hidden_dims[i - 1]))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i - 1], encoder_hidden_dims[i]))
        self.encoder = nn.ModuleList(encoder_layers)

        decoder_layers = []
        for i in range(len(decoder_hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(encoder_hidden_dims[-1], decoder_hidden_dims[i]))
            else:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Linear(decoder_hidden_dims[i - 1], decoder_hidden_dims[i]))
        self.decoder = nn.ModuleList(decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full AE forward → reconstructed vector (L2-normalized)."""
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """512d → latent_dim (L2-normalized)."""
        for m in self.encoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """latent_dim → 512d (L2-normalized)."""
        for m in self.decoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    @staticmethod
    def l2_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((output - target) ** 2).mean()

    @staticmethod
    def cos_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1 - F.cosine_similarity(output, target, dim=-1).mean()

    @staticmethod
    def total_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L2 + 0.001 * cosine — as in LangSplat autoencoder/train.py."""
        return Autoencoder.l2_loss(output, target) + 0.001 * Autoencoder.cos_loss(output, target)
