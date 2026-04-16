"""
models/autoencoder.py
Convolutional Autoencoder for anomaly detection on 224×224 face images.
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────

def _enc_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv → BN → LeakyReLU (stride-2 to halve spatial dims)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """ConvTranspose → BN → ReLU (stride-2 to double spatial dims)."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ──────────────────────────────────────────────────────────────
# Autoencoder
# ──────────────────────────────────────────────────────────────

class ConvAutoencoder(nn.Module):
    """
    Symmetric Convolutional Autoencoder.

    Input  : (B, 3, img_size, img_size)
    Latent : (B, latent_dim)
    Output : (B, 3, img_size, img_size)

    Spatial flow (encoder):
        img_size → img_size/2 → img_size/4 → img_size/8 → img_size/16 → img_size/32
    Then flattened → linear bottleneck → reshaped → decoder mirrors encoder.
    """

    def __init__(self, latent_dim: int = 256, img_size: int = 224):
        super().__init__()
        assert img_size % 32 == 0, "img_size must be divisible by 32"
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.spatial_dim = img_size // 32

        # ── Encoder ──────────────────────────────────────────
        # 3×224×224  →  32×112×112
        # 32×112×112 →  64×56×56
        # 64×56×56   → 128×28×28
        # 128×28×28  → 256×14×14
        # 256×14×14  → 512×7×7
        self.encoder_cnn = nn.Sequential(
            _enc_block(3,   32),
            _enc_block(32,  64),
            _enc_block(64,  128),
            _enc_block(128, 256),
            _enc_block(256, 512),
        )  # → (B, 512, 7, 7)

        self.flatten    = nn.Flatten()                       # → (B, 512*spatial_dim*spatial_dim)
        self.enc_fc     = nn.Linear(512 * self.spatial_dim * self.spatial_dim, latent_dim)

        # ── Decoder ──────────────────────────────────────────
        self.dec_fc     = nn.Linear(latent_dim, 512 * self.spatial_dim * self.spatial_dim)
        self.unflatten  = nn.Unflatten(1, (512, self.spatial_dim, self.spatial_dim))

        self.decoder_cnn = nn.Sequential(
            _dec_block(512, 256),   # → 14×14
            _dec_block(256, 128),   # → 28×28
            _dec_block(128, 64),    # → 56×56
            _dec_block(64,  32),    # → 112×112
            # Final layer: upsample to 224×224, output pixel values in [0, 1]
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    # ----------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_cnn(x)
        return self.enc_fc(self.flatten(h))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.unflatten(self.dec_fc(z))
        return self.decoder_cnn(h)

    def forward(self, x: torch.Tensor):
        z    = self.encode(x)
        recon = self.decode(z)
        return recon, z


# ──────────────────────────────────────────────────────────────
# Anomaly score helper
# ──────────────────────────────────────────────────────────────

def reconstruction_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample MSE between original and reconstructed images.

    Args:
        original      : (B, C, H, W)
        reconstructed : (B, C, H, W)

    Returns:
        errors : (B,)  — one scalar per sample
    """
    return ((original - reconstructed) ** 2).mean(dim=[1, 2, 3])