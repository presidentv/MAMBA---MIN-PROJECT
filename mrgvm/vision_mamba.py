"""Phase 3 -- Motion Reliability Guided Vision Mamba (MRG-VM).

Pipeline for one clip of T reliable face crops:

    frames (T, 3, H, W)
        -> PatchEmbed                     (T, N_patches, D)
        -> + spatial position embedding
        -> spatial MambaEncoder           bidirectional scan over the patch grid
        -> spatial pool                   (T, D)   per-frame appearance embedding
        -> + temporal position embedding
        -> ReliabilityGuidedTemporalMamba bidirectional scan over time, MRS-guided
        -> reliability-weighted pool      (D,)     clip behavioural embedding

WHERE THE "MOTION RELIABILITY GUIDED" PART ACTUALLY LIVES
---------------------------------------------------------
Phase 1 already discards frames below the MRS threshold, but retained frames are
not equally trustworthy -- an MRS of 0.52 and an MRS of 0.95 both survive the
gate. Two mechanisms carry that residual reliability into the model, and both
are switchable so Phase 6 can ablate them independently:

1. **dt modulation** (``guide_delta``). In a state-space model the timestep dt
   controls how much the hidden state moves at each step: h_t = exp(dt*A)h_{t-1}
   + (dt*B)x_t. Scaling dt by a function of MRS therefore makes a low-reliability
   frame *literally update the state less*, without masking it out or otherwise
   breaking the sequence. This is the natural place to inject a per-timestep
   confidence into an SSM, and it has no equivalent in a transformer, where the
   analogous move would be an attention-bias hack.

2. **Reliability-weighted pooling** (``guide_pooling``). The final temporal pool
   is an MRS-weighted mean rather than a flat mean, so unreliable frames
   contribute proportionally less to the clip embedding.

The mapping from MRS to a dt multiplier is
``scale = min_scale + (1 - min_scale) * mrs``, so a perfect frame is unmodified
(scale 1) and the worst surviving frame is damped to ``min_scale`` rather than
silenced. Zeroing it outright would make the block ignore real motion and is not
what "guided" should mean.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import MambaEncoder, RMSNorm


class PatchEmbed(nn.Module):
    """Split a face crop into non-overlapping patches and linearly embed them."""

    def __init__(
        self, image_size: int = 112, patch_size: int = 16, in_channels: int = 3, d_model: int = 128
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size {image_size} must be divisible by patch_size {patch_size}")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        self.projection = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, num_patches, d_model)."""
        x = self.projection(x)                 # (B, D, grid, grid)
        return x.flatten(2).transpose(1, 2)    # (B, num_patches, D)


class SpatialVisionMamba(nn.Module):
    """Bidirectional Mamba over the patch sequence of a single frame."""

    def __init__(
        self,
        image_size: int = 112,
        patch_size: int = 16,
        d_model: int = 128,
        depth: int = 2,
        d_state: int = 16,
        dropout: float = 0.1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, 3, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.encoder = MambaEncoder(d_model, depth, d_state, dropout=dropout)
        self.pooling = pooling

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """(B*T, 3, H, W) -> (B*T, d_model) per-frame appearance embedding."""
        tokens = self.patch_embed(frames) + self.position
        tokens = self.dropout(tokens)
        encoded = self.encoder(tokens)
        if self.pooling == "max":
            return encoded.max(dim=1).values
        return encoded.mean(dim=1)


class ReliabilityGuidedTemporalMamba(nn.Module):
    """Bidirectional Mamba over time, with MRS driving dt and the final pooling."""

    def __init__(
        self,
        d_model: int = 128,
        depth: int = 2,
        d_state: int = 16,
        dropout: float = 0.1,
        max_frames: int = 64,
        guide_delta: bool = True,
        guide_pooling: bool = True,
        min_delta_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, max_frames, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.encoder = MambaEncoder(d_model, depth, d_state, dropout=dropout)
        self.norm = RMSNorm(d_model)
        self.guide_delta = guide_delta
        self.guide_pooling = guide_pooling
        self.min_delta_scale = min_delta_scale

    def reliability_to_delta_scale(self, mrs: torch.Tensor) -> torch.Tensor:
        floor = self.min_delta_scale
        return floor + (1.0 - floor) * mrs.clamp(0.0, 1.0)

    def forward(
        self, frame_embeddings: torch.Tensor, mrs: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frame_embeddings: (B, T, d_model)
            mrs:              (B, T) per-frame Motion Reliability Score in [0, 1]
            mask:             (B, T) True where the frame is real (not padding)

        Returns:
            (clip_embedding (B, d_model), per_frame (B, T, d_model))
        """
        batch, length, _ = frame_embeddings.shape
        x = frame_embeddings + self.position[:, :length]
        x = self.dropout(x)

        delta_scale = None
        if self.guide_delta:
            delta_scale = self.reliability_to_delta_scale(mrs)
            # Padding contributes nothing: floor its dt so the state barely moves.
            delta_scale = torch.where(mask, delta_scale, torch.full_like(delta_scale, 1e-3))

        encoded = self.norm(self.encoder(x, delta_scale))

        weights = mask.to(encoded.dtype)
        if self.guide_pooling:
            weights = weights * mrs.clamp(min=0.0).to(encoded.dtype)
        denominator = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        clip_embedding = (encoded * weights.unsqueeze(-1)).sum(dim=1) / denominator
        return clip_embedding, encoded


class MRGVisionMamba(nn.Module):
    """Phase 3 end to end: reliable face crops -> deep behavioural embedding."""

    def __init__(
        self,
        image_size: int = 112,
        patch_size: int = 16,
        d_model: int = 128,
        spatial_depth: int = 2,
        temporal_depth: int = 2,
        d_state: int = 16,
        dropout: float = 0.1,
        max_frames: int = 64,
        guide_delta: bool = True,
        guide_pooling: bool = True,
        min_delta_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.spatial = SpatialVisionMamba(
            image_size, patch_size, d_model, spatial_depth, d_state, dropout
        )
        self.temporal = ReliabilityGuidedTemporalMamba(
            d_model, temporal_depth, d_state, dropout, max_frames,
            guide_delta, guide_pooling, min_delta_scale,
        )

    @property
    def embedding_dim(self) -> int:
        return self.d_model

    def forward(
        self, frames: torch.Tensor, mrs: torch.Tensor, mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames: (B, T, 3, H, W) normalised face crops
            mrs:    (B, T) Motion Reliability Score per frame
            mask:   (B, T) True where the frame is real

        Returns:
            dict with ``clip_embedding`` (B, D) and ``frame_embeddings`` (B, T, D)
        """
        batch, length = frames.shape[:2]
        flattened = frames.reshape(batch * length, *frames.shape[2:])
        per_frame = self.spatial(flattened).reshape(batch, length, self.d_model)
        clip_embedding, encoded = self.temporal(per_frame, mrs, mask)
        return {
            "clip_embedding": clip_embedding,
            "frame_embeddings": encoded,
            "spatial_embeddings": per_frame,
        }
