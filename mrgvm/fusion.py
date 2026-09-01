"""Phase 4 -- Adaptive Feature Fusion, and Phase 5 -- the MLP classifier.

Phase 4 fuses two heterogeneous streams:

    Vision Mamba clip embedding   (D_m)  learned, dense, opaque
    landmark geometric features   (D_g)  hand-engineered, sparse, interpretable

"Adaptive" is taken to mean the mixture is *learned per clip* rather than fixed.
``AdaptiveFeatureFusion`` projects both streams to a common width, then computes
a gate from their concatenation and uses it to interpolate between them. A clip
whose frames are clean and whose landmarks are stable can lean on geometry; a
clip where the mesh was jittery can lean on appearance. The gate is returned
alongside the fused vector so Phase 7 can report which stream actually drove
each prediction.

``ConcatFusion`` is the ablation control for Phase 6: same parameter budget,
same projections, no gating. If adaptive fusion cannot beat it, the gate is
decoration.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    """Ablation control: project both streams, concatenate, no gating."""

    def __init__(self, mamba_dim: int, geometric_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.mamba_projection = nn.Sequential(
            nn.Linear(mamba_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        ) if mamba_dim > 0 else None
        self.geometric_projection = nn.Sequential(
            nn.Linear(geometric_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        ) if geometric_dim > 0 else None
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * sum(
            1 for p in (self.mamba_projection, self.geometric_projection) if p is not None
        )

    def forward(
        self, mamba: Optional[torch.Tensor], geometric: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        parts = []
        if self.mamba_projection is not None and mamba is not None:
            parts.append(self.mamba_projection(mamba))
        if self.geometric_projection is not None and geometric is not None:
            parts.append(self.geometric_projection(geometric))
        fused = torch.cat(parts, dim=-1)
        return {"fused": self.dropout(fused), "gate": None}


class AdaptiveFeatureFusion(nn.Module):
    """Learned per-clip gated interpolation between the two streams."""

    def __init__(self, mamba_dim: int, geometric_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.2) -> None:
        super().__init__()
        if mamba_dim <= 0 or geometric_dim <= 0:
            raise ValueError(
                "AdaptiveFeatureFusion needs both streams; use ConcatFusion for "
                "single-stream ablations."
            )
        self.mamba_projection = nn.Sequential(
            nn.Linear(mamba_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.geometric_projection = nn.Sequential(
            nn.Linear(geometric_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        # Gate sees both streams and emits one weight per feature channel, so the
        # mixture can differ across dimensions rather than being a single scalar.
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2

    def forward(
        self, mamba: torch.Tensor, geometric: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        m = self.mamba_projection(mamba)
        g = self.geometric_projection(geometric)
        gate = torch.sigmoid(self.gate(torch.cat([m, g], dim=-1)))
        mixed = gate * m + (1.0 - gate) * g
        # Keep the mixture AND the interaction term: the gate alone would discard
        # information whenever it saturates.
        fused = self.norm(torch.cat([mixed, m * g], dim=-1))
        return {"fused": self.dropout(fused), "gate": gate}


def build_fusion(
    kind: str, mamba_dim: int, geometric_dim: int, hidden_dim: int, dropout: float
) -> nn.Module:
    if kind == "adaptive" and mamba_dim > 0 and geometric_dim > 0:
        return AdaptiveFeatureFusion(mamba_dim, geometric_dim, hidden_dim, dropout)
    # Falls back to concatenation for kind == "concat" and for any single-stream
    # ablation, where there is nothing to gate between.
    return ConcatFusion(mamba_dim, geometric_dim, hidden_dim, dropout)


class MLPClassifier(nn.Module):
    """Phase 5 -- deliberately lightweight engagement head.

    The PDF specifies a *lightweight* MLP, and at this sample size that is not a
    compromise but the correct choice: two hidden layers with dropout is already
    more capacity than ~36 training clips can constrain.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64),
        num_classes: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers = []
        current = input_dim
        for width in hidden_dims:
            layers += [
                nn.Linear(current, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            current = width
        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(current, num_classes)
        self.feature_dim = current

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.backbone(x)
        return {"logits": self.output(hidden), "penultimate": hidden}


class CrossAttentionFusion(nn.Module):
    """Phase 4 (v2) -- cross-attention between the two temporal streams.

    Replaces the clip-level gated fusion. The important difference is *where* it
    sits: gated fusion pooled each stream to a single vector first and then mixed
    two vectors, so it could never align a moment in the video with the landmark
    measurements from that same moment. Cross-attention operates on the
    sequences, so the behavioural stream can attend to the geometric stream
    frame by frame and vice versa.

    Attention is bidirectional and symmetric, so neither modality is privileged.
    Per-head weights are averaged and returned, because Phase 7 needs them for
    attention rollout and because "which frames did the model look at" is a
    reportable result on its own.
    """

    def __init__(
        self, visual_dim: int, geometric_dim: int, hidden_dim: int = 128,
        num_heads: int = 4, dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.geometric_projection = nn.Sequential(
            nn.Linear(geometric_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.visual_to_geometric = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.geometric_to_visual = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_visual = nn.LayerNorm(hidden_dim)
        self.norm_geometric = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2

    def forward(
        self,
        visual: torch.Tensor,
        geometric: torch.Tensor,
        mask: torch.Tensor,
        pool_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual:    (B, T, Dv) temporal behavioural features
            geometric: (B, T, Dg) landmark features, aligned frame-for-frame
            mask:      (B, T) True where the frame is real
            pool_weights: optional (B, T) reliability weights for the final pool

        Returns ``fused`` (B, 2*hidden) plus the two attention maps.
        """
        v = self.visual_projection(visual)
        g = self.geometric_projection(geometric)
        pad = ~mask

        attended_v, weights_vg = self.visual_to_geometric(
            v, g, g, key_padding_mask=pad, need_weights=True, average_attn_weights=True
        )
        attended_g, weights_gv = self.geometric_to_visual(
            g, v, v, key_padding_mask=pad, need_weights=True, average_attn_weights=True
        )
        v = self.norm_visual(v + self.dropout(attended_v))
        g = self.norm_geometric(g + self.dropout(attended_g))

        weights = mask.to(v.dtype)
        if pool_weights is not None:
            weights = weights * pool_weights.clamp(min=0.0).to(v.dtype)
        denominator = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled_v = (v * weights.unsqueeze(-1)).sum(dim=1) / denominator
        pooled_g = (g * weights.unsqueeze(-1)).sum(dim=1) / denominator

        return {
            "fused": self.dropout(torch.cat([pooled_v, pooled_g], dim=-1)),
            "gate": None,
            "attention_visual_to_geometric": weights_vg,
            "attention_geometric_to_visual": weights_gv,
        }
