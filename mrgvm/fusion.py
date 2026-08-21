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
