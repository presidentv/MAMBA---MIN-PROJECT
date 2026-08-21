"""The assembled MRG-VM model: Phase 3 + Phase 4 + Phase 5.

    frames + MRS  --Phase 3-->  Vision Mamba clip embedding
    geometric     --pooling-->  clip-level geometric vector
                  --Phase 4-->  adaptive gated fusion
                  --Phase 5-->  lightweight MLP -> engagement logits

Every stream can be switched off from the config, which is what makes the
Phase 6 ablation table a set of config overrides rather than six variant models.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import MRGVMConfig
from .fusion import MLPClassifier, build_fusion
from .vision_mamba import MRGVisionMamba


def pool_geometric(
    geometric: torch.Tensor, mask: torch.Tensor, mrs: torch.Tensor,
    mode: str = "stats", weight_by_mrs: bool = True,
) -> torch.Tensor:
    """Collapse a (B, T, D) geometric sequence to a clip-level vector.

    ``stats`` concatenates mean/std/min/max, which preserves the *dynamics* of
    each descriptor -- the std of head yaw is head-pose stability, and a flat
    mean would throw exactly that away.
    """
    weights = mask.to(geometric.dtype)
    if weight_by_mrs:
        weights = weights * mrs.clamp(min=0.0).to(geometric.dtype)
    weights = weights.unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp(min=1e-6)

    mean = (geometric * weights).sum(dim=1) / denominator
    if mode == "mean":
        return mean

    variance = (((geometric - mean.unsqueeze(1)) ** 2) * weights).sum(dim=1) / denominator
    std = torch.sqrt(variance.clamp(min=1e-12))

    very_large = torch.finfo(geometric.dtype).max
    valid = mask.unsqueeze(-1)
    minimum = geometric.masked_fill(~valid, very_large).min(dim=1).values
    maximum = geometric.masked_fill(~valid, -very_large).max(dim=1).values
    return torch.cat([mean, std, minimum, maximum], dim=-1)


class MRGVMModel(nn.Module):
    """Full Motion Reliability Guided Vision Mamba engagement classifier."""

    def __init__(self, cfg: MRGVMConfig, geometric_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_mamba = cfg.fusion.use_mamba
        self.use_geometric = cfg.fusion.use_geometric
        if not self.use_mamba and not self.use_geometric:
            raise ValueError("At least one of use_mamba / use_geometric must be True.")

        self.backbone: Optional[MRGVisionMamba] = None
        mamba_dim = 0
        if self.use_mamba:
            vm = cfg.vision_mamba
            self.backbone = MRGVisionMamba(
                image_size=vm.image_size, patch_size=vm.patch_size, d_model=vm.d_model,
                spatial_depth=vm.spatial_depth, temporal_depth=vm.temporal_depth,
                d_state=vm.d_state, dropout=vm.dropout, max_frames=vm.max_frames,
                guide_delta=vm.guide_delta, guide_pooling=vm.guide_pooling,
                min_delta_scale=vm.min_delta_scale,
            )
            mamba_dim = vm.d_model

        multiplier = 4 if cfg.fusion.geometric_pooling == "stats" else 1
        self.geometric_dim = geometric_dim * multiplier if self.use_geometric else 0

        self.fusion = build_fusion(
            cfg.fusion.kind if (self.use_mamba and self.use_geometric) else "concat",
            mamba_dim, self.geometric_dim, cfg.fusion.hidden_dim, cfg.fusion.dropout,
        )
        self.classifier = MLPClassifier(
            self.fusion.output_dim, tuple(cfg.classifier.hidden_dims),
            cfg.classifier.num_classes, cfg.classifier.dropout,
        )

    def forward(
        self,
        frames: torch.Tensor,
        geometric: torch.Tensor,
        mrs: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mamba_embedding = None
        if self.use_mamba:
            mamba_embedding = self.backbone(frames, mrs, mask)["clip_embedding"]

        geometric_vector = None
        if self.use_geometric:
            geometric_vector = pool_geometric(
                geometric, mask, mrs, self.cfg.fusion.geometric_pooling,
                weight_by_mrs=self.cfg.vision_mamba.guide_pooling,
            )

        fused = self.fusion(mamba_embedding, geometric_vector)
        output = self.classifier(fused["fused"])
        return {
            "logits": output["logits"],
            "penultimate": output["penultimate"],
            "fused": fused["fused"],
            "gate": fused["gate"],
            "mamba_embedding": mamba_embedding,
            "geometric_vector": geometric_vector,
        }

    @torch.no_grad()
    def embed(
        self, frames: torch.Tensor, geometric: torch.Tensor,
        mrs: torch.Tensor, mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Phase 3/4 deliverable: embeddings without the classification head."""
        self.eval()
        output = self.forward(frames, geometric, mrs, mask)
        return {
            "mamba_embedding": output["mamba_embedding"],
            "geometric_vector": output["geometric_vector"],
            "fused": output["fused"],
            "gate": output["gate"],
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
