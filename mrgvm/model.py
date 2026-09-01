"""The assembled MRG-VM model: Phase 3 + Phase 4 + Phase 5.

Two architectures live here, selected by config so both stay runnable and
directly comparable:

**v1** (``fusion.kind = 'adaptive' | 'concat'``) -- the published version.
    Each stream is pooled to a clip-level vector first, then the two vectors are
    mixed. Reliability is a fixed function of the scalar MRS.

**v2** (``fusion.kind = 'cross_attention'``) -- the upgrade.
    Both streams stay as sequences until after fusion, so cross-attention can
    align a moment in the video with the landmark measurements from that same
    moment. Reliability is a *learned* combination of the five sub-scores, and a
    learned conditioner modulates the temporal stream through FiLM, a null-token
    gate and a dt scale rather than a fixed multiply.

Every stream and mechanism can be switched off independently, which is what
makes the Phase 6 ablation a set of config overrides rather than a set of
variant models.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import MRGVMConfig
from .fusion import CrossAttentionFusion, MLPClassifier, build_fusion
from .reliability import (
    NUM_SUBSCORES,
    AdaptiveReliabilityConditioner,
    LearnableReliability,
)
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
        self.cross_attention = cfg.fusion.kind == "cross_attention"
        if not self.use_mamba and not self.use_geometric:
            raise ValueError("At least one of use_mamba / use_geometric must be True.")
        if self.cross_attention and not (self.use_mamba and self.use_geometric):
            raise ValueError("cross_attention fusion needs both streams enabled.")

        # ---- learned reliability (v2) ------------------------------------ #
        self.learnable_reliability = (
            LearnableReliability(NUM_SUBSCORES)
            if cfg.reliability.learn_weights
            else None
        )
        self.conditioner = (
            AdaptiveReliabilityConditioner(
                d_model=cfg.vision_mamba.d_model,
                num_factors=NUM_SUBSCORES,
                hidden_dim=cfg.reliability.hidden_dim,
                min_dt_scale=cfg.reliability.min_dt_scale,
                use_film=cfg.reliability.use_film,
                use_gate=cfg.reliability.use_gate,
                use_dt=cfg.reliability.use_dt,
                dropout=cfg.reliability.dropout,
            )
            if (cfg.reliability.adaptive_conditioning and cfg.fusion.use_mamba)
            else None
        )

        # ---- Phase 3 ------------------------------------------------------ #
        self.backbone: Optional[MRGVisionMamba] = None
        mamba_dim = 0
        if self.use_mamba:
            vm = cfg.vision_mamba
            self.backbone = MRGVisionMamba(
                image_size=vm.image_size, patch_size=vm.patch_size, d_model=vm.d_model,
                spatial_depth=vm.spatial_depth, temporal_depth=vm.temporal_depth,
                d_state=vm.d_state, dropout=vm.dropout, max_frames=vm.max_frames,
                guide_delta=vm.guide_delta, guide_pooling=vm.guide_pooling,
                min_delta_scale=vm.min_delta_scale, delta_map=vm.delta_map,
            )
            mamba_dim = vm.d_model

        # ---- Phase 4 ------------------------------------------------------ #
        multiplier = 4 if cfg.fusion.geometric_pooling == "stats" else 1
        self.pooled_geometric_dim = geometric_dim * multiplier if self.use_geometric else 0
        self.raw_geometric_dim = geometric_dim

        if self.cross_attention:
            # Sequence-level fusion: the geometric stream enters per frame, not
            # pooled, so attention can align the two in time.
            self.fusion = CrossAttentionFusion(
                visual_dim=mamba_dim,
                geometric_dim=geometric_dim,
                hidden_dim=cfg.fusion.hidden_dim,
                num_heads=cfg.fusion.num_heads,
                dropout=cfg.fusion.dropout,
            )
        else:
            self.fusion = build_fusion(
                cfg.fusion.kind if (self.use_mamba and self.use_geometric) else "concat",
                mamba_dim, self.pooled_geometric_dim,
                cfg.fusion.hidden_dim, cfg.fusion.dropout,
            )

        # ---- Phase 5 (structure unchanged) -------------------------------- #
        # CORAL emits K-1 cumulative logits instead of K class scores; the MLP is
        # otherwise exactly as before, including depth, widths and dropout.
        num_classes = cfg.classifier.num_classes
        out_features = num_classes - 1 if cfg.train.loss == "coral" else num_classes
        self.classifier = MLPClassifier(
            self.fusion.output_dim, tuple(cfg.classifier.hidden_dims),
            out_features, cfg.classifier.dropout,
        )

    def compute_reliability(
        self, mrs: torch.Tensor, sub_scores: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Per-frame reliability: learned from the five sub-scores, or the scalar."""
        if self.learnable_reliability is not None and sub_scores is not None:
            return self.learnable_reliability(sub_scores)
        return mrs

    def forward(
        self,
        frames: torch.Tensor,
        geometric: torch.Tensor,
        mrs: torch.Tensor,
        mask: torch.Tensor,
        sub_scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        reliability = self.compute_reliability(mrs, sub_scores)

        visual_sequence = None
        visual_vector = None
        if self.use_mamba:
            output = self.backbone(
                frames, reliability, mask,
                conditioner=self.conditioner, sub_scores=sub_scores,
            )
            visual_sequence = output["frame_embeddings"]
            visual_vector = output["clip_embedding"]

        geometric_vector = None
        if self.use_geometric:
            geometric_vector = pool_geometric(
                geometric, mask, reliability, self.cfg.fusion.geometric_pooling,
                weight_by_mrs=self.cfg.vision_mamba.guide_pooling,
            )

        if self.cross_attention:
            fused = self.fusion(
                visual_sequence, geometric, mask,
                pool_weights=reliability if self.cfg.vision_mamba.guide_pooling else None,
            )
        else:
            fused = self.fusion(visual_vector, geometric_vector)

        output = self.classifier(fused["fused"])
        return {
            "logits": output["logits"],
            "penultimate": output["penultimate"],
            "fused": fused["fused"],
            "gate": fused.get("gate"),
            "reliability": reliability,
            "mamba_embedding": visual_vector,
            "geometric_vector": geometric_vector,
            "attention_visual_to_geometric": fused.get("attention_visual_to_geometric"),
            "attention_geometric_to_visual": fused.get("attention_geometric_to_visual"),
        }

    @torch.no_grad()
    def embed(
        self, frames: torch.Tensor, geometric: torch.Tensor,
        mrs: torch.Tensor, mask: torch.Tensor,
        sub_scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Phase 3/4 deliverable: embeddings without the classification head."""
        self.eval()
        output = self.forward(frames, geometric, mrs, mask, sub_scores)
        return {
            "mamba_embedding": output["mamba_embedding"],
            "geometric_vector": output["geometric_vector"],
            "fused": output["fused"],
            "gate": output["gate"],
        }

    def reliability_report(self) -> Dict[str, float]:
        """The learned meaning of 'reliability', for logging and the writeup."""
        if self.learnable_reliability is None:
            return {}
        return self.learnable_reliability.named_weights()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
