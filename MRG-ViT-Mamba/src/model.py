"""The assembled MRG-ViT--Mamba model.

    ViT frame embeddings [B,T,D]  --(x MRS)-->  Mamba  -->  pooled [B,d_model]
                                                                    |
                              landmark clip features [B,3L] --------+
                                                                    v
                                          normalise -> project -> fuse -> MLP -> [B,4]

Terminology (spec section 3): this is **ViT + Mamba temporal modelling** -- a ViT
spatial encoder per frame, then Mamba across frames. It is deliberately *not*
called Vim/Vision Mamba, because the visual backbone is a ViT, not a
bidirectional Mamba over image tokens.

The model operates on *cached* ViT features by default (``forward``), which is
what training uses. ``forward_from_frames`` runs the ViT inline for the
end-to-end path and for fine-tuning.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .classifier import MLPClassifier
from .fusion import BranchProjection, FeatureFusion
from .mamba_ref import mamba_backend_info
from .temporal_mamba import MRSWeighting, TemporalMamba


class LearnableMRSWeights(nn.Module):
    """Learns how to combine the five MRS components: r_t = sum_c w_c * comp_c(t).

    w = softmax(theta) keeps the weights positive and summing to one with no
    projection step, and theta initialised to zeros starts training at exactly
    the 0.2-each baseline. Gradient reaches theta through both reliability
    routes -- the feature multiply and the reliability-weighted pooling.

    IMPORTANT SEMANTIC NOTE. Learning these weights from the classification loss
    means engagement labels shape MRS, which departs from the spec rule that MRS
    is computed from observation quality alone. The five *components* remain
    label-free; only their combination is learned. The concrete risk is that the
    model raises w_head_pose because turned heads correlate with disengagement in
    the training set, turning a reliability term into an engagement feature. The
    learned weights are logged every epoch so that drift is visible rather than
    hidden.
    """

    def __init__(self, num_components: int = 5):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(num_components))

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.theta, dim=0)

    def forward(self, components: torch.Tensor) -> torch.Tensor:
        if components.dim() != 3:
            raise ValueError(f"expected [B, T, C], got {tuple(components.shape)}")
        r = (components * self.weights()).sum(dim=-1)
        return r.clamp(0.0, 1.0)


class MRGViTMamba(nn.Module):
    def __init__(self, vit_dim: int, landmark_dim: int, cfg,
                 vit_encoder: nn.Module | None = None):
        super().__init__()
        self.vit_dim = int(vit_dim)
        self.landmark_dim = int(landmark_dim)
        self.cfg = cfg

        m = cfg["mamba"]
        f = cfg["fusion"]
        c = cfg["classifier"]
        proj_dim = int(f["projected_dim"])
        self.use_landmarks = bool(f.get("use_landmark_branch", True))

        self.vit_encoder = vit_encoder  # optional; only needed for the end-to-end path

        self.learnable_mrs = (LearnableMRSWeights(5)
                              if bool(cfg["mrs"].get("learnable_weights", False)) else None)
        self.mrs_weighting = MRSWeighting(
            mode=cfg["mrs"].get("weighting_mode", "multiply"),
            floor=float(cfg["mrs"].get("residual_floor", 0.2)),
        )
        self.temporal = TemporalMamba(
            input_dim=self.vit_dim,
            d_model=int(m["d_model"]),
            n_layers=int(m["n_layers"]),
            d_state=int(m["d_state"]),
            d_conv=int(m["d_conv"]),
            expand=int(m["expand"]),
            dropout=float(m.get("dropout", 0.1)),
            bidirectional=bool(m.get("bidirectional", False)),
            pooling=str(m.get("pooling", "mean")),
        )

        self.deep_branch = BranchProjection(
            input_dim=int(m["d_model"]), output_dim=proj_dim,
            norm=str(f.get("deep_norm", "layernorm")), dropout=float(f.get("dropout", 0.1)))
        self.landmark_branch = (
            BranchProjection(input_dim=self.landmark_dim, output_dim=proj_dim,
                             norm=str(f.get("landmark_norm", "standard")),
                             dropout=float(f.get("dropout", 0.1)))
            if self.use_landmarks else None
        )
        self.fusion = FeatureFusion(proj_dim, mode=str(f.get("mode", "concat")),
                                    use_landmark_branch=self.use_landmarks)
        self.classifier = MLPClassifier(
            input_dim=self.fusion.output_dim,
            num_classes=int(cfg["dataset"]["num_classes"]),
            hidden_dims=tuple(int(x) for x in c["hidden_dims"]),
            activation=str(c.get("activation", "gelu")),
            dropout=float(c.get("dropout", 0.3)),
        )

    # ------------------------------------------------------------------ paths
    def forward(self, vit_features: torch.Tensor, mrs: torch.Tensor,
                landmark_features: torch.Tensor | None = None,
                return_intermediates: bool = False,
                mrs_components: torch.Tensor | None = None):
        """Cached-feature path.

        vit_features      [B, T, vit_dim]
        mrs               [B, T]           reliability in [0, 1]
        landmark_features [B, landmark_dim]
        """
        # With learnable weights the reliability score is recomputed here from the
        # cached components, so the combination is part of the graph.
        if self.learnable_mrs is not None and mrs_components is not None:
            mrs = self.learnable_mrs(mrs_components)
        weighted = self.mrs_weighting(vit_features, mrs)
        # mrs is passed on as well: with pooling="mrs_weighted" the temporal
        # aggregation is where the reliability signal survives normalisation.
        sequence, pooled = self.temporal(weighted, mrs)
        deep = self.deep_branch(pooled)

        land = None
        if self.use_landmarks:
            if landmark_features is None:
                raise ValueError("landmark branch is enabled but no landmark features were given")
            land = self.landmark_branch(landmark_features)

        fused = self.fusion(deep, land)
        logits = self.classifier(fused)

        if return_intermediates:
            return logits, {
                "weighted_features": weighted,
                "temporal_sequence": sequence,
                "temporal_pooled": pooled,
                "deep_projected": deep,
                "landmark_projected": land,
                "fused": fused,
                "mrs_used": mrs,
            }
        return logits

    def forward_from_frames(self, frames: torch.Tensor, mrs: torch.Tensor,
                            landmark_features: torch.Tensor | None = None):
        """End-to-end path: [B, T, C, H, W] -> [B, num_classes]."""
        if self.vit_encoder is None:
            raise RuntimeError(
                "forward_from_frames needs a ViT encoder. Construct the model with "
                "vit_encoder=..., or use the cached-feature forward().")
        return self.forward(self.vit_encoder(frames), mrs, landmark_features)

    def classify_fused(self, fused: torch.Tensor) -> torch.Tensor:
        """Entry point for SHAP, which attributes over the fused representation."""
        return self.classifier(fused)

    # ------------------------------------------------------------------- info
    def parameter_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        per_part = {}
        for name in ("temporal", "deep_branch", "landmark_branch", "fusion", "classifier"):
            module = getattr(self, name, None)
            if module is not None:
                per_part[name] = sum(p.numel() for p in module.parameters())
        if self.vit_encoder is not None:
            per_part["vit_encoder"] = sum(p.numel() for p in self.vit_encoder.parameters())
        return {"total": total, "trainable": trainable,
                "frozen": total - trainable, "per_module": per_part}

    def describe(self) -> dict:
        return {
            "architecture": "ViT spatial encoder + Mamba temporal encoder "
                            "(MRG-ViT--Mamba); not Vim/Vision Mamba",
            "vit_dim": self.vit_dim,
            "landmark_dim": self.landmark_dim if self.use_landmarks else None,
            "mrs_weighting_mode": self.mrs_weighting.mode,
            "learnable_mrs_weights": (
                [round(float(v), 6) for v in self.learnable_mrs.weights().detach().cpu()]
                if self.learnable_mrs is not None else None),
            "mamba": {
                "d_model": self.temporal.d_model,
                "n_layers": len(self.temporal.blocks),
                "bidirectional": self.temporal.bidirectional,
                "pooling": self.temporal.pooling,
                **mamba_backend_info(),
            },
            "fusion": {"mode": self.fusion.mode, "output_dim": self.fusion.output_dim},
            "classifier_input_dim": self.classifier.input_dim,
            "num_classes": self.classifier.num_classes,
            "parameters": self.parameter_counts(),
        }
