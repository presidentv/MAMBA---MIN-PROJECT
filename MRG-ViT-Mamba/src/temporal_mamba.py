"""MRS reliability weighting, the Mamba temporal encoder, and temporal pooling.

This is the reliability-guided mechanism at the centre of the project
(spec sections 11, 14, 15, 16):

    [B, T, D_vit]  x  [B, T]  ->  [B, T, D_vit]      reliability weighting
                                   -> Linear -> [B, T, d_model]
                                   -> Mamba stack   -> [B, T, d_model]
                                   -> pooling       -> [B, d_model]

The temporal length T and the ordering of positions are invariant through the
weighting step. Frames are never dropped; unreliable ones are attenuated in
place, which is what keeps the sequence intact for the state-space model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .mamba_ref import RMSNorm, build_mamba_block, mamba_backend_info


class MRSWeighting(nn.Module):
    """Applies the per-frame reliability score to the per-frame ViT features.

    Modes:
      ``multiply``  f'_t = r_t * f_t                (spec section 11 baseline)
      ``none``      f'_t = f_t                      (ablation: no reliability signal)
      ``residual``  f'_t = (floor + (1-floor)*r_t) * f_t

    ``residual`` exists because pure multiplication drives a frame's contribution
    to exactly zero as r_t -> 0, which removes the frame's content while keeping
    its position. The floor keeps a fraction of the signal. It is offered as a
    labelled option, not as an improvement -- scripts/run_ablation.py measures it.
    """

    def __init__(self, mode: str = "multiply", floor: float = 0.2):
        super().__init__()
        if mode not in ("multiply", "none", "residual"):
            raise ValueError(f"unknown MRS weighting mode: {mode!r}")
        self.mode = mode
        self.floor = float(floor)

    def forward(self, features: torch.Tensor, mrs: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return features
        if features.dim() != 3:
            raise ValueError(f"expected features [B, T, D], got {tuple(features.shape)}")
        if mrs.shape != features.shape[:2]:
            raise ValueError(
                f"MRS shape {tuple(mrs.shape)} does not match features {tuple(features.shape[:2])}")
        r = mrs.unsqueeze(-1)
        if self.mode == "residual":
            r = self.floor + (1.0 - self.floor) * r
        out = features * r
        assert out.shape == features.shape, "MRS weighting must preserve [B, T, D]"
        return out

    def extra_repr(self) -> str:
        return f"mode={self.mode}, floor={self.floor}"


class AttentionPool(nn.Module):
    """Learned scalar attention over time. Offered alongside mean pooling so that
    'mean vs. something else' is an experiment rather than an assumption."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)   # [B, T, 1]
        return (x * weights).sum(dim=1)


POOLING_MODES = ("mean", "last", "attention", "mrs_weighted")


class TemporalMamba(nn.Module):
    """Projection to d_model, a stack of Mamba blocks, then temporal pooling.

    A NOTE ON NORMALISATION AND WHY MRS POOLING EXISTS
    --------------------------------------------------
    An earlier version of this module applied ``nn.LayerNorm`` immediately after
    ``input_proj``. That silently destroyed the entire reliability mechanism, and
    the ablation study caught it: the "multiply by MRS" and "no MRS" arms
    returned bit-identical metrics on 4 of 5 seeds.

    The reason is that LayerNorm and RMSNorm are **scale-invariant**. For a
    per-frame scalar r_t > 0,

        Norm(W (r_t f_t) + b)  ==  Norm(W f_t + b/r_t)  ->  Norm(W f_t)  as b -> 0

    so multiplying the ViT features by the reliability score and then normalising
    them puts the signal straight back where it started. Only the projection bias
    survived, which is why one seed differed slightly instead of not at all.

    The same argument applies at the output: ``final_norm`` renormalises every
    timestep to unit RMS, so a subsequent ``mean`` pool weights an unreliable
    frame exactly as much as a reliable one -- the opposite of the intent.

    Two consequences, both implemented here:

      * there is no input LayerNorm; the projection output feeds the blocks
        directly, so the r_t scaling actually reaches the SiLU gate and the
        selective dt/B/C projections, which are *not* scale-invariant;
      * ``pooling="mrs_weighted"`` aggregates as a reliability-weighted mean,
        sum_t r_t h_t / sum_t r_t, which is where the reliability signal can
        express itself without being normalised away.

    Temporal positions are still all present and in order -- no frame is dropped
    (spec section 11). This is a fix to *where* the reliability is applied, not a
    change to what it means.
    """

    def __init__(self, input_dim: int, d_model: int = 256, n_layers: int = 2,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.1, bidirectional: bool = False,
                 pooling: str = "mrs_weighted"):
        super().__init__()
        if pooling not in POOLING_MODES:
            raise ValueError(f"unknown pooling: {pooling!r}; choose from {POOLING_MODES}")
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.pooling = pooling
        self.bidirectional = bool(bidirectional)

        # Explicit projection: the ViT dimension is whatever the backbone reports,
        # so this layer is sized at construction time from a measured value.
        # Deliberately NOT followed by a normalisation layer -- see the class
        # docstring.
        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        self.blocks = nn.ModuleList([
            build_mamba_block(self.d_model, d_state=d_state, d_conv=d_conv,
                              expand=expand, dropout=dropout, bidirectional=bidirectional)
            for _ in range(int(n_layers))
        ])
        self.final_norm = RMSNorm(self.d_model)
        self.pool = AttentionPool(self.d_model) if pooling == "attention" else None

    @property
    def backend(self) -> dict:
        return mamba_backend_info()

    def forward(self, x: torch.Tensor,
                mrs: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, T, input_dim] -> (sequence [B, T, d_model], pooled [B, d_model])."""
        if x.dim() != 3:
            raise ValueError(f"expected [B, T, D], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"TemporalMamba was built for input_dim={self.input_dim} but received "
                f"{x.shape[-1]}. Rebuild the model against the actual ViT dimension.")

        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)

        if self.pooling == "mean":
            pooled = h.mean(dim=1)
        elif self.pooling == "last":
            pooled = h[:, -1]
        elif self.pooling == "attention":
            pooled = self.pool(h)
        else:  # mrs_weighted
            if mrs is None:
                raise ValueError(
                    "pooling='mrs_weighted' needs the per-frame reliability scores; "
                    "pass mrs=... to forward()")
            if mrs.shape != h.shape[:2]:
                raise ValueError(f"mrs shape {tuple(mrs.shape)} does not match "
                                 f"sequence {tuple(h.shape[:2])}")
            w = mrs.unsqueeze(-1)
            # eps guards the degenerate case of a clip whose every frame scored 0.
            pooled = (h * w).sum(dim=1) / (w.sum(dim=1) + 1e-6)
        return h, pooled
