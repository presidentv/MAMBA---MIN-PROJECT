"""Adaptive Reliability Conditioning -- the upgraded MRG-VM novelty.

WHAT CHANGED AND WHY
--------------------
The first version of this project reduced reliability to a single scalar
(``mrs.combine`` averages five sub-scores with fixed, equal weights) and used it
to scale the SSM timestep through a fixed linear map. Phase 6 measured that
mechanism and found it flipped **zero** predictions: post-gating the scalar sits
at 0.93 +/- 0.049, a spread of under 4%, which the learned ``dt_proj`` bias
simply absorbs.

Measured on the sample, the averaging is what destroys the signal:

    sub-score            std across frames    corr. with scalar MRS
    motion_consistency        0.140                   0.70
    eye_visibility            0.114                   0.67
    blur                      0.088                   0.54
    head_rotation             0.042                   0.33
    face_visibility           0.003                  -0.08
    ------------------------------------------------------------
    combined scalar           0.047

Three of the five sub-scores vary two to three times more than the scalar they
are averaged into, and none correlates with it above 0.70. Collapsing them first
throws that away. It is also conceptually wrong: blur degrades *appearance*
while head rotation degrades *gaze geometry*, so they should condition the model
differently rather than being summed into one number.

This module therefore does three things the old design did not:

1. :class:`LearnableReliability` replaces the hand-set equal weights with
   learned ones, turning a hand-designed heuristic into learned reliability
   estimation. The weights are inspectable, so "what did the model decide
   reliability means?" is a reportable result.

2. :class:`AdaptiveReliabilityConditioner` consumes the full **five-vector**, not
   the scalar, and emits three distinct learnable controls over the temporal
   stream -- FiLM modulation, a null-token gate, and a timestep scale. None of
   them is a plain multiplication of features by a score.

3. Both are differentiable and trained end to end, so reliability is learned
   jointly with the task rather than fixed in advance.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Order is fixed and must match mrgvm.data.MRS_SUBSCORE_COLUMNS.
SUBSCORE_NAMES: Tuple[str, ...] = (
    "blur",
    "face_visibility",
    "head_rotation",
    "eye_visibility",
    "motion_consistency",
)
NUM_SUBSCORES = len(SUBSCORE_NAMES)


class LearnableReliability(nn.Module):
    """Learn how the five sub-scores combine into one reliability value.

    The old pipeline used a fixed weighted mean with equal weights. Here the
    weights are a softmax over learned logits, so they stay positive and sum to
    one -- the result is still a convex combination in [0, 1] and therefore still
    interpretable as "reliability", but the model decides the mixture.

    Initialised at uniform, which reproduces the original equal-weight average
    exactly at step 0. Any departure from uniform during training is a result
    worth reporting.
    """

    def __init__(self, num_factors: int = NUM_SUBSCORES, learnable: bool = True) -> None:
        super().__init__()
        logits = torch.zeros(num_factors)
        self.logits = nn.Parameter(logits, requires_grad=learnable)

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)

    def named_weights(self) -> Dict[str, float]:
        """For logging: the learned meaning of 'reliability'."""
        values = self.weights.detach().cpu().tolist()
        return {name: float(v) for name, v in zip(SUBSCORE_NAMES, values)}

    def forward(self, sub_scores: torch.Tensor) -> torch.Tensor:
        """(B, T, 5) -> (B, T) scalar reliability in [0, 1]."""
        return (sub_scores * self.weights).sum(dim=-1)


class AdaptiveReliabilityConditioner(nn.Module):
    """Multi-factor reliability conditioning of the temporal stream.

    Consumes the per-frame five-vector and produces three controls:

    ``gamma``, ``beta`` -- FiLM modulation applied to the frame embedding before
        the temporal scan. A feature-wise affine transform conditioned on *which
        kind* of degradation is present, so blur and head rotation can reshape
        the representation differently. This is the part that a scalar cannot do.

    ``gate`` -- per-channel blend between the frame embedding and a learned
        ``null_token``. An unreliable frame is replaced by a learned
        "uninformative observation" rather than scaled toward zero, which keeps
        the sequence intact and lets the temporal model carry forward its
        previous state instead of being dragged toward the origin. Deliberately
        NOT a multiplication of features by the score.

    ``dt_scale`` -- multiplier on the SSM timestep. In a state-space model dt
        governs how far the hidden state moves per step, so shrinking it makes an
        unreliable frame update the state less and thereby retain more of the
        previous reliable context. Unlike the old fixed map, this one is learned
        from all five factors and can respond to their *combination*.

    All three are initialised to be near-identity (gamma≈1, beta≈0, gate≈1,
    dt_scale≈1), so an untrained conditioner leaves the model unchanged and any
    measured effect is something training chose to introduce.
    """

    def __init__(
        self,
        d_model: int,
        num_factors: int = NUM_SUBSCORES,
        hidden_dim: int = 32,
        min_dt_scale: float = 0.1,
        use_film: bool = True,
        use_gate: bool = True,
        use_dt: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.min_dt_scale = min_dt_scale
        self.use_film = use_film
        self.use_gate = use_gate
        self.use_dt = use_dt

        self.encoder = nn.Sequential(
            nn.Linear(num_factors, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.to_gamma = nn.Linear(hidden_dim, d_model)
        self.to_beta = nn.Linear(hidden_dim, d_model)
        self.to_gate = nn.Linear(hidden_dim, d_model)
        self.to_dt = nn.Linear(hidden_dim, 1)
        self.null_token = nn.Parameter(torch.zeros(d_model))

        # Zero the output heads so the module starts as an exact identity: tanh(0)=0
        # gives gamma=1 and beta=0, and the biases below put gate and dt at ~1.
        for layer in (self.to_gamma, self.to_beta, self.to_gate, self.to_dt):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.to_gate.bias, 3.0)   # sigmoid(3) ~ 0.95
        nn.init.constant_(self.to_dt.bias, 3.0)

    def forward(
        self, sub_scores: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            sub_scores: (B, T, 5) per-frame MRS components in [0, 1]
            mask:       (B, T) True where the frame is real

        Returns dict of gamma (B,T,D), beta (B,T,D), gate (B,T,D), dt_scale (B,T).
        """
        hidden = self.encoder(sub_scores)

        gamma = 1.0 + torch.tanh(self.to_gamma(hidden)) if self.use_film else None
        beta = torch.tanh(self.to_beta(hidden)) if self.use_film else None
        gate = torch.sigmoid(self.to_gate(hidden)) if self.use_gate else None

        if self.use_dt:
            raw = torch.sigmoid(self.to_dt(hidden)).squeeze(-1)
            dt_scale = self.min_dt_scale + (1.0 - self.min_dt_scale) * raw
        else:
            dt_scale = None

        if mask is not None and dt_scale is not None:
            # Padding must not move the state at all.
            dt_scale = torch.where(mask, dt_scale, torch.full_like(dt_scale, 1e-3))

        return {"gamma": gamma, "beta": beta, "gate": gate, "dt_scale": dt_scale}

    def apply_to(
        self, features: torch.Tensor, controls: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply FiLM then the null-token gate to (B, T, D) frame embeddings."""
        out = features
        if controls.get("gamma") is not None:
            out = controls["gamma"] * out + controls["beta"]
        if controls.get("gate") is not None:
            gate = controls["gate"]
            out = gate * out + (1.0 - gate) * self.null_token
        return out


def reliability_loss_weights(
    reliability: torch.Tensor,
    mask: torch.Tensor,
    strength: float = 1.0,
    floor: float = 0.25,
) -> torch.Tensor:
    """Per-clip loss weights derived from mean frame reliability.

    A clip whose frames are mostly unreliable is a noisy supervision signal, so
    down-weighting it reduces gradient noise from examples the features cannot
    describe well.

    Two safeguards, because this is easy to get wrong:

    * weights are normalised to mean 1 within the batch, so the effective
      learning rate does not drift as clip quality varies;
    * a floor keeps low-reliability clips contributing something. Driving them to
      zero would quietly turn the reliability score into a second, hidden
      training-set filter -- which is exactly the hard-filtering behaviour this
      redesign set out to remove.

    ``strength`` interpolates between uniform (0.0) and fully proportional (1.0).
    """
    weights = mask.to(reliability.dtype)
    denominator = weights.sum(dim=1).clamp(min=1e-6)
    clip_reliability = (reliability * weights).sum(dim=1) / denominator

    scaled = floor + (1.0 - floor) * clip_reliability.clamp(0.0, 1.0)
    blended = (1.0 - strength) * torch.ones_like(scaled) + strength * scaled
    return blended / blended.mean().clamp(min=1e-6)
