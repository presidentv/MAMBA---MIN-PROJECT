"""Step 4 -- temporal transformer encoders, cross-attention fusion, ordinal head.

Architecture (all dimensions config-driven via :class:`config.ModelConfig`):

    gaze   -> Linear -> +PositionalEncoding -> TransformerEncoder (2 layers, d=128)
    affect -> Linear -> +PositionalEncoding -> TransformerEncoder (2 layers, d=128)
                                   |
                        CrossAttentionFusion (each modality attends to the other)
                                   |
                          masked mean pool -> ordinal head

Ordinal loss: CORAL (Cao, Mirjalili & Raschka, 2020).
The head emits K-1 = 3 logits that share a single weight vector and differ only
in their bias, which forces the cumulative probabilities
P(y > 0) >= P(y > 1) >= P(y > 2) to be monotone by construction. A cumulative-link
(ordinal logistic) model has the same intent but must learn its thresholds
freely, and on ~36 training clips those thresholds routinely invert and produce
nonsense. CORAL cannot invert, which is why it is the default here. Plain
categorical cross-entropy remains available via ``ModelConfig.loss = 'ce'`` for
ablation -- it is the honest control for "does ordinality actually help".
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over real timesteps only. ``mask`` is True where the step is valid."""
    weights = mask.unsqueeze(-1).to(x.dtype)
    total = weights.sum(dim=1).clamp(min=1.0)
    return (x * weights).sum(dim=1) / total


class TemporalEncoder(nn.Module):
    """Project one modality to d_model and run a small transformer over time."""

    def __init__(self, input_dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, cfg.d_model)
        self.input_norm = nn.LayerNorm(cfg.d_model)
        self.positional = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            # Pre-norm: markedly more stable than post-norm when training a
            # transformer on only a few dozen sequences.
            norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first and would only
        # warn and disable itself; setting it explicitly keeps the log clean.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.num_layers, enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.input_norm(self.input_projection(x))
        h = self.positional(h)
        # nn.Transformer expects True == "ignore this position".
        return self.encoder(h, src_key_padding_mask=~mask)


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention: each modality queries the other.

    Kept symmetric so neither branch is privileged, then concatenated. The
    per-head attention weights are returned because the XAI stage
    (``explain.py``) needs them for attention rollout.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gaze_to_affect = nn.MultiheadAttention(
            cfg.d_model, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.affect_to_gaze = nn.MultiheadAttention(
            cfg.d_model, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm_gaze = nn.LayerNorm(cfg.d_model)
        self.norm_affect = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self, gaze: torch.Tensor, affect: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        pad = ~mask
        attended_gaze, weights_ga = self.gaze_to_affect(
            gaze, affect, affect, key_padding_mask=pad, need_weights=True, average_attn_weights=True
        )
        attended_affect, weights_ag = self.affect_to_gaze(
            affect, gaze, gaze, key_padding_mask=pad, need_weights=True, average_attn_weights=True
        )
        fused_gaze = self.norm_gaze(gaze + self.dropout(attended_gaze))
        fused_affect = self.norm_affect(affect + self.dropout(attended_affect))
        return fused_gaze, fused_affect, {"gaze_to_affect": weights_ga, "affect_to_gaze": weights_ag}


# --------------------------------------------------------------------------- #
# Ordinal head + loss (CORAL)
# --------------------------------------------------------------------------- #
class CoralHead(nn.Module):
    """K-1 cumulative logits sharing one weight vector, with independent biases."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.shared = nn.Linear(in_features, 1, bias=False)
        # Biases initialised in decreasing order so the cumulative probabilities
        # start out already monotone.
        self.bias = nn.Parameter(torch.linspace(1.0, -1.0, num_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared(x) + self.bias  # (B, K-1)


def coral_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """``levels[i, k] = 1`` iff ``label_i > k``; the CORAL binary target matrix."""
    thresholds = torch.arange(num_classes - 1, device=labels.device).unsqueeze(0)
    return (labels.unsqueeze(1) > thresholds).float()


def coral_loss(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted binary cross-entropy over the K-1 cumulative tasks."""
    targets = coral_targets(labels, num_classes)
    per_task = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if weights is not None:
        per_task = per_task * weights[labels].unsqueeze(1)
    return per_task.sum(dim=1).mean()


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    """Predicted class = number of cumulative thresholds passed."""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1).long()


def coral_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert cumulative logits into a proper K-class distribution.

    P(y=0)   = 1 - P(y>0)
    P(y=k)   = P(y>k-1) - P(y>k)
    P(y=K-1) = P(y>K-2)

    Cumulative probabilities are made monotone with a running minimum first --
    CORAL's shared weight makes inversions impossible at convergence, but they
    can still occur early in training, and a negative "probability" would break
    the late-fusion baseline that averages these distributions.
    """
    cumulative = torch.sigmoid(logits)
    cumulative, _ = torch.cummin(cumulative, dim=1)
    ones = torch.ones(cumulative.size(0), 1, device=cumulative.device, dtype=cumulative.dtype)
    zeros = torch.zeros_like(ones)
    upper = torch.cat([ones, cumulative], dim=1)
    lower = torch.cat([cumulative, zeros], dim=1)
    return (upper - lower).clamp(min=1e-8)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class SingleModalityModel(nn.Module):
    """Baselines 3 and 4: one temporal encoder, no fusion."""

    def __init__(self, input_dim: int, cfg: ModelConfig, num_classes: int = 4) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.encoder = TemporalEncoder(input_dim, cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        out_features = num_classes - 1 if cfg.loss == "coral" else num_classes
        self.head = (
            CoralHead(cfg.d_model, num_classes)
            if cfg.loss == "coral"
            else nn.Linear(cfg.d_model, out_features)
        )

    def forward(self, gaze: torch.Tensor, affect: torch.Tensor, mask: torch.Tensor,
                modality: str = "gaze") -> Dict[str, torch.Tensor]:
        x = gaze if modality == "gaze" else affect
        encoded = self.encoder(x, mask)
        pooled = self.dropout(masked_mean(encoded, mask))
        return {"logits": self.head(pooled), "pooled": pooled, "attention": {}}


class CrossAttentionFusionModel(nn.Module):
    """The full two-encoder + cross-attention fusion model."""

    def __init__(self, gaze_dim: int, affect_dim: int, cfg: ModelConfig, num_classes: int = 4) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.gaze_encoder = TemporalEncoder(gaze_dim, cfg)
        self.affect_encoder = TemporalEncoder(affect_dim, cfg)
        self.fusion = CrossAttentionFusion(cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        out_features = num_classes - 1 if cfg.loss == "coral" else num_classes
        self.head = (
            CoralHead(cfg.d_model * 2, num_classes)
            if cfg.loss == "coral"
            else nn.Linear(cfg.d_model * 2, out_features)
        )

    def forward(self, gaze: torch.Tensor, affect: torch.Tensor,
                mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded_gaze = self.gaze_encoder(gaze, mask)
        encoded_affect = self.affect_encoder(affect, mask)
        fused_gaze, fused_affect, attention = self.fusion(encoded_gaze, encoded_affect, mask)
        pooled = torch.cat(
            [masked_mean(fused_gaze, mask), masked_mean(fused_affect, mask)], dim=-1
        )
        pooled = self.dropout(pooled)
        return {"logits": self.head(pooled), "pooled": pooled, "attention": attention}


# --------------------------------------------------------------------------- #
# Shared loss / decode helpers
# --------------------------------------------------------------------------- #
def compute_loss(
    logits: torch.Tensor, labels: torch.Tensor, cfg: ModelConfig,
    num_classes: int = 4, class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if cfg.loss == "coral":
        return coral_loss(logits, labels, num_classes, class_weights)
    return F.cross_entropy(logits, labels, weight=class_weights)


def decode_predictions(logits: torch.Tensor, cfg: ModelConfig) -> torch.Tensor:
    if cfg.loss == "coral":
        return coral_predict(logits)
    return logits.argmax(dim=1)


def decode_probabilities(logits: torch.Tensor, cfg: ModelConfig) -> torch.Tensor:
    if cfg.loss == "coral":
        return coral_probabilities(logits)
    return F.softmax(logits, dim=1)


def build_model(name: str, gaze_dim: int, affect_dim: int, cfg: ModelConfig,
                num_classes: int = 4) -> nn.Module:
    """Factory used by both train.py and baselines.py."""
    if name == "cross_attention_fusion":
        return CrossAttentionFusionModel(gaze_dim, affect_dim, cfg, num_classes)
    if name == "gaze_only":
        return SingleModalityModel(gaze_dim, cfg, num_classes)
    if name == "affect_only":
        return SingleModalityModel(affect_dim, cfg, num_classes)
    raise ValueError(f"Unknown neural model {name!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
