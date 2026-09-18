"""Per-branch normalisation, projection, and fusion of the two branches.

Spec section 18 is explicit that normalisation and projection are *different
operations*: a linear layer y = Wx + b changes dimensionality and
representation, it does not control scale or distribution. So each branch is
normalised first, then projected, and the two steps are separate modules.

The landmark branch additionally uses a scaler whose mean and variance are fitted
on the **training split only** (spec Rule 8). The fitted statistics are stored in
the module's buffers so they travel with the checkpoint and cannot drift.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class StandardScaler(nn.Module):
    """Feature standardisation with train-fitted statistics, stored as buffers.

    Kept as a module (rather than a sklearn object applied in the dataset) so
    that the statistics are saved and loaded with the model checkpoint, which
    makes it impossible to evaluate with one set of statistics and train with
    another.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("scale", torch.ones(dim))
        self.register_buffer("fitted", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def fit(self, x: np.ndarray | torch.Tensor, eps: float = 1e-6) -> "StandardScaler":
        arr = torch.as_tensor(np.asarray(x, dtype=np.float32))
        if arr.dim() != 2:
            raise ValueError(f"expected [N, D] to fit, got {tuple(arr.shape)}")
        self.mean.copy_(arr.mean(dim=0))
        std = arr.std(dim=0)
        # Constant columns get scale 1 rather than exploding; they carry no
        # information and must not become numerically dominant.
        std = torch.where(std < eps, torch.ones_like(std), std)
        self.scale.copy_(std)
        self.fitted.fill_(True)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted.item()):
            raise RuntimeError(
                "StandardScaler used before being fitted. Fit it on the training "
                "split with src.train.fit_landmark_scaler() before any forward pass."
            )
        return (x - self.mean) / self.scale


class BranchProjection(nn.Module):
    """normalise -> linear projection -> activation -> dropout."""

    def __init__(self, input_dim: int, output_dim: int, norm: str = "layernorm",
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        if norm == "layernorm":
            self.norm: nn.Module = nn.LayerNorm(input_dim)
        elif norm == "standard":
            self.norm = StandardScaler(input_dim)
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"unknown norm: {norm!r}")
        self.proj = nn.Linear(input_dim, output_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"branch expects {self.input_dim}-d input, got {x.shape[-1]}")
        return self.dropout(self.act(self.proj(self.norm(x))))


class FeatureFusion(nn.Module):
    """Combine the projected deep and landmark branches.

    ``concat`` is the primary implementation (spec section 19): F = [D_p || L_p],
    giving 2 * projected_dim.

    ``gated`` implements the adaptive alternative with its formula stated
    explicitly rather than gestured at:

        a = sigmoid(W [D_p || L_p] + b),   a in R^{projected_dim}
        F = [a * D_p  ||  (1 - a) * L_p]

    so the gate reallocates per-dimension emphasis between the branches while the
    output width stays the same, making the two modes directly comparable. No
    claim is made that gating is better; scripts/run_ablation.py measures it.
    """

    def __init__(self, projected_dim: int, mode: str = "concat",
                 use_landmark_branch: bool = True):
        super().__init__()
        if mode not in ("concat", "gated"):
            raise ValueError(f"unknown fusion mode: {mode!r}")
        self.mode = mode
        self.projected_dim = int(projected_dim)
        self.use_landmark_branch = bool(use_landmark_branch)
        self.output_dim = self.projected_dim * (2 if use_landmark_branch else 1)
        self.gate = (nn.Linear(2 * projected_dim, projected_dim)
                     if (mode == "gated" and use_landmark_branch) else None)

    def forward(self, deep: torch.Tensor, landmark: torch.Tensor | None) -> torch.Tensor:
        if not self.use_landmark_branch or landmark is None:
            return deep
        if self.mode == "concat":
            fused = torch.cat([deep, landmark], dim=-1)
        else:
            alpha = torch.sigmoid(self.gate(torch.cat([deep, landmark], dim=-1)))
            fused = torch.cat([alpha * deep, (1.0 - alpha) * landmark], dim=-1)
        if fused.shape[-1] != self.output_dim:
            raise RuntimeError(
                f"fusion produced {fused.shape[-1]}-d output, expected {self.output_dim}")
        return fused

    def extra_repr(self) -> str:
        return (f"mode={self.mode}, projected_dim={self.projected_dim}, "
                f"output_dim={self.output_dim}, landmark_branch={self.use_landmark_branch}")
