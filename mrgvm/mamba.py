"""Pure-PyTorch Mamba (S6) selective state-space blocks.

WHY THIS IS HAND-WRITTEN
------------------------
The reference ``mamba-ssm`` package ships a fused CUDA kernel and refuses to
build without ``nvcc``; this machine is CPU-only, so it cannot be installed.
The selective-scan recurrence below is the same algorithm as Gu & Dao (2023),
just expressed in portable PyTorch instead of a fused kernel. It is slower --
the scan is a Python loop over the sequence axis rather than a hardware
parallel scan -- but numerically it computes the same thing, it trains, and it
runs anywhere. On the sequence lengths this project uses (49 spatial patches,
50 temporal frames) the loop is short and the batch axis supplies the
parallelism, so the cost is acceptable.

The discretised recurrence, per channel:

    h_t = exp(dt_t * A) . h_{t-1} + (dt_t * B_t) * x_t
    y_t = <C_t, h_t> + D * x_t

"Selective" means dt, B and C are produced from the input at every timestep
rather than being fixed, which is what lets the model choose what to remember.

Memory note: ``exp(dt * A)`` is (batch, length, d_inner, d_state). Materialising
it for the whole sequence at once is hundreds of MB even at this scale, so it is
recomputed per timestep inside the scan and never held for the full sequence.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm, as used throughout the Mamba paper."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(norm + self.eps)


def selective_scan(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Run the S6 recurrence.

    Args:
        x:     (batch, length, d_inner)  input sequence
        delta: (batch, length, d_inner)  positive timestep, already softplus'd
        A:     (d_inner, d_state)        state matrix (negative, real)
        B:     (batch, length, d_state)  input-dependent input matrix
        C:     (batch, length, d_state)  input-dependent output matrix
        D:     (d_inner,)                skip connection

    Returns:
        (batch, length, d_inner)
    """
    batch, length, d_inner = x.shape
    d_state = A.shape[-1]

    state = x.new_zeros((batch, d_inner, d_state))
    outputs = []
    for t in range(length):
        # Zero-order hold discretisation of the continuous-time system.
        delta_t = delta[:, t].unsqueeze(-1)                 # (b, d_inner, 1)
        decay = torch.exp(delta_t * A.unsqueeze(0))         # (b, d_inner, d_state)
        input_term = delta_t * B[:, t].unsqueeze(1)         # (b, 1->d_inner, d_state)
        state = decay * state + input_term * x[:, t].unsqueeze(-1)
        outputs.append(torch.einsum("bds,bs->bd", state, C[:, t]))

    y = torch.stack(outputs, dim=1)                         # (b, length, d_inner)
    return y + x * D


class MambaBlock(nn.Module):
    """One directional Mamba (S6) block: gated selective SSM with a local conv."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank or max(1, math.ceil(d_model / 16))

        self.in_projection = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )
        # Produces the input-dependent (dt, B, C) -- the "selective" part.
        self.x_projection = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_projection = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is stored as log(-A) so that A stays strictly negative (stable) and
        # is initialised to the S4D-real spectrum 1..d_state.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_projection = nn.Linear(self.d_inner, d_model, bias=False)

        self._init_dt_bias()

    def _init_dt_bias(self, dt_min: float = 0.001, dt_max: float = 0.1) -> None:
        """Initialise dt so softplus(bias) lands in [dt_min, dt_max], as in the paper."""
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_projection.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(
        self, x: torch.Tensor, delta_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, length, d_model)
            delta_scale: optional (batch, length) multiplier on the SSM timestep.
                This is the hook the MRG-VM reliability guidance uses -- see
                vision_mamba.ReliabilityGuidedTemporalMamba.
        """
        batch, length, _ = x.shape

        projected = self.in_projection(x)                       # (b, l, 2*d_inner)
        hidden, gate = projected.chunk(2, dim=-1)

        # Depthwise causal conv over the sequence axis.
        hidden = hidden.transpose(1, 2)                         # (b, d_inner, l)
        hidden = self.conv1d(hidden)[:, :, :length]
        hidden = hidden.transpose(1, 2)
        hidden = F.silu(hidden)

        parameters = self.x_projection(hidden)
        delta, B, C = torch.split(
            parameters, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_projection(delta))           # (b, l, d_inner)
        if delta_scale is not None:
            # Shrinking dt shrinks both the state decay and the input term, so a
            # low-reliability timestep moves the state less. See module note in
            # vision_mamba.py for why this is the right place to inject MRS.
            delta = delta * delta_scale.unsqueeze(-1).clamp(min=1e-4)

        A = -torch.exp(self.A_log)
        y = selective_scan(hidden, delta, A, B, C, self.D)
        y = y * F.silu(gate)
        return self.out_projection(y)


class BidirectionalMambaBlock(nn.Module):
    """Vision Mamba (Vim) block: forward + backward scans over the same sequence.

    Plain Mamba is causal, which is the right inductive bias for language but the
    wrong one for a patch grid or a short clip, where position t+1 is as
    informative as t-1. Vim's fix -- scanning both directions and merging -- is
    what makes the SSM usable as a vision backbone, so it is used everywhere in
    this project rather than the unidirectional block.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        merge: str = "sum",
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.forward_block = MambaBlock(d_model, d_state, d_conv, expand)
        self.backward_block = MambaBlock(d_model, d_state, d_conv, expand)
        self.merge = merge
        self.merge_projection = (
            nn.Linear(2 * d_model, d_model, bias=False) if merge == "concat" else None
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, delta_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        normed = self.norm(x)

        forward_out = self.forward_block(normed, delta_scale)

        flipped = torch.flip(normed, dims=[1])
        flipped_scale = None if delta_scale is None else torch.flip(delta_scale, dims=[1])
        backward_out = torch.flip(self.backward_block(flipped, flipped_scale), dims=[1])

        if self.merge_projection is not None:
            merged = self.merge_projection(torch.cat([forward_out, backward_out], dim=-1))
        else:
            merged = forward_out + backward_out
        return residual + self.dropout(merged)


class MambaEncoder(nn.Module):
    """A stack of bidirectional Mamba blocks with a final norm."""

    def __init__(
        self,
        d_model: int,
        depth: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                BidirectionalMambaBlock(d_model, d_state, d_conv, expand, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = RMSNorm(d_model)

    def forward(
        self, x: torch.Tensor, delta_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, delta_scale)
        return self.norm(x)
