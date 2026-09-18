"""Mamba selective state-space model -- official PyTorch reference path.

WHAT THIS IS
------------
A vendored copy of the **reference (non-fused) implementation** from the official
Mamba repository, https://github.com/state-spaces/mamba (Apache-2.0, Albert Gu &
Tri Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
arXiv:2312.00752). Specifically it mirrors

  * ``mamba_ssm/ops/selective_scan_interface.py::selective_scan_ref``
  * ``mamba_ssm/modules/mamba_simple.py::Mamba``  (its slow path)
  * ``mamba_ssm/ops/triton/layer_norm.py::RMSNorm``
  * ``mamba_ssm/modules/block.py::Block``         (pre-norm residual wrapper)

The S6 recurrence, the input-dependent (selective) Delta/B/C projections, the
depthwise causal conv, the SiLU gate, the A_log/D parameterisation and the
dt_proj initialisation schedule are all reproduced as in the official code.

WHY IT IS VENDORED RATHER THAN pip-INSTALLED
--------------------------------------------
``pip install mamba-ssm`` was attempted on this machine and failed. Verbatim:

    UserWarning: mamba_ssm was requested, but nvcc was not found. Are you sure
    your environment has nvcc available? ...
    torch.__version__ = 2.14.0+cpu
    ...
    File "<string>", line 177, in <module>
    packaging/version.py", line 200, in __init__
        match = self._regex.search(version)
    TypeError: expected string or bytes-like object, got 'NoneType'

PyPI ships ``mamba-ssm`` as a source distribution only; its build step requires
the CUDA Toolkit (``nvcc``), which is not installed here, and Windows is not an
officially supported build target for it.

WHAT THE DIFFERENCE ACTUALLY IS
-------------------------------
The fused CUDA kernel and this reference path compute the *same function*. The
kernel fuses the scan into one pass with recomputation to avoid materialising
the [B, D, L, N] state tensor; the reference materialises it and loops over the
sequence in Python. So:

  * architecture, parameters and outputs: identical (up to float ordering)
  * speed and memory: substantially worse, and it grows with sequence length

That trade-off is acceptable here because the temporal sequence is T frames
(8-64), not thousands of tokens. It is NOT acceptable to describe this as
anything other than what it is, so the model records
``mamba_backend="pytorch_reference"`` in every artifact it writes.

This is a Mamba, not an LSTM or a Transformer standing in for one (spec Rule 6).
If ``mamba_ssm`` is ever importable, ``build_mamba_block`` prefers it
automatically and reports ``mamba_backend="mamba_ssm_cuda"``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

__all__ = ["selective_scan_ref", "Mamba", "RMSNorm", "MambaBlock", "build_mamba_block",
           "mamba_backend_info"]


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                       delta_softplus=False, return_last_state=False):
    """Reference selective scan.

    Shapes (real-valued case, which is what Mamba's default configuration uses):
        u        (batch, dim, seqlen)
        delta    (batch, dim, seqlen)
        A        (dim, dstate)
        B, C     (batch, dstate, seqlen)   -- input-dependent, hence "selective"
        D        (dim,)
        z        (batch, dim, seqlen)      -- gate
        out      (batch, dim, seqlen)

    The recurrence is x_t = exp(delta_t A) x_{t-1} + delta_t B_t u_t,
    y_t = C_t x_t (+ D u_t), gated by SiLU(z).
    """
    dtype_in = u.dtype
    # The official implementation upcasts to float32 so that fp16/bf16 inputs do
    # not lose the state accumulation. Promote rather than hard-cast, so a
    # float64 caller (the equivalence test in scripts/test_mamba.py) keeps its
    # precision and every operand shares one dtype -- mixing a float64 A with a
    # float32 C is a type error that only appears outside the fp32 happy path.
    work = torch.promote_types(torch.promote_types(u.dtype, A.dtype), torch.float32)
    u = u.to(work)
    delta = delta.to(work)
    A = A.to(work)
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].to(work)
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, seqlen = u.shape
    dstate = A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    B = B.to(work)
    C = C.to(work)

    # deltaA: (batch, dim, seqlen, dstate)
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    else:
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)

    x = A.new_zeros((batch, dim, dstate))
    ys = []
    last_state = None
    for i in range(seqlen):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum("bdn,dn->bd", x, C)
        else:
            y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
        if i == seqlen - 1:
            last_state = x
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (batch, dim, seqlen)

    out = y if D is None else y + u * rearrange(D.to(work), "d -> d 1")
    if z is not None:
        out = out * F.silu(z.to(work))
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


class Mamba(nn.Module):
    """A single Mamba block (mixer), reference path.

    Constructor arguments and their defaults follow ``mamba_ssm.modules.
    mamba_simple.Mamba``. ``use_fast_path`` is absent because there is no fast
    path here; that is stated rather than silently ignored.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dt_rank: str | int = "auto", dt_min: float = 0.001, dt_max: float = 0.1,
                 dt_init: str = "random", dt_scale: float = 1.0, dt_init_floor: float = 1e-4,
                 conv_bias: bool = True, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, bias=conv_bias,
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # --- dt_proj initialisation, verbatim from the official implementation ---
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"dt_init={dt_init!r} is not one of 'constant'/'random'")

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus, so that softplus(bias) == dt at initialisation.
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D-real initialisation of A.
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n",
                   d=self.d_inner).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """(batch, seqlen, d_model) -> (batch, seqlen, d_model)"""
        _batch, seqlen, _dim = hidden_states.shape

        xz = self.in_proj(hidden_states)             # (b, l, 2*d_inner)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)                    # each (b, d_inner, l)

        x = self.act(self.conv1d(x)[..., :seqlen])   # causal depthwise conv

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()            # (d_inner, b*l)
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) n -> b n l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) n -> b n l", l=seqlen).contiguous()

        A = -torch.exp(self.A_log.float())           # (d_inner, d_state), strictly negative

        y = selective_scan_ref(
            x, dt, A, B, C, self.D.float(), z=z,
            delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
        )
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)


class RMSNorm(nn.Module):
    """Root-mean-square layer norm, as used throughout the Mamba stack."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(dtype)


class MambaBlock(nn.Module):
    """Pre-norm residual wrapper: ``x + Mamba(RMSNorm(x))``.

    ``bidirectional=True`` additionally runs a second, independently
    parameterised Mamba over the reversed sequence and sums the two -- the
    bi-directional scan idea from Vision Mamba (Vim, arXiv:2401.09417). It is
    off by default; the unidirectional block is the spec's baseline, and the
    bidirectional variant is reported only as a labelled ablation.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = Mamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bidirectional = bidirectional
        self.mixer_rev = (Mamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                          if bidirectional else None)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        out = self.mixer(h)
        if self.mixer_rev is not None:
            out = out + torch.flip(self.mixer_rev(torch.flip(h, dims=[1])), dims=[1])
        return x + self.dropout(out)


def mamba_backend_info() -> dict:
    """Report which Mamba implementation is actually in use. Written into every
    results artifact so no reader has to guess."""
    try:
        import mamba_ssm  # noqa: F401

        return {
            "backend": "mamba_ssm_cuda",
            "version": getattr(mamba_ssm, "__version__", "unknown"),
            "source": "official state-spaces/mamba package (fused CUDA kernel)",
        }
    except Exception as exc:
        return {
            "backend": "pytorch_reference",
            "version": None,
            "source": "vendored reference implementation from state-spaces/mamba "
                      "(selective_scan_ref + mamba_simple.Mamba slow path)",
            "mamba_ssm_import_error": f"{type(exc).__name__}: {exc}",
        }


def build_mamba_block(d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                      dropout: float = 0.0, bidirectional: bool = False) -> nn.Module:
    """Prefer the official CUDA package when it is importable; otherwise use the
    vendored reference path. Either way the block is a Mamba."""
    if not bidirectional:
        try:
            from mamba_ssm import Mamba as MambaCuda  # type: ignore

            class _CudaBlock(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.norm = RMSNorm(d_model)
                    self.mixer = MambaCuda(d_model=d_model, d_state=d_state,
                                           d_conv=d_conv, expand=expand)
                    self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

                def forward(self, x):
                    return x + self.dropout(self.mixer(self.norm(x)))

            return _CudaBlock()
        except Exception:
            pass
    return MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                      dropout=dropout, bidirectional=bidirectional)
