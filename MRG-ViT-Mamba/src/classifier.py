"""MLP engagement classifier over the fused representation.

Spec section 20: 512 -> 256 -> act -> dropout -> 128 -> act -> 4 raw logits.
The widths come from config so the "512" is whatever the fusion layer actually
produced, not a hard-coded number that silently disagrees with it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 4,
                 hidden_dims: tuple[int, ...] = (256, 128),
                 activation: str = "gelu", dropout: float = 0.3):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; choose from {list(_ACTIVATIONS)}")
        act_cls = _ACTIVATIONS[activation]

        layers: list[nn.Module] = []
        prev = int(input_dim)
        for width in hidden_dims:
            layers += [nn.Linear(prev, int(width)), act_cls(), nn.Dropout(dropout)]
            prev = int(width)
        # Final layer produces raw logits; softmax lives in the loss, not here.
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(prev, int(num_classes))
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"classifier expects {self.input_dim}-d input, got {x.shape[-1]}")
        return self.head(self.features(x))
