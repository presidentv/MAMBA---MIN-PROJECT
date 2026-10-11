"""Tests for the prior that keeps the learned MRS weights from collapsing.

The full-scale run ended with motion_consistency at 0.990 and the other four
components near 0.001: the softmax had collapsed and a five-component
reliability score had quietly become a one-component one. Nothing failed, no
error was raised, and the number that the project's central claim rests on was
wrong.

w = softmax(theta) and theta = 0 is exactly the uniform 0.2 weighting, so
decay on theta is a prior towards uniform. These tests pin the three
behaviours that matter: it bounds a runaway, it still lets a real signal
through, and it does not invent structure from noise.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import LearnableMRSWeights  # noqa: E402
from src.param_groups import build_param_groups  # noqa: E402
from src.utils import load_config  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# Steps the full run takes: 5,357 train clips / effective batch 32, x 40 epochs.
FULL_RUN_STEPS = 6700


def drive(lr: float, wd: float, steps: int = FULL_RUN_STEPS,
          pressure: float = 1.0, noise: float = 0.0, seed: int = 0):
    """Optimise theta under gradient pressure onto one component."""
    gen = torch.Generator().manual_seed(seed)
    mrs = LearnableMRSWeights(5)
    opt = torch.optim.AdamW(mrs.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        w = mrs.weights()
        loss = -pressure * w[4]
        if noise:
            loss = loss + noise * (torch.randn(5, generator=gen) * w).sum()
        loss.backward()
        opt.step()
    return mrs.weights().detach()


def test_without_the_prior_the_weights_collapse():
    # Reproduces the failure: this is what the full run did.
    w = drive(lr=0.04, wd=0.0, steps=2000)
    assert float(w.max()) > 0.95


def test_lowering_the_learning_rate_alone_does_not_prevent_collapse():
    # The tempting fix, and the reason the real fix is decay rather than lr.
    w = drive(lr=0.002, wd=0.0, steps=2000)
    assert float(w.max()) > 0.95


def test_the_prior_bounds_a_runaway():
    w = drive(lr=0.01, wd=1.0)
    assert float(w.max()) < 0.70
    assert float(w.min()) > 0.05


def test_the_bound_is_set_by_decay_not_by_learning_rate():
    slow = drive(lr=0.01, wd=1.0)
    fast = drive(lr=0.02, wd=1.0)
    assert float(slow.max()) == pytest.approx(float(fast.max()), abs=0.05)


def test_stronger_decay_gives_a_tighter_bound():
    assert float(drive(lr=0.01, wd=2.0).max()) < float(drive(lr=0.01, wd=1.0).max())


def test_a_real_signal_still_moves_the_weights_off_uniform():
    # A prior that pinned every weight at 0.2 would remove the contribution
    # the project claims: that the weights are learned, not asserted.
    w = drive(lr=0.01, wd=1.0, pressure=0.15, noise=0.3, steps=3000)
    assert float(w.max()) > 0.25
    assert int(w.argmax()) == 4


def test_noise_alone_leaves_the_weights_at_uniform():
    w = drive(lr=0.01, wd=1.0, pressure=0.0, noise=0.3, steps=3000)
    assert float(w.max()) < 0.24 and float(w.min()) > 0.16


def test_weights_always_sum_to_one():
    for wd in (0.0, 1.0):
        assert float(drive(lr=0.02, wd=wd, steps=200).sum()) == pytest.approx(1.0)


# ------------------------------------------------------------- wiring
class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.learnable_mrs = LearnableMRSWeights(5)
        self.classifier = torch.nn.Linear(4, 4)


def test_mrs_weight_decay_reaches_the_optimiser_group():
    _, s = build_param_groups(_Tiny(), head_lr=3e-4, backbone_lr=1e-5,
                              weight_decay=0.05, mrs_lr=0.01, mrs_weight_decay=1.0)
    mrs = [g for g in s["groups"] if g["name"].startswith("mrs")]
    assert len(mrs) == 1 and mrs[0]["weight_decay"] == 1.0 and mrs[0]["lr"] == 0.01


def test_mrs_weight_decay_defaults_to_off_for_older_configs():
    _, s = build_param_groups(_Tiny(), head_lr=3e-4, backbone_lr=1e-5,
                              weight_decay=0.05, mrs_lr=0.01)
    mrs = [g for g in s["groups"] if g["name"].startswith("mrs")]
    assert mrs[0]["weight_decay"] == 0.0


def test_full_config_sets_the_prior():
    t = load_config(REPO / "configs" / "config_full.yaml")["training"]
    assert t["mrs_weight_decay"] >= 1.0, "the long run must carry the prior"
