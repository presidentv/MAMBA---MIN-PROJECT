"""Tests for the v2 fine-tuning recipe: augmentation and parameter groups.

The failure these guard against is silence. A broken augmentation that returns
its input, or a layer-decay rule that gives every block the same rate, costs a
multi-hour run and leaves no error behind -- only a slightly different number.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augment import ClipAugment  # noqa: E402
from src.param_groups import build_param_groups  # noqa: E402
from src.utils import load_config  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------- augmentation
def _clip(t=8, s=64):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (t, s, s, 3), dtype=np.uint8)


def test_augment_preserves_shape_and_dtype():
    out = ClipAugment()(_clip())
    assert out.shape == (8, 64, 64, 3) and out.dtype == np.uint8


def test_augment_actually_changes_the_clip():
    np.random.seed(1)
    clip = _clip()
    assert not np.array_equal(ClipAugment()(clip), clip)


def test_augment_erase_box_is_identical_across_frames():
    # A per-frame erase box would look like motion to the temporal encoder.
    np.random.seed(3)
    clip = np.full((6, 32, 32, 3), 100, dtype=np.uint8)
    aug = ClipAugment(scale=(1.0, 1.0), ratio=(1.0, 1.0), brightness=0.0,
                      contrast=0.0, erase_prob=1.0, erase_scale=(0.1, 0.1))
    out = aug(clip)
    erased = out[0] != 100
    for f in range(1, out.shape[0]):
        assert np.array_equal(out[f] != 100, erased)
    assert erased.any()


def test_augment_crop_is_identical_across_frames():
    # Give every frame the same content; the same crop box must give identical
    # output frames.
    np.random.seed(5)
    frame = np.random.default_rng(7).integers(0, 256, (48, 48, 3), dtype=np.uint8)
    clip = np.stack([frame] * 4)
    aug = ClipAugment(brightness=0.0, contrast=0.0, erase_prob=0.0)
    out = aug(clip)
    for f in range(1, 4):
        assert np.array_equal(out[f], out[0])


def test_augment_stays_in_range_under_extreme_photometrics():
    aug = ClipAugment(scale=(1.0, 1.0), ratio=(1.0, 1.0), brightness=0.9,
                      contrast=0.9, erase_prob=0.0)
    for seed in range(5):
        np.random.seed(seed)
        out = aug(_clip())
        assert out.min() >= 0 and out.max() <= 255


def test_augment_disabled_by_config():
    assert ClipAugment.from_config(None) is None
    assert ClipAugment.from_config({"enabled": False}) is None
    assert ClipAugment.from_config({"enabled": True}) is not None


def test_augment_rejects_wrong_shape():
    with pytest.raises(ValueError):
        ClipAugment()(np.zeros((8, 64, 64), dtype=np.uint8))


# ------------------------------------------------------------- param groups
class _FakeBackbone(nn.Module):
    """Minimal stand-in with timm's ViT parameter naming."""

    def __init__(self, n_blocks=4, dim=8):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 5, dim))
        self.patch_embed = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(dim)

    def no_weight_decay(self):
        return {"cls_token", "pos_embed"}


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit_encoder = nn.Module()
        self.vit_encoder.backbone = _FakeBackbone()
        self.learnable_mrs = nn.Module()
        self.learnable_mrs.theta = nn.Parameter(torch.zeros(5))
        self.classifier = nn.Linear(8, 4)


def _groups(**kw):
    kw.setdefault("head_lr", 3e-4)
    kw.setdefault("backbone_lr", 3e-5)
    kw.setdefault("weight_decay", 0.05)
    kw.setdefault("mrs_lr", 0.02)
    return build_param_groups(_FakeModel(), **kw)


def test_layer_decay_lowers_the_rate_towards_the_stem():
    _, s = _groups(layer_decay=0.75)
    bb = [g for g in s["groups"] if g["name"].startswith("backbone")]
    assert len(bb) > 2
    assert max(g["lr"] for g in bb) == pytest.approx(3e-5)      # top layer
    assert min(g["lr"] for g in bb) < 3e-5 * 0.5                # stem, decayed


def test_layer_decay_of_one_is_a_single_backbone_rate():
    _, s = _groups(layer_decay=1.0)
    bb = {g["lr"] for g in s["groups"] if g["name"].startswith("backbone")}
    assert bb == {3e-5}
    assert s["layer_decay"] is None


def test_norms_biases_and_tokens_are_not_decayed():
    _, s = _groups(layer_decay=0.75)
    assert any(g["weight_decay"] == 0.0 for g in s["groups"])
    assert any(g["weight_decay"] == 0.05 for g in s["groups"])
    model = _FakeModel()
    groups, _ = build_param_groups(model, head_lr=3e-4, backbone_lr=3e-5,
                                   weight_decay=0.05, mrs_lr=0.02, layer_decay=0.75)
    decayed = {id(p) for g in groups if g["weight_decay"] > 0 for p in g["params"]}
    bb = model.vit_encoder.backbone
    for p in (bb.cls_token, bb.pos_embed, bb.norm.weight, bb.norm.bias,
              bb.patch_embed.bias, model.classifier.bias):
        assert id(p) not in decayed
    assert id(bb.patch_embed.weight) in decayed


def test_mrs_weights_keep_their_own_rate_and_no_decay():
    _, s = _groups(layer_decay=0.75)
    mrs = [g for g in s["groups"] if g["name"].startswith("mrs")]
    assert len(mrs) == 1
    assert mrs[0]["lr"] == 0.02 and mrs[0]["weight_decay"] == 0.0


def test_every_trainable_parameter_lands_in_exactly_one_group():
    model = _FakeModel()
    groups, _ = build_param_groups(model, head_lr=3e-4, backbone_lr=3e-5,
                                   weight_decay=0.05, mrs_lr=0.02, layer_decay=0.75)
    seen = [id(p) for g in groups for p in g["params"]]
    expected = [id(p) for p in model.parameters() if p.requires_grad]
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen))


def test_frozen_parameters_are_excluded():
    model = _FakeModel()
    model.vit_encoder.backbone.patch_embed.weight.requires_grad = False
    groups, _ = build_param_groups(model, head_lr=3e-4, backbone_lr=3e-5,
                                   weight_decay=0.05, mrs_lr=0.02, layer_decay=0.75)
    seen = {id(p) for g in groups for p in g["params"]}
    assert id(model.vit_encoder.backbone.patch_embed.weight) not in seen


# ------------------------------------------------------------------- config
def test_v2_config_enables_all_three_fixes():
    cfg = load_config(REPO / "configs" / "config_ft32_v2.yaml")
    assert cfg["augmentation"]["enabled"] is True
    assert cfg["vit"]["layer_decay"] == 0.75
    assert cfg["vit"]["drop_path_rate"] > 0
    # and must still be the same data/model as the run it is compared against
    v1 = load_config(REPO / "configs" / "config_ft32.yaml")
    assert cfg["video"]["num_frames"] == v1["video"]["num_frames"]
    assert cfg["mamba"] == v1["mamba"]
    assert cfg["training"]["epochs"] == v1["training"]["epochs"]
