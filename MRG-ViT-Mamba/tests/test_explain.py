"""Worked-example explanations for the fine-tuned model (src/explain.py).

A tiny, randomly initialised timm ViT stands in for ViT-B/16 so these run on
CPU in seconds with nothing to download; the code path is the same.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.explain import (  # noqa: E402
    ViTGradCAM, choose_examples, explain_clip, integrated_gradients, pick_frame,
)
from src.landmarks import CLIP_FEATURE_DIM  # noqa: E402
from src.model import MRGViTMamba  # noqa: E402
from src.utils import load_config  # noqa: E402
from src.vit_encoder import ViTFrameEncoder  # noqa: E402

TINY_VIT = "timm:vit_tiny_patch16_224"


def _pred(cid, subj, true, pred, conf):
    probs = [(1 - conf) / 3] * 4
    probs[pred] = conf
    return {"clip_id": cid, "subject_id": subj, "true": true, "pred": pred, "probs": probs}


# ------------------------------------------------------------------ selection
def test_choose_examples_mixes_right_wrong_and_rare_classes():
    preds = [
        _pred("a", "s1", 2, 2, .95), _pred("b", "s1", 3, 2, .90),   # same person twice
        _pred("c", "s2", 3, 3, .80), _pred("d", "s3", 3, 2, .85),
        _pred("e", "s4", 0, 0, .50), _pred("f", "s5", 1, 3, .40),
    ]
    chosen = choose_examples(preds, 3)
    assert [p["clip_id"] for p in chosen] == ["a", "d", "e"]
    assert len({p["subject_id"] for p in chosen}) == 3
    assert chosen[0]["true"] == chosen[0]["pred"]          # confident and right
    assert chosen[1]["true"] != chosen[1]["pred"]          # confident and wrong
    assert chosen[2]["true"] == 0                          # rarest class not yet shown


def test_choose_examples_without_mistakes_or_enough_people():
    preds = [_pred("a", "s1", 2, 2, .9), _pred("b", "s2", 3, 3, .8), _pred("c", "s1", 0, 0, .7)]
    chosen = choose_examples(preds, 3)
    assert [p["clip_id"] for p in chosen] == ["a", "b"]    # only two different people


def test_pick_frame_prefers_the_heaviest_then_the_middle():
    assert pick_frame([.1, .5, .2]) == 1
    assert pick_frame([.25, .25, .25, .25]) in (1, 2)
    assert pick_frame(np.full(32, 1 / 32)) in (15, 16)


# ------------------------------------------------------ integrated gradients
def test_integrated_gradients_is_complete():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.GELU(), torch.nn.Linear(16, 1))
    f = lambda z: net(z)[:, 0]                              # noqa: E731
    x, base = torch.randn(6), torch.randn(6)
    attr, err = integrated_gradients(f, x, base, steps=128)
    assert attr.shape == (6,)
    with torch.no_grad():
        gap = float(f(x[None]) - f(base[None]))
    assert abs(float(attr.sum()) - gap) == pytest.approx(abs(err), abs=1e-6)
    assert abs(err) < 1e-3 * max(1.0, abs(gap))


def test_integrated_gradients_on_linear_is_exact():
    w = torch.tensor([1.0, -2.0, 3.0])
    attr, err = integrated_gradients(lambda z: z @ w, torch.tensor([1.0, 1.0, 1.0]),
                                     torch.zeros(3), steps=4)
    assert torch.allclose(attr, w) and abs(err) < 1e-6


# ------------------------------------------------------------------ Grad-CAM
@pytest.mark.parametrize("chunk", [4, 3])        # one chunk, and a split into 3 + 1
def test_gradcam_shape_and_range(chunk):
    enc = ViTFrameEncoder(TINY_VIT, pretrained=False, freeze=True, chunk_size=chunk)
    enc.frozen = False
    cam = ViTGradCAM(enc.backbone)
    frames = torch.randn(1, 4, 3, 224, 224, requires_grad=True)
    enc(frames).sum().backward()
    maps = cam.maps()
    cam.close()
    assert maps.shape == (4, 14, 14)
    assert maps.min() >= 0 and maps.max() <= 1 + 1e-6


# -------------------------------------------------------------- whole clip
def test_explain_clip_on_a_real_model():
    torch.manual_seed(0)
    cfg = load_config(REPO / "configs" / "config_full.yaml")
    enc = ViTFrameEncoder(TINY_VIT, pretrained=False, freeze=True, chunk_size=32)
    model = MRGViTMamba(vit_dim=enc.spec.embed_dim, landmark_dim=CLIP_FEATURE_DIM,
                        cfg=cfg, vit_encoder=enc)
    model.landmark_branch.norm.fit(np.random.randn(20, CLIP_FEATURE_DIM))
    T = 6
    item = {"frames": torch.randn(T, 3, 224, 224), "mrs": torch.rand(T),
            "mrs_components": torch.rand(T, 5),
            "landmark_features": torch.randn(CLIP_FEATURE_DIM), "label": torch.tensor(2)}

    ex = explain_clip(model, item, torch.device("cpu"), ig_steps=128)

    assert 0 <= ex["pred"] < 4 and ex["true"] == 2
    assert ex["probs"].sum() == pytest.approx(1.0, abs=1e-5)
    assert ex["gradcam"].shape == (T, 14, 14)
    assert ex["pool_weights"].sum() == pytest.approx(1.0, abs=1e-4)
    # reliability is recomputed from the components with the learned weights
    w = model.learnable_mrs.weights().detach().numpy()
    assert np.allclose(ex["mrs"], np.clip(item["mrs_components"].numpy() @ w, 0, 1), atol=1e-5)
    dec = ex["logit_decomposition"]
    total = dec["zero_input_logit"] + dec["deep_branch"] + dec["landmark_branch"]
    assert total == pytest.approx(dec["predicted_logit"], abs=5e-3)
    la = ex["landmark_attribution"]
    assert la["attribution"].shape == (CLIP_FEATURE_DIM,)
    assert abs(la["completeness_error"]) < 5e-3
    assert "eye_openness" in la["group_attribution"]
    # the encoder is left as it was found, and no gradients are left behind
    assert enc.frozen is True
    assert all(p.grad is None for p in model.parameters())
