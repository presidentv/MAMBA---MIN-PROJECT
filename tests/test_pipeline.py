"""Tests for the properties the project actually claims.

The claims worth testing are the ones a reader would otherwise have to take on
trust: that the MRS components behave monotonically, that alignment is
invertible, that CORAL's cumulative probabilities are a valid distribution, that
the Mamba scan is genuinely bidirectional and reliability-conditioned, and that
subject leakage across splits is impossible rather than merely unlikely.

Run:  .venv/Scripts/python.exe -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "pipeline", ROOT / "src"):
    sys.path.insert(0, str(extra))

from engagement_pipeline import alignment, mrs, pose  # noqa: E402


# --------------------------------------------------------------------------- #
# Motion Reliability Score
# --------------------------------------------------------------------------- #
def test_blur_score_is_monotone_in_sharpness():
    """A blurrier image must never score higher than a sharper one."""
    import cv2

    rng = np.random.RandomState(0)
    sharp = rng.randint(0, 255, (120, 120, 3), dtype=np.uint8)
    scores = [
        mrs.blur_score(cv2.GaussianBlur(sharp, (0, 0), sigma), 150.0) if sigma else
        mrs.blur_score(sharp, 150.0)
        for sigma in (0, 1.0, 2.0, 4.0)
    ]
    assert scores == sorted(scores, reverse=True), scores


def test_all_subscores_are_bounded():
    """Every component must land in [0, 1] -- the combiner assumes it."""
    assert mrs.face_visibility_score(None) == 0.0
    assert mrs.face_visibility_score(2.0) == 1.0
    assert mrs.face_visibility_score(-1.0) == 0.0
    assert mrs.head_rotation_score(0.0, 60.0) == 1.0
    assert mrs.head_rotation_score(120.0, 60.0) == 0.0
    assert mrs.head_rotation_score(float("nan"), 60.0) == 0.0


def test_head_rotation_score_decreases_with_deviation():
    values = [mrs.head_rotation_score(d, 60.0) for d in (0, 15, 30, 45, 60)]
    assert values == sorted(values, reverse=True)


def test_combine_respects_weights():
    components = mrs.MRSComponents(
        blur=1.0, face_visibility=0.0, head_rotation=0.0,
        eye_visibility=0.0, motion_consistency=0.0,
    )
    equal = {name: 1.0 for name in mrs.COMPONENT_NAMES}
    assert mrs.combine(components, equal) == pytest.approx(0.2)

    blur_only = {name: 0.0 for name in mrs.COMPONENT_NAMES}
    blur_only["blur"] = 1.0
    assert mrs.combine(components, blur_only) == pytest.approx(1.0)


def test_missing_face_scores_zero():
    """No detection must produce zero reliability, never a default-high value."""
    assert mrs.eye_visibility_score(None, None, None, (100, 100)) == 0.0


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def test_alignment_places_eyes_at_canonical_positions():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = alignment.align_face(
        image, right_eye=(300, 240), left_eye=(360, 250),
        output_size=224, desired_left_eye_x=0.35, desired_eye_y=0.38,
    )
    moved = alignment.apply_affine(
        np.array([[300, 240], [360, 250]], dtype=float), result.affine
    )
    assert moved[0][0] == pytest.approx(0.35 * 224, abs=1.0)
    assert moved[1][0] == pytest.approx(0.65 * 224, abs=1.0)
    # Both eyes land on the same horizontal line -- that is what alignment means.
    assert moved[0][1] == pytest.approx(moved[1][1], abs=1.0)
    assert moved[0][1] == pytest.approx(0.38 * 224, abs=1.0)


def test_affine_round_trips():
    """Phase 2 back-projects landmarks through this inverse; it must be exact."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = alignment.align_face(image, (280, 200), (340, 214))
    points = np.array([[100.0, 150.0], [400.0, 300.0], [12.5, 470.0]])
    forward = alignment.apply_affine(points, result.affine)
    back = alignment.apply_affine(forward, alignment.invert_affine(result.affine))
    np.testing.assert_allclose(back, points, atol=1e-6)


def test_affine_dict_round_trips():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = alignment.align_face(image, (280, 200), (340, 214))
    restored = alignment.dict_to_affine(alignment.affine_to_dict(result.affine))
    np.testing.assert_allclose(restored, result.affine, atol=1e-12)


def test_alignment_removes_roll():
    """A tilted eye line must come out level, which is why roll is added back."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    tilted = alignment.align_face(image, (300, 200), (360, 260))
    assert tilted.rotation_deg == pytest.approx(45.0, abs=0.5)


# --------------------------------------------------------------------------- #
# Head pose
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "angles", [(20.0, 0.0, 0.0), (0.0, 20.0, 0.0), (0.0, 0.0, 20.0), (10.0, -15.0, 5.0)]
)
def test_euler_decomposition_round_trips(angles):
    yaw, pitch, roll = np.radians(angles)
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    recovered = pose.rotation_matrix_to_euler(rz @ ry @ rx)
    np.testing.assert_allclose(recovered, angles, atol=1e-6)


# --------------------------------------------------------------------------- #
# CORAL ordinal head
# --------------------------------------------------------------------------- #
def test_coral_probabilities_form_a_distribution():
    from models import coral_probabilities

    logits = torch.randn(32, 3) * 3.0
    probabilities = coral_probabilities(logits)
    assert probabilities.shape == (32, 4)
    assert (probabilities > 0).all(), "negative probability"
    np.testing.assert_allclose(
        probabilities.sum(dim=1).detach().numpy(), np.ones(32), atol=1e-5
    )


def test_coral_predictions_track_threshold_count():
    from models import coral_predict

    # All thresholds passed -> class 3; none passed -> class 0.
    assert coral_predict(torch.tensor([[9.0, 9.0, 9.0]])).item() == 3
    assert coral_predict(torch.tensor([[-9.0, -9.0, -9.0]])).item() == 0
    assert coral_predict(torch.tensor([[9.0, 9.0, -9.0]])).item() == 2


def test_coral_targets_are_cumulative():
    from models import coral_targets

    targets = coral_targets(torch.tensor([0, 1, 2, 3]), 4)
    expected = torch.tensor([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
    torch.testing.assert_close(targets, expected)


# --------------------------------------------------------------------------- #
# Mamba
# --------------------------------------------------------------------------- #
def test_mamba_block_is_bidirectional():
    """Perturbing a late timestep must change an early one, or the backward
    scan is not running -- which would silently reduce Vim to plain Mamba."""
    from mrgvm.mamba import BidirectionalMambaBlock

    torch.manual_seed(0)
    block = BidirectionalMambaBlock(32, d_state=8).eval()
    a = torch.randn(1, 20, 32)
    b = a.clone()
    b[0, 15] += 5.0
    with torch.no_grad():
        difference = (block(a) - block(b)).abs()[0]
    assert difference[2].sum().item() > 1e-6


def test_selective_scan_gradients_are_finite():
    from mrgvm.mamba import MambaEncoder

    encoder = MambaEncoder(32, depth=1, d_state=8)
    x = torch.randn(2, 16, 32, requires_grad=True)
    encoder(x).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_delta_scale_changes_the_output():
    """If dt modulation had no effect the reliability guidance would be inert."""
    from mrgvm.mamba import MambaEncoder

    torch.manual_seed(0)
    encoder = MambaEncoder(32, depth=1, d_state=8).eval()
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        base = encoder(x)
        damped = encoder(x, delta_scale=torch.full((2, 16), 0.1))
    assert not torch.allclose(base, damped)


# --------------------------------------------------------------------------- #
# Reliability conditioning
# --------------------------------------------------------------------------- #
def test_learnable_reliability_starts_as_the_equal_weight_mean():
    """v2 must reduce exactly to v1's averaging at initialisation."""
    from mrgvm.reliability import LearnableReliability

    module = LearnableReliability(5)
    sub_scores = torch.rand(3, 10, 5)
    torch.testing.assert_close(module(sub_scores), sub_scores.mean(dim=-1))


def test_learnable_reliability_weights_are_a_simplex():
    from mrgvm.reliability import LearnableReliability

    module = LearnableReliability(5)
    with torch.no_grad():
        module.logits.copy_(torch.tensor([2.0, -1.0, 0.5, 0.0, 1.0]))
    weights = module.weights
    assert weights.sum().item() == pytest.approx(1.0)
    assert (weights > 0).all()


def test_conditioner_is_near_identity_at_init():
    """Any measured effect must come from training, not from initialisation."""
    from mrgvm.reliability import AdaptiveReliabilityConditioner

    conditioner = AdaptiveReliabilityConditioner(16).eval()
    features = torch.randn(2, 8, 16)
    with torch.no_grad():
        controls = conditioner(torch.rand(2, 8, 5), torch.ones(2, 8, dtype=torch.bool))
        out = conditioner.apply_to(features, controls)
    torch.testing.assert_close(controls["gamma"], torch.ones_like(controls["gamma"]))
    torch.testing.assert_close(controls["beta"], torch.zeros_like(controls["beta"]))
    assert (out - features).abs().mean().item() < 0.1


def test_conditioner_responds_to_its_input():
    from mrgvm.reliability import AdaptiveReliabilityConditioner

    torch.manual_seed(0)
    conditioner = AdaptiveReliabilityConditioner(16)
    for layer in (conditioner.to_gamma, conditioner.to_gate, conditioner.to_dt):
        torch.nn.init.normal_(layer.weight, std=0.5)
    with torch.no_grad():
        good = conditioner(torch.full((1, 4, 5), 0.95), torch.ones(1, 4, dtype=torch.bool))
        bad = conditioner(torch.full((1, 4, 5), 0.10), torch.ones(1, 4, dtype=torch.bool))
    assert not torch.allclose(good["dt_scale"], bad["dt_scale"])


def test_padding_is_frozen_out_of_the_state():
    from mrgvm.reliability import AdaptiveReliabilityConditioner

    conditioner = AdaptiveReliabilityConditioner(16)
    mask = torch.tensor([[True, True, False, False]])
    controls = conditioner(torch.rand(1, 4, 5), mask)
    assert controls["dt_scale"][0, 2].item() < 1e-2
    assert controls["dt_scale"][0, 3].item() < 1e-2


def test_reliability_loss_weights_are_mean_one_and_floored():
    """Weights must not drift the effective learning rate, and must never reach
    zero -- that would turn the score into a second hidden training filter."""
    from mrgvm.reliability import reliability_loss_weights

    reliability = torch.tensor([[0.9, 0.9, 0.9], [0.05, 0.05, 0.05]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    weights = reliability_loss_weights(reliability, mask, strength=1.0, floor=0.25)
    assert weights.mean().item() == pytest.approx(1.0, abs=1e-5)
    assert (weights > 0).all()
    assert weights[0] > weights[1]


# --------------------------------------------------------------------------- #
# Split integrity
# --------------------------------------------------------------------------- #
def test_subject_leakage_raises():
    """Leakage must be impossible to reach by accident, not merely warned about."""
    from mrgvm.data import MRGVMClip, verify_split_integrity

    def clip(subject, split):
        return MRGVMClip(
            clip_id=f"{subject}_{split}", subject_id=subject, split=split,
            frames=np.zeros((1, 4, 4, 3), np.uint8), mrs=np.ones(1, np.float32),
            sub_scores=np.ones((1, 5), np.float32), geometric=np.zeros((1, 3), np.float32),
            labels={"Engagement": 1},
        )

    verify_split_integrity([clip("A", "Train"), clip("B", "Test")])
    with pytest.raises(ValueError, match="leakage"):
        verify_split_integrity([clip("A", "Train"), clip("A", "Test")])


# --------------------------------------------------------------------------- #
# Corruption protocol
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["gaussian_blur", "motion_blur", "occlusion",
                                  "darkness", "jpeg", "sensor_noise"])
def test_corruptions_preserve_shape_and_dtype(name):
    from mrgvm.robustness import corrupt_frames

    frames = np.random.RandomState(0).randint(0, 255, (4, 32, 32, 3), dtype=np.uint8)
    out = corrupt_frames(frames, name, severity=3)
    assert out.shape == frames.shape
    assert out.dtype == np.uint8


def test_corruption_severity_is_monotone_for_blur():
    """Severity must actually mean severity, or the degradation curve is noise."""
    from mrgvm.robustness import corrupt_frames

    frames = np.random.RandomState(0).randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
    sharpness = [
        mrs.laplacian_variance(corrupt_frames(frames, "gaussian_blur", s)[0])
        for s in (1, 3, 5)
    ]
    assert sharpness == sorted(sharpness, reverse=True), sharpness
