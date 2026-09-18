"""Unit tests for the parts where a silent error would be hardest to notice.

These are deliberately about *invariants* rather than snapshots: temporal
ordering, value ranges, cache-key sensitivity, and the leakage rules. A snapshot
test would pass while the pipeline quietly did the wrong thing.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fusion import FeatureFusion, StandardScaler  # noqa: E402
from src.landmarks import (  # noqa: E402
    CLIP_FEATURE_DIM, FEATURE_DIM, aggregate_clip_features,
)
from src.losses import class_weights_from_counts  # noqa: E402
from src.mamba_ref import Mamba, selective_scan_ref  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.mrs import COMPONENTS, MRSCalibration, head_pose_score, mrs_from_arrays  # noqa: E402
from src.preprocess import pack_crops, stage1_key, unpack_crops  # noqa: E402
from src.temporal_mamba import MRSWeighting, TemporalMamba  # noqa: E402
from src.utils import load_config  # noqa: E402
from src.video_sampling import uniform_indices  # noqa: E402


# --------------------------------------------------------------------------- #
# Temporal sampling
# --------------------------------------------------------------------------- #
def test_uniform_indices_matches_the_spec_formula():
    n, t = 300, 16
    got = uniform_indices(n, t)
    want = [int(round(k * (n - 1) / (t - 1))) for k in range(t)]
    assert got == want
    assert got[0] == 0 and got[-1] == n - 1


@pytest.mark.parametrize("t", [1, 2, 8, 16, 32, 64])
def test_uniform_indices_length_and_bounds(t):
    idx = uniform_indices(300, t)
    assert len(idx) == t
    assert all(0 <= i < 300 for i in idx)
    assert idx == sorted(idx), "sampling must preserve temporal order"


def test_uniform_indices_handles_fewer_frames_than_requested():
    # Shape must stay [T] even when the clip is shorter than T.
    idx = uniform_indices(5, 16)
    assert len(idx) == 16
    assert max(idx) == 4


def test_uniform_indices_rejects_nonsense():
    with pytest.raises(ValueError):
        uniform_indices(0, 16)
    with pytest.raises(ValueError):
        uniform_indices(10, 0)


# --------------------------------------------------------------------------- #
# MRS
# --------------------------------------------------------------------------- #
def test_calibration_is_monotonic_and_bounded():
    cal = MRSCalibration.fit(np.linspace(1, 500, 400), np.linspace(0.01, 0.3, 400))
    scores = [cal.blur_score(v) for v in np.linspace(1, 500, 50)]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores), "sharper frames must never score lower"
    assert cal.blur_score(-5.0) == pytest.approx(0.0)


def test_calibration_rejects_degenerate_input():
    with pytest.raises(ValueError):
        MRSCalibration.fit(np.ones(100), np.ones(100))   # constant blur
    with pytest.raises(ValueError):
        MRSCalibration.fit(np.linspace(1, 10, 5), np.linspace(1, 10, 5))  # too few


def test_head_pose_reliability_falls_with_rotation_and_ignores_roll():
    cfg = {"head_pose_full_reliability_deg": 15.0, "head_pose_zero_reliability_deg": 60.0}
    assert head_pose_score((0, 0, 0), cfg) == 1.0
    assert head_pose_score((10, -5, 0), cfg) == 1.0
    assert head_pose_score((90, 0, 0), cfg) == 0.0
    assert 0.0 < head_pose_score((40, 0, 0), cfg) < 1.0
    # Roll is corrected by alignment, so it must not affect reliability.
    assert head_pose_score((0, 0, 80), cfg) == head_pose_score((0, 0, 0), cfg)
    # Unknown pose is neither trusted nor discarded.
    assert head_pose_score(None, cfg) == 0.5
    assert head_pose_score((np.nan, np.nan, np.nan), cfg) == 0.5


def test_mrs_stays_in_unit_interval_for_extreme_inputs():
    cfg = load_config()
    cal = MRSCalibration.uncalibrated()
    t = 16
    rng = np.random.default_rng(0)
    mrs = mrs_from_arrays(
        blur_raw=rng.uniform(0, 5000, t),
        face_area_fraction=rng.uniform(0, 1, t),
        detector_confidence=np.where(rng.random(t) > 0.5, rng.random(t), np.nan),
        head_pose_deg=rng.uniform(-180, 180, (t, 3)),
        eye_visibility=rng.random(t),
        motion_consistency=rng.random(t),
        cfg_mrs=dict(cfg["mrs"]), calibration=cal)
    assert mrs.shape == (t,)
    assert np.all(np.isfinite(mrs))
    assert mrs.min() >= 0.0 and mrs.max() <= 1.0


def test_mrs_weights_are_the_five_documented_components():
    cfg = load_config()
    assert set(cfg["mrs"]["weights"]) == set(COMPONENTS)


# --------------------------------------------------------------------------- #
# MRS weighting -- the core mechanism
# --------------------------------------------------------------------------- #
def test_mrs_weighting_preserves_shape_and_position():
    b, t, d = 3, 16, 64
    feats = torch.randn(b, t, d)
    mrs = torch.rand(b, t)
    out = MRSWeighting("multiply")(feats, mrs)
    assert out.shape == feats.shape
    for i in range(b):
        for j in range(t):
            assert torch.allclose(out[i, j], feats[i, j] * mrs[i, j], atol=1e-6)


def test_mrs_weighting_none_is_identity():
    feats = torch.randn(2, 8, 32)
    assert torch.equal(MRSWeighting("none")(feats, torch.rand(2, 8)), feats)


def test_mrs_weighting_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        MRSWeighting("multiply")(torch.randn(2, 8, 32), torch.rand(2, 7))


def test_reliability_weighted_pooling_actually_depends_on_mrs():
    """Regression test for the bug the ablation exposed.

    Feature multiplication alone is cancelled by scale-invariant normalisation,
    so this asserts the pooled output really does move when reliability changes.
    """
    torch.manual_seed(0)
    model = TemporalMamba(input_dim=32, d_model=16, n_layers=1, dropout=0.0,
                          pooling="mrs_weighted").eval()
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        a = model(x, torch.ones(2, 8))[1]
        b = model(x, torch.rand(2, 8))[1]
    assert not torch.allclose(a, b, atol=1e-5), \
        "reliability-weighted pooling ignored the MRS vector"


def test_temporal_mamba_has_no_scale_removing_input_norm():
    """The input LayerNorm is what silently cancelled the whole mechanism."""
    model = TemporalMamba(input_dim=32, d_model=16, n_layers=1, pooling="mean")
    assert not hasattr(model, "input_norm"), \
        "an input normalisation layer here makes MRS feature weighting a no-op"


# --------------------------------------------------------------------------- #
# Mamba
# --------------------------------------------------------------------------- #
def test_selective_scan_is_causal():
    torch.manual_seed(0)
    model = TemporalMamba(input_dim=16, d_model=16, n_layers=1, dropout=0.0,
                          bidirectional=False, pooling="mean").eval()
    x = torch.randn(2, 10, 16)
    y = x.clone()
    y[:, 5:] += 5.0
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(y)
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-6), "future input leaked backwards"
    assert not torch.allclose(a[:, 5:], b[:, 5:], atol=1e-3)


def test_mamba_output_shape_and_finiteness():
    m = Mamba(d_model=32, d_state=8).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 12, 32))
    assert out.shape == (2, 12, 32)
    assert torch.isfinite(out).all()


def test_selective_scan_handles_float64_without_dtype_error():
    """Regression test: A was left un-cast while u/B/C were forced to float32."""
    b, d, l, n = 1, 3, 5, 2
    out = selective_scan_ref(
        torch.randn(b, d, l, dtype=torch.float64),
        torch.rand(b, d, l, dtype=torch.float64) + 0.1,
        -torch.rand(d, n, dtype=torch.float64) - 0.1,
        torch.randn(b, n, l, dtype=torch.float64),
        torch.randn(b, n, l, dtype=torch.float64),
        torch.randn(d, dtype=torch.float64))
    assert out.dtype == torch.float64
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Landmarks / fusion / classifier
# --------------------------------------------------------------------------- #
def test_clip_aggregation_dimension_is_three_times_L():
    per_frame = np.random.rand(16, FEATURE_DIM).astype(np.float32)
    out = aggregate_clip_features(per_frame, np.ones(16, dtype=bool))
    assert out.shape == (CLIP_FEATURE_DIM,) == (3 * FEATURE_DIM,)
    assert np.isfinite(out).all()


def test_clip_aggregation_ignores_frames_without_a_face_for_mean_and_std():
    per_frame = np.zeros((8, FEATURE_DIM), dtype=np.float32)
    per_frame[:4] = 1.0                       # detected frames
    mask = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    out = aggregate_clip_features(per_frame, mask)
    assert np.allclose(out[:FEATURE_DIM], 1.0), "zeroed missing frames dragged the mean down"


def test_clip_aggregation_survives_all_frames_missing():
    out = aggregate_clip_features(np.zeros((8, FEATURE_DIM), np.float32),
                                  np.zeros(8, dtype=bool))
    assert out.shape == (CLIP_FEATURE_DIM,)
    assert np.isfinite(out).all()


def test_scaler_refuses_to_run_before_being_fitted():
    scaler = StandardScaler(10)
    with pytest.raises(RuntimeError, match="before being fitted"):
        scaler(torch.randn(2, 10))


def test_scaler_handles_constant_columns_without_exploding():
    scaler = StandardScaler(3)
    data = np.stack([np.array([1.0, 5.0, 2.0]) for _ in range(20)])
    data[:, 0] = np.linspace(0, 1, 20)
    scaler.fit(data)
    out = scaler(torch.tensor(data, dtype=torch.float32))
    assert torch.isfinite(out).all()


def test_fusion_widths():
    f = FeatureFusion(256, mode="concat")
    assert f.output_dim == 512
    assert f(torch.randn(4, 256), torch.randn(4, 256)).shape == (4, 512)
    g = FeatureFusion(256, mode="gated")
    assert g(torch.randn(4, 256), torch.randn(4, 256)).shape == (4, 512)
    solo = FeatureFusion(256, mode="concat", use_landmark_branch=False)
    assert solo.output_dim == 256


# --------------------------------------------------------------------------- #
# Metrics and class weights
# --------------------------------------------------------------------------- #
def test_metrics_match_hand_computed_values():
    y_true = [0, 0, 1, 1, 2, 2, 3, 3]
    y_pred = [0, 1, 1, 1, 2, 2, 3, 0]
    m = compute_metrics(y_true, y_pred, 4)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["per_class"]["Low"]["recall"] == pytest.approx(1.0)
    assert m["per_class"]["High"]["f1"] == pytest.approx(1.0)
    assert m["num_samples"] == 8


def test_metrics_report_absent_and_never_predicted_classes():
    m = compute_metrics([0, 0, 2, 2], [0, 0, 2, 2], 4)
    assert m["classes_absent_from_split"] == [1, 3]
    assert m["macro_f1"] < m["accuracy"], "macro-F1 must be dragged down by absent classes"

    m2 = compute_metrics([0, 1, 1, 1], [0, 0, 0, 0], 4)
    assert 1 in m2["classes_present_but_never_predicted"]


def test_class_weights_come_from_training_counts_only():
    w = class_weights_from_counts({0: 10, 1: 90}, 2, "inverse_frequency")
    assert w[0] > w[1], "the rarer class must get the larger weight"
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-5)


def test_absent_class_gets_zero_weight_not_infinity():
    w = class_weights_from_counts({0: 5, 2: 5}, 4, "inverse_frequency")
    assert torch.isfinite(w).all()
    assert w[1] == 0.0 and w[3] == 0.0


# --------------------------------------------------------------------------- #
# Cache integrity
# --------------------------------------------------------------------------- #
def test_cache_key_changes_when_preprocessing_changes():
    cfg = load_config()
    base = stage1_key(cfg)

    cfg2 = load_config()
    cfg2["video"]["num_frames"] = 32
    assert stage1_key(cfg2) != base, "frame count must invalidate the cache"

    cfg3 = load_config()
    cfg3["face"]["crop_padding"] = 0.5
    assert stage1_key(cfg3) != base, "crop settings must invalidate the cache"

    cfg4 = load_config()
    assert stage1_key(cfg4, "blur_s2") != base, "a corruption must get its own cache"


def test_cache_key_ignores_mrs_weights_because_raw_signals_are_cached():
    cfg = load_config()
    base = stage1_key(cfg)
    cfg2 = load_config()
    cfg2["mrs"]["weights"]["blur"] = 0.9
    assert stage1_key(cfg2) == base, \
        "MRS weights are applied at load time, so they must not invalidate Stage 1"


def test_crop_packing_round_trips():
    # Smooth, structured content, because that is what a face crop is. Uniform
    # random noise is the pathological case for JPEG (no spatial correlation) and
    # would only measure the codec, not the packing.
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    base = (128 + 60 * np.sin(xx / 9.0) + 40 * np.cos(yy / 7.0))
    crops = np.stack([
        np.clip(np.stack([base + 20 * i, base, base - 20 * i], axis=-1), 0, 255).astype(np.uint8)
        for i in range(4)
    ])

    data, offsets = pack_crops(crops)
    out = unpack_crops(data, offsets)

    assert out.shape == crops.shape
    assert out.dtype == np.uint8
    assert len(offsets) == len(crops) + 1
    assert offsets[0] == 0 and offsets[-1] == data.size
    # Quality 95 on smooth content is near-lossless; a packing/offset bug would
    # scramble frames and blow this up by an order of magnitude.
    assert np.abs(out.astype(int) - crops.astype(int)).mean() < 3.0


def test_crop_packing_keeps_frames_in_order():
    """An offset bug would round-trip cleanly per frame but permute the sequence."""
    crops = np.stack([np.full((32, 32, 3), v, dtype=np.uint8) for v in (10, 90, 170, 250)])
    out = unpack_crops(*pack_crops(crops))
    recovered = [int(round(float(f.mean()))) for f in out]
    assert recovered == pytest.approx([10, 90, 170, 250], abs=2)


# --------------------------------------------------------------------------- #
# Leakage rules
# --------------------------------------------------------------------------- #
def test_official_split_is_subject_disjoint():
    from src.dataset_index import build_index, subject_overlaps

    cfg = load_config()
    index = build_index(cfg)
    if not any(index.clips.values()):
        pytest.skip("dataset not present")
    overlaps = subject_overlaps(index)
    assert overlaps["train_val"] == []
    assert overlaps["train_test"] == []
    assert overlaps["val_test"] == []


def test_engagement_labels_are_within_the_documented_domain():
    from src.dataset_index import build_index

    cfg = load_config()
    index = build_index(cfg)
    if not any(index.clips.values()):
        pytest.skip("dataset not present")
    values = {c.engagement for c in index.all_clips()}
    assert values <= {0, 1, 2, 3}, f"unexpected engagement codes: {values}"
