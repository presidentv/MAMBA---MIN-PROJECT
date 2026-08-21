"""Typed, serialisable configuration for both pipeline phases.

Every tunable lives here as a dataclass field with a default. A JSON config file
may override any subset of the fields, and CLI flags override the file, so the
precedence is:  CLI  >  --config file  >  dataclass default.

JSON is used rather than YAML so that the pipeline needs no dependency beyond
opencv-python / mediapipe / numpy / pandas / pyarrow.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


# --------------------------------------------------------------------------- #
# Phase 1
# --------------------------------------------------------------------------- #
@dataclass
class MRSWeights:
    """Relative weights of the five Motion Reliability Score components.

    They do not need to sum to 1 -- the combiner normalises by their sum -- so a
    user can express "blur matters twice as much" as simply blur=2.0.
    """

    blur: float = 1.0
    face_visibility: float = 1.0
    head_rotation: float = 1.0
    eye_visibility: float = 1.0
    motion_consistency: float = 1.0

    def as_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)


@dataclass
class Phase1Config:
    # --- sampling -------------------------------------------------------- #
    sample_fps: float = 5.0
    # Target sampling rate in frames/second. DAiSEE clips are ~10 s at 30 fps,
    # so the default yields ~50 frames per clip.

    max_frames_per_clip: int = 0
    # 0 = no cap. Useful for smoke tests on the full dataset.

    # --- detection ------------------------------------------------------- #
    min_detection_confidence: float = 0.4
    # Deliberately lower than the MRS face-visibility expectation so that
    # low-confidence frames are *scored and logged* rather than silently
    # vanishing before the MRS is computed.

    use_mesh_for_pose: bool = True
    # DAiSEE frames often contain bystanders and the subject's face is small
    # (~115 px in a 640x480 frame), so head pose fitted to only the six BlazeFace
    # keypoints carries a large yaw bias. With this on, Phase 1 additionally runs
    # the 478-point mesh to obtain pose from MediaPipe's own transformation
    # matrix and true eye openness. Measured cost: 1.7 ms/frame detector-only vs
    # 10.2 ms/frame with the mesh (~1.3 h for full DAiSEE on one worker).
    # Turn off to trade pose accuracy for ~6x faster Phase 1.

    max_subject_jump_ratio: float = 1.5
    # A frame's chosen face must lie within this multiple of the running median
    # face width from the previous accepted face centre. Prevents the "primary
    # face" flipping to a bystander mid-clip.

    # --- alignment / normalisation --------------------------------------- #
    output_size: int = 224
    desired_left_eye_x: float = 0.35
    # Horizontal position of the subject's left eye in the aligned crop, as a
    # fraction of width. The right eye lands at (1 - this), which fixes scale.
    desired_eye_y: float = 0.38
    # Vertical position of the eye line in the aligned crop.
    image_format: str = "png"
    # 'png' (lossless, best for downstream landmarking) or 'jpg' (~6x smaller;
    # recommended when scaling to the full 9k-clip DAiSEE).
    jpeg_quality: int = 95

    save_normalized_npy: bool = False
    # Also persist the float32 mean/std-normalised tensor per frame. Off by
    # default: it multiplies storage ~10x and is trivially recomputable from the
    # stored image via alignment.normalize_image().
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # --- MRS sub-score calibration --------------------------------------- #
    # The two constants below were calibrated on 240 sampled frames spread over
    # 12 clips from all three splits of the sample (see README, "MRS calibration").
    blur_var_reference: float = 150.0
    # Variance-of-Laplacian (on the NATIVE-resolution face ROI) at which the blur
    # sub-score saturates to 1.0. Observed distribution: p10=75, p25=110,
    # p50=150, p90=424. Anchoring at the median means the term does not reward
    # extra sharpness, it only penalises the blurred tail -- which is what a
    # reliability gate should do.
    max_head_deviation_deg: float = 60.0
    # L2 norm of (yaw, pitch, roll) at which the head-rotation sub-score hits 0.
    max_flow_magnitude_px: float = 8.0
    # Median optical-flow magnitude between consecutive *aligned* crops at which
    # the motion-consistency sub-score hits 0. Observed natural range on good
    # frames: p50=1.5, p90=3.0, p95=3.6 px. The zero-point sits at ~2.2x the p95
    # so ordinary micro-motion is barely penalised while genuine detector jitter
    # (which lands far outside this range) is.
    eye_bbox_margin: float = 0.25
    # Fractional expansion of the detection box for eye-keypoint containment.

    weights: MRSWeights = field(default_factory=MRSWeights)

    # --- retention ------------------------------------------------------- #
    mrs_threshold: float = 0.5
    low_retention_warn_ratio: float = 0.30
    # Clips retaining less than this fraction of sampled frames are written to
    # the warnings log (0.30 retained == more than 70% discarded).

    # --- runtime --------------------------------------------------------- #
    workers: int = 1
    splits: Tuple[str, ...] = ("Train", "Validation", "Test")
    video_extensions: Tuple[str, ...] = (".avi", ".mp4", ".mov", ".mkv", ".webm")
    limit_clips: int = 0
    # 0 = all. Process only the first N discovered clips (smoke tests).
    overwrite: bool = True


# --------------------------------------------------------------------------- #
# Phase 2
# --------------------------------------------------------------------------- #
@dataclass
class Phase2Config:
    min_face_detection_confidence: float = 0.3
    min_face_presence_confidence: float = 0.3
    min_tracking_confidence: float = 0.3
    # Looser than Phase 1 because Phase 2 runs on frames that already passed the
    # MRS gate; a miss here is a real failure, not noise.

    output_blendshapes: bool = False
    # 52 ARKit blendshape scores. Off by default -- the affect branch uses a
    # dedicated FER model -- but useful as an interpretable extra feature set.

    track_break_threshold: float = 0.08
    # Mean per-landmark L2 displacement (normalised crop units) between
    # consecutive retained frames above which the track is considered broken.

    workers: int = 1
    splits: Tuple[str, ...] = ("Train", "Validation", "Test")
    limit_clips: int = 0
    overwrite: bool = True


# --------------------------------------------------------------------------- #
# (de)serialisation helpers
# --------------------------------------------------------------------------- #
def _coerce(value: Any, target_type: Any) -> Any:
    """Best-effort coercion of JSON scalars/lists into the declared field type."""
    type_str = str(target_type)
    if "Tuple" in type_str or "tuple" in type_str:
        return tuple(value) if isinstance(value, list) else value
    if target_type is float and isinstance(value, int):
        return float(value)
    return value


def _apply_overrides(instance: Any, overrides: Dict[str, Any]) -> None:
    """Recursively apply a dict of overrides onto a dataclass instance in place."""
    valid = {f.name: f for f in fields(instance)}
    for key, value in overrides.items():
        if key not in valid:
            raise KeyError(
                "Unknown config key '{}' for {}. Valid keys: {}".format(
                    key, type(instance).__name__, sorted(valid)
                )
            )
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_overrides(current, value)
        else:
            setattr(instance, key, _coerce(value, valid[key].type))


def load_config(cls, config_path=None, section=None, **cli_overrides):
    """Build a config: defaults <- JSON file section <- non-None CLI kwargs."""
    cfg = cls()
    if config_path is not None:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if section is not None:
            raw = raw.get(section, {})
        _apply_overrides(cfg, raw)
    clean = {k: v for k, v in cli_overrides.items() if v is not None}
    _apply_overrides(cfg, clean)
    return cfg


def config_to_dict(cfg) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)


def dump_config(cfg, path: Path) -> None:
    """Persist the fully-resolved config next to the outputs, for provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(cfg), indent=2), encoding="utf-8")
