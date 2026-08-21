"""The Motion Reliability Score (MRS) and its five sub-scores.

Each sub-score is mapped to [0, 1] where 1 means "fully reliable", and the final
MRS is their weighted mean. Every calibration constant lives in
:class:`~engagement_pipeline.config.Phase1Config` so the score can be retuned
without touching this file.

    blur                sharpness via variance of Laplacian, measured on the
                        face ROI at its NATIVE resolution. Not on the whole frame
                        (a busy background masks a blurred face) and not on the
                        aligned crop (DAiSEE faces are ~115 px and get upscaled
                        to 224, so interpolation would report every frame as
                        soft regardless of true sharpness).

    face_visibility     the detector's own confidence; 0 when no face was found.

    head_rotation       1 - |(yaw, pitch, roll)| / max_deviation. Extreme pose
                        makes landmarks and gaze unreliable even when the image
                        itself is sharp.

    eye_visibility      whether both eyes are present, inside the frame and
                        inside the (margin-expanded) detection box, with a
                        plausible interocular separation.

    motion_consistency  1 - median optical-flow magnitude between CONSECUTIVE
                        ALIGNED crops / max_flow. Because alignment already
                        cancels genuine head translation and roll, large
                        residual flow indicates detector jitter rather than real
                        movement -- which is exactly what we want to penalise.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

COMPONENT_NAMES = (
    "blur",
    "face_visibility",
    "head_rotation",
    "eye_visibility",
    "motion_consistency",
)


@dataclass
class MRSComponents:
    blur: float = 0.0
    face_visibility: float = 0.0
    head_rotation: float = 0.0
    eye_visibility: float = 0.0
    motion_consistency: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def laplacian_variance(bgr: np.ndarray) -> float:
    """Variance of the Laplacian: the standard no-reference sharpness measure."""
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def face_roi(image: np.ndarray, bbox_xyxy: Sequence[float], margin: float = 0.0) -> np.ndarray:
    """Crop the face box out of the ORIGINAL frame, clamped to image bounds.

    Returns the whole image if the box is degenerate, so callers never have to
    guard against an empty slice.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    box_w, box_h = x2 - x1, y2 - y1
    x1 = int(max(0, np.floor(x1 - margin * box_w)))
    y1 = int(max(0, np.floor(y1 - margin * box_h)))
    x2 = int(min(width, np.ceil(x2 + margin * box_w)))
    y2 = int(min(height, np.ceil(y2 + margin * box_h)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return image
    return image[y1:y2, x1:x2]


def blur_score(roi_bgr: np.ndarray, reference_variance: float) -> float:
    """Saturating map from Laplacian variance to [0, 1].

    Pass the native-resolution face ROI (see :func:`face_roi`), not the aligned
    crop -- see the module docstring for why.
    """
    if reference_variance <= 0:
        return 1.0
    return _clip01(laplacian_variance(roi_bgr) / reference_variance)


def face_visibility_score(confidence: Optional[float]) -> float:
    """Detector confidence is already a [0, 1] reliability measure."""
    if confidence is None:
        return 0.0
    return _clip01(float(confidence))


def head_rotation_score(deviation_deg: float, max_deviation_deg: float) -> float:
    """Linear falloff from frontal (1.0) to ``max_deviation_deg`` (0.0)."""
    if not np.isfinite(deviation_deg):
        return 0.0
    if max_deviation_deg <= 0:
        return 1.0
    return _clip01(1.0 - deviation_deg / max_deviation_deg)


def eye_visibility_score(
    right_eye: Optional[Sequence[float]],
    left_eye: Optional[Sequence[float]],
    bbox_xyxy: Optional[Sequence[float]],
    frame_shape: Sequence[int],
    margin: float = 0.25,
    mesh_eye_openness: Optional[float] = None,
) -> float:
    """Graded score: each eye contributes 0.5 when present and plausibly placed.

    When Phase 1 also runs the face mesh, ``mesh_eye_openness`` (the mean eye
    aspect ratio) modulates the result so that a closed or occluded eye is not
    scored as fully visible just because the coarse keypoint exists.
    """
    if right_eye is None or left_eye is None or bbox_xyxy is None:
        return 0.0

    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    box_w, box_h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    ex1, ey1 = x1 - margin * box_w, y1 - margin * box_h
    ex2, ey2 = x2 + margin * box_w, y2 + margin * box_h

    score = 0.0
    for eye in (right_eye, left_eye):
        ex, ey = float(eye[0]), float(eye[1])
        in_frame = 0 <= ex < width and 0 <= ey < height
        in_box = ex1 <= ex <= ex2 and ey1 <= ey <= ey2
        if in_frame and in_box:
            score += 0.5

    # Reject geometrically implausible eye pairs (collapsed or absurdly wide).
    interocular = float(np.hypot(left_eye[0] - right_eye[0], left_eye[1] - right_eye[1]))
    ratio = interocular / box_w
    if ratio < 0.15 or ratio > 1.2:
        score *= 0.5

    if mesh_eye_openness is not None and np.isfinite(mesh_eye_openness):
        # EAR ~0.30 is a comfortably open eye; ~0.10 is closed. Map onto [0.3, 1]
        # so a blink degrades the score without zeroing an otherwise good frame.
        openness = _clip01((float(mesh_eye_openness) - 0.10) / 0.20)
        score *= 0.3 + 0.7 * openness

    return _clip01(score)


def motion_consistency_score(
    aligned_bgr: np.ndarray,
    previous_aligned_bgr: Optional[np.ndarray],
    max_flow_magnitude_px: float,
) -> tuple:
    """Return ``(score, median_flow_px, reference_available)``.

    With no previous aligned crop (first sampled frame of a clip, or the previous
    frame had no detection) the score is a neutral 1.0 and the caller is told the
    reference was missing, so the manifest can distinguish "consistent" from
    "unmeasured" rather than quietly conflating them.
    """
    if previous_aligned_bgr is None or previous_aligned_bgr.shape != aligned_bgr.shape:
        return 1.0, float("nan"), False

    previous_grey = cv2.cvtColor(previous_aligned_bgr, cv2.COLOR_BGR2GRAY)
    current_grey = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        previous_grey, current_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude = np.linalg.norm(flow, axis=2)
    # Median rather than mean: robust to a few high-flow pixels at the crop edge
    # where BORDER_REPLICATE padding creates artificial motion.
    median_flow = float(np.median(magnitude))
    if max_flow_magnitude_px <= 0:
        return 1.0, median_flow, True
    return _clip01(1.0 - median_flow / max_flow_magnitude_px), median_flow, True


def combine(components: MRSComponents, weights: Dict[str, float]) -> float:
    """Weighted mean of the sub-scores; weights need not sum to 1."""
    values = components.as_dict()
    total_weight = 0.0
    accumulated = 0.0
    for name in COMPONENT_NAMES:
        weight = float(weights.get(name, 0.0))
        if weight == 0.0:
            continue
        accumulated += weight * float(values[name])
        total_weight += weight
    if total_weight <= 0:
        return 0.0
    return _clip01(accumulated / total_weight)
