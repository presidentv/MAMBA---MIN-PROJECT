"""Canonical eye-line face alignment and pixel normalisation.

The aligned crop is the unit of everything downstream: Phase 2 landmarks and the
affect model both read these crops rather than re-decoding video, so the
2x3 affine used to produce each crop is stored alongside it. That makes every
landmark back-projectable to original-frame pixel coordinates via
:func:`invert_affine`, and keeps the roll angle recoverable after alignment has
deliberately removed it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np


@dataclass
class AlignedFace:
    image: np.ndarray
    """(size, size, 3) BGR uint8 aligned crop."""
    affine: np.ndarray
    """(2, 3) transform mapping original-frame pixels -> aligned-crop pixels."""
    rotation_deg: float
    """In-plane rotation removed by the alignment. Positive = the original eye
    line was tilted clockwise in the image; add this back to any roll measured
    in the aligned crop to recover the original-frame roll."""
    scale: float
    """Isotropic scale factor applied (aligned px per original px)."""
    interocular_px: float
    """Eye-to-eye distance in the ORIGINAL frame; a direct proxy for face size."""


def align_face(
    image: np.ndarray,
    right_eye: Sequence[float],
    left_eye: Sequence[float],
    output_size: int = 224,
    desired_left_eye_x: float = 0.35,
    desired_eye_y: float = 0.38,
) -> AlignedFace:
    """Rotate, scale and translate so the eyes sit on a fixed canonical line.

    ``right_eye`` / ``left_eye`` are the SUBJECT's right and left eyes in
    original-frame pixel coordinates, so ``right_eye`` normally has the smaller
    x. Fixing both eye positions in the output pins translation, rotation and
    scale simultaneously, which is what makes crops comparable across frames.
    """
    right = np.asarray(right_eye, dtype=np.float64)[:2]
    left = np.asarray(left_eye, dtype=np.float64)[:2]

    delta = left - right
    interocular = float(np.hypot(delta[0], delta[1]))
    angle_deg = float(np.degrees(np.arctan2(delta[1], delta[0])))

    # The subject's right eye lands at desired_left_eye_x (it is on the left of
    # the IMAGE), the left eye mirrors it, so their separation fixes the scale.
    desired_separation = (1.0 - 2.0 * desired_left_eye_x) * output_size
    scale = desired_separation / interocular if interocular > 1e-6 else 1.0

    eyes_centre = ((right + left) / 2.0).tolist()
    affine = cv2.getRotationMatrix2D(tuple(eyes_centre), angle_deg, scale)
    # Shift the (already rotated/scaled) eye centre onto its canonical spot.
    affine[0, 2] += output_size * 0.5 - eyes_centre[0]
    affine[1, 2] += output_size * desired_eye_y - eyes_centre[1]

    aligned = cv2.warpAffine(
        image,
        affine,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return AlignedFace(
        image=aligned,
        affine=affine.astype(np.float64),
        rotation_deg=angle_deg,
        scale=float(scale),
        interocular_px=interocular,
    )


def invert_affine(affine: np.ndarray) -> np.ndarray:
    """Invert a 2x3 affine so aligned-crop pixels map back to original pixels."""
    return cv2.invertAffineTransform(np.asarray(affine, dtype=np.float64))


def apply_affine(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine to an (N, 2) array of points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))])
    return homogeneous @ np.asarray(affine, dtype=np.float64).T


def affine_to_dict(affine: np.ndarray, prefix: str = "affine") -> dict:
    """Flatten a 2x3 affine into six scalar columns for tabular storage."""
    flat = np.asarray(affine, dtype=np.float64).reshape(6)
    keys = ["a00", "a01", "a02", "a10", "a11", "a12"]
    return {f"{prefix}_{k}": float(v) for k, v in zip(keys, flat)}


def dict_to_affine(row: dict, prefix: str = "affine") -> np.ndarray:
    """Inverse of :func:`affine_to_dict`."""
    keys = ["a00", "a01", "a02", "a10", "a11", "a12"]
    return np.array([float(row[f"{prefix}_{k}"]) for k in keys], dtype=np.float64).reshape(2, 3)


def normalize_image(
    bgr: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Convert a BGR uint8 crop to a normalised float32 CHW RGB tensor.

    Kept as a function rather than baked into the saved artefacts: storing
    float32 tensors would inflate the output tree roughly tenfold for something
    that is a deterministic function of the stored image. The mean/std defaults
    are the ImageNet statistics that the frozen affect backbones expect.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.transpose(rgb, (2, 0, 1))
