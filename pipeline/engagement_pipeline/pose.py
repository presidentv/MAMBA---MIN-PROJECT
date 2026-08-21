"""Head-pose estimation and rotation-matrix decomposition.

Two independent estimators are provided because the two phases have different
information available:

Phase 1 has only the six BlazeFace keypoints, so it uses ``solvePnP`` against a
coarse six-point anthropometric model. This is cheap (it runs on every sampled
frame of every clip) and accurate enough to drive the MRS head-rotation term.

Phase 2 has the full 478-point mesh *and* MediaPipe's own 4x4 facial
transformation matrix, so it reads pose straight off that matrix and
cross-checks it with a ``solvePnP`` fit over a stable landmark subset.

Angle convention (shared by both, degrees):
    yaw    +ve = subject turns to their own left
    pitch  +ve = subject looks up
    roll   +ve = subject's head tilts clockwise in the image
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# 3D reference models (millimetres, +X to the subject's left as seen in the
# image, +Y down, +Z away from the camera).
# --------------------------------------------------------------------------- #
BLAZEFACE_6PT_MODEL = np.array(
    [
        [-30.0, -32.0, -30.0],  # right eye  (subject's right -> image left)
        [30.0, -32.0, -30.0],   # left eye
        [0.0, 0.0, 0.0],        # nose tip (origin)
        [0.0, 35.0, -25.0],     # mouth centre
        [-75.0, -20.0, -95.0],  # right ear tragion
        [75.0, -20.0, -95.0],   # left ear tragion
    ],
    dtype=np.float64,
)

# Mesh landmark indices that are stable under expression change, paired with
# their approximate 3D positions in the same frame as the model above.
MESH_PNP_INDICES: Tuple[int, ...] = (1, 152, 33, 263, 61, 291)
MESH_PNP_MODEL = np.array(
    [
        [0.0, 0.0, 0.0],        # 1   nose tip
        [0.0, 63.0, -13.0],     # 152 chin
        [-43.0, -32.0, -26.0],  # 33  right eye outer corner
        [43.0, -32.0, -26.0],   # 263 left eye outer corner
        [-28.0, 28.0, -24.0],   # 61  right mouth corner
        [28.0, 28.0, -24.0],    # 291 left mouth corner
    ],
    dtype=np.float64,
)


@dataclass
class HeadPose:
    yaw: float
    pitch: float
    roll: float
    success: bool = True
    reprojection_error: float = float("nan")

    @property
    def deviation(self) -> float:
        """L2 magnitude of the three angles: 0 for a perfectly frontal head."""
        return float(np.sqrt(self.yaw**2 + self.pitch**2 + self.roll**2))

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.yaw, self.pitch, self.roll)


UNKNOWN_POSE = HeadPose(yaw=float("nan"), pitch=float("nan"), roll=float("nan"), success=False)


def default_camera_matrix(width: int, height: int) -> np.ndarray:
    """Pinhole intrinsics assuming a ~60 degree horizontal FOV webcam.

    DAiSEE ships no calibration data, so focal length is approximated by the
    image width. Absolute translation is meaningless under this assumption, but
    the *rotation* -- all we consume -- is only mildly affected.
    """
    focal = float(width)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_matrix_to_euler(rotation: np.ndarray) -> Tuple[float, float, float]:
    """Decompose a 3x3 rotation matrix into (yaw, pitch, roll) in degrees.

    Uses the y-x-z (yaw-pitch-roll) ordering that matches the axis convention
    documented at the top of this module, with an explicit gimbal-lock branch.
    """
    r = np.asarray(rotation, dtype=np.float64)
    sy = float(np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
    if sy > 1e-6:
        rot_x = np.arctan2(r[2, 1], r[2, 2])   # about X -> pitch
        rot_y = np.arctan2(-r[2, 0], sy)       # about Y -> yaw
        rot_z = np.arctan2(r[1, 0], r[0, 0])   # about Z -> roll
    else:  # gimbal lock: X and Z rotations are degenerate, pin roll to 0
        rot_x = np.arctan2(-r[1, 2], r[1, 1])
        rot_y = np.arctan2(-r[2, 0], sy)
        rot_z = 0.0
    return (
        float(np.degrees(rot_y)),
        float(np.degrees(rot_x)),
        float(np.degrees(rot_z)),
    )


def _solve(
    object_points: np.ndarray,
    image_points: np.ndarray,
    width: int,
    height: int,
) -> HeadPose:
    camera_matrix = default_camera_matrix(width, height)
    distortion = np.zeros((4, 1), dtype=np.float64)
    object_points = np.ascontiguousarray(object_points.reshape(-1, 3))
    image_points = np.ascontiguousarray(image_points.reshape(-1, 2).astype(np.float64))
    if object_points.shape[0] != image_points.shape[0] or object_points.shape[0] < 4:
        return UNKNOWN_POSE

    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        return UNKNOWN_POSE
    if not ok:
        return UNKNOWN_POSE

    rotation, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = rotation_matrix_to_euler(rotation)

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    error = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)))

    # cv2's world-to-camera rotation has +Y down; flip pitch/roll sign so that
    # "looking up" and "clockwise tilt" come out positive as documented.
    return HeadPose(yaw=yaw, pitch=-pitch, roll=-roll, success=True, reprojection_error=error)


def pose_from_keypoints(keypoints: np.ndarray, width: int, height: int) -> HeadPose:
    """Estimate head pose from the six BlazeFace keypoints (Phase 1)."""
    if keypoints is None or len(keypoints) < 6:
        return UNKNOWN_POSE
    return _solve(BLAZEFACE_6PT_MODEL, np.asarray(keypoints)[:6, :2], width, height)


def pose_from_mesh(
    points_normalised: np.ndarray,
    width: int,
    height: int,
    indices: Sequence[int] = MESH_PNP_INDICES,
    model: np.ndarray = MESH_PNP_MODEL,
) -> HeadPose:
    """Estimate head pose by fitting a stable mesh landmark subset (Phase 2)."""
    if points_normalised is None or points_normalised.shape[0] <= max(indices):
        return UNKNOWN_POSE
    image_points = points_normalised[list(indices)][:, :2].astype(np.float64)
    image_points[:, 0] *= width
    image_points[:, 1] *= height
    return _solve(model, image_points, width, height)


def pose_from_transformation_matrix(matrix: Optional[np.ndarray]) -> HeadPose:
    """Read pose off MediaPipe's 4x4 facial transformation matrix (Phase 2).

    This is MediaPipe's own canonical-face-to-camera fit and is generally the
    steadier of the two estimates, since it uses all 478 points rather than six.
    """
    if matrix is None:
        return UNKNOWN_POSE
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        return UNKNOWN_POSE
    yaw, pitch, roll = rotation_matrix_to_euler(matrix[:3, :3])
    return HeadPose(yaw=yaw, pitch=-pitch, roll=-roll, success=True)
