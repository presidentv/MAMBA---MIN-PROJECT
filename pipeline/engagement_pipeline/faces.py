"""Thin wrappers over the MediaPipe Tasks vision models.

Both wrappers are context managers and hold a single native task instance, which
is *not* thread-safe -- one instance per process. The Phase drivers therefore
build them lazily inside each worker process rather than sharing one.

Empirically verified against mediapipe 1.0.1 on this dataset:
  * FaceDetector returns ``bounding_box`` in **pixels** and ``keypoints`` in
    **normalised [0, 1]** image coordinates, with confidence in
    ``detection.categories[0].score``.
  * The 6 keypoints are ordered
    ``[right_eye, left_eye, nose_tip, mouth_centre, right_ear, left_ear]``
    where right/left are the **subject's** own, i.e. ``right_eye`` sits at the
    smaller x for a frontal face.
  * FaceLandmarker returns 478 landmarks -- the 468 mesh points plus the
    iris refinement block at indices 468-477 -- and, when requested, a 4x4
    facial transformation matrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Keypoint indices in the BlazeFace 6-point output.
KP_RIGHT_EYE = 0
KP_LEFT_EYE = 1
KP_NOSE_TIP = 2
KP_MOUTH = 3
KP_RIGHT_EAR = 4
KP_LEFT_EAR = 5
KEYPOINT_NAMES = ("right_eye", "left_eye", "nose_tip", "mouth", "right_ear", "left_ear")

# Landmark index blocks in the 478-point iris-refined mesh.
NUM_MESH_LANDMARKS = 478
LEFT_IRIS_IDX = (468, 469, 470, 471, 472)
RIGHT_IRIS_IDX = (473, 474, 475, 476, 477)

# Eye contour subsets used for the eye-aspect-ratio / openness signals. Ordered
# (outer_corner, upper_1, upper_2, inner_corner, lower_1, lower_2) so that the
# classic 6-point EAR formula applies directly.
LEFT_EYE_EAR_IDX = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR_IDX = (362, 385, 387, 263, 373, 380)
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)


@dataclass
class FaceDetection:
    """One detected face in image (pixel) coordinates."""

    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    keypoints: np.ndarray
    """(6, 2) float array of pixel coordinates, in ``KEYPOINT_NAMES`` order."""

    @property
    def right_eye(self) -> np.ndarray:
        return self.keypoints[KP_RIGHT_EYE]

    @property
    def left_eye(self) -> np.ndarray:
        return self.keypoints[KP_LEFT_EYE]

    @property
    def bbox_wh(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return x2 - x1, y2 - y1

    @property
    def area(self) -> float:
        w, h = self.bbox_wh
        return max(w, 0.0) * max(h, 0.0)


@dataclass
class FaceLandmarks:
    """478-point mesh result for a single face."""

    points: np.ndarray
    """(478, 3) normalised coordinates: x, y in [0, 1] of the input image, z is
    relative depth in roughly the same scale as x."""
    transformation_matrix: Optional[np.ndarray] = None
    """(4, 4) canonical-face -> camera transform, when requested."""
    blendshapes: Optional[np.ndarray] = None
    blendshape_names: Optional[Sequence[str]] = None


def _import_tasks():
    """Import MediaPipe lazily so that ``--help`` works without loading the runtime."""
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    return mp, mpp, vision


def to_mp_image(bgr: np.ndarray):
    """Wrap a BGR uint8 OpenCV frame as an sRGB MediaPipe Image."""
    mp, _, _ = _import_tasks()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


class FaceDetectorWrapper:
    """BlazeFace short-range detector, used by Phase 1 for gating and alignment."""

    def __init__(self, model_path: Path, min_detection_confidence: float = 0.4):
        _, mpp, vision = _import_tasks()
        options = vision.FaceDetectorOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=min_detection_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect_all(self, bgr: np.ndarray) -> List[FaceDetection]:
        """Return every detection in the frame, in pixel coordinates.

        DAiSEE was filmed in shared rooms, so bystanders are common; the caller
        selects the subject via :func:`select_primary_face` rather than blindly
        taking the top score.
        """
        height, width = bgr.shape[:2]
        result = self._detector.detect(to_mp_image(bgr))
        if not result.detections:
            return []

        detections: List[FaceDetection] = []
        for det in result.detections:
            score = float(det.categories[0].score) if det.categories else 0.0
            box = det.bounding_box
            bbox = (
                float(box.origin_x),
                float(box.origin_y),
                float(box.origin_x + box.width),
                float(box.origin_y + box.height),
            )
            keypoints = np.array(
                [[kp.x * width, kp.y * height] for kp in det.keypoints], dtype=np.float64
            )
            if keypoints.shape[0] < 6:
                # Should not happen with BlazeFace, but never index blindly.
                logger.debug("Detection with %d keypoints ignored", keypoints.shape[0])
                continue
            detections.append(FaceDetection(confidence=score, bbox_xyxy=bbox, keypoints=keypoints))
        return detections

    def detect(self, bgr: np.ndarray) -> Optional[FaceDetection]:
        """Convenience wrapper returning the single largest, most confident face."""
        detections = self.detect_all(bgr)
        if not detections:
            return None
        return max(detections, key=lambda d: (d.confidence, d.area))

    def close(self) -> None:
        try:
            self._detector.close()
        except Exception:  # pragma: no cover - native teardown is best-effort
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


class FaceLandmarkerWrapper:
    """478-point iris-refined face mesh, used by Phase 2."""

    def __init__(
        self,
        model_path: Path,
        min_face_detection_confidence: float = 0.3,
        min_face_presence_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
        output_blendshapes: bool = False,
    ):
        _, mpp, vision = _import_tasks()
        options = vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=output_blendshapes,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._want_blendshapes = output_blendshapes

    def detect(self, bgr: np.ndarray) -> Optional[FaceLandmarks]:
        result = self._landmarker.detect(to_mp_image(bgr))
        if not result.face_landmarks:
            return None

        points = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]], dtype=np.float32
        )
        matrix = None
        if getattr(result, "facial_transformation_matrixes", None):
            matrix = np.array(result.facial_transformation_matrixes[0], dtype=np.float64)

        blendshapes = names = None
        if self._want_blendshapes and getattr(result, "face_blendshapes", None):
            categories = result.face_blendshapes[0]
            blendshapes = np.array([c.score for c in categories], dtype=np.float32)
            names = [c.category_name for c in categories]

        return FaceLandmarks(
            points=points,
            transformation_matrix=matrix,
            blendshapes=blendshapes,
            blendshape_names=names,
        )

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # pragma: no cover
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def bbox_centre(detection: FaceDetection) -> np.ndarray:
    x1, y1, x2, y2 = detection.bbox_xyxy
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)


def select_primary_face(
    detections: Sequence[FaceDetection],
    previous_centre: Optional[np.ndarray] = None,
    reference_width: Optional[float] = None,
    max_jump_ratio: float = 1.5,
) -> Optional[FaceDetection]:
    """Pick the clip's subject from possibly several detected faces.

    On the first frame the largest, most confident face wins -- in DAiSEE the
    recorded subject is by far the closest to the webcam. On later frames a
    candidate within ``max_jump_ratio`` face-widths of the previously accepted
    centre is preferred, which stops the "primary face" hopping to a bystander
    when the subject briefly turns away. If nothing is close enough, the
    fallback is again largest-and-most-confident, and the caller can detect the
    discontinuity from the returned centre.
    """
    if not detections:
        return None
    if previous_centre is None or reference_width is None or reference_width <= 0:
        return max(detections, key=lambda d: (d.area, d.confidence))

    limit = max_jump_ratio * reference_width
    nearby = [d for d in detections if np.linalg.norm(bbox_centre(d) - previous_centre) <= limit]
    if nearby:
        return min(nearby, key=lambda d: float(np.linalg.norm(bbox_centre(d) - previous_centre)))
    return max(detections, key=lambda d: (d.area, d.confidence))


def eye_aspect_ratio(points: np.ndarray, idx: Sequence[int]) -> float:
    """Classic 6-point eye aspect ratio (Soukupova & Cech 2016).

    ``points`` may be normalised or pixel coordinates -- EAR is a ratio, so it is
    scale invariant either way, provided x and y share a scale.
    """
    p = points[list(idx)][:, :2].astype(np.float64)
    horizontal = np.linalg.norm(p[0] - p[3])
    if horizontal <= 1e-9:
        return 0.0
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    return float(vertical / (2.0 * horizontal))


def iris_centre(points: np.ndarray, idx: Sequence[int] = LEFT_IRIS_IDX) -> np.ndarray:
    """Centroid of one iris landmark block, as (x, y, z)."""
    return points[list(idx)].mean(axis=0)
