"""Face detection, cropping and roll alignment.

Detector: MediaPipe Tasks ``vision.FaceDetector`` (BlazeFace short-range).
It is used because MediaPipe is already a hard requirement for the landmark
branch, and because it returns a *confidence score* and eye keypoints, both of
which the MRS face-visibility and alignment steps need.

Failure policy (spec section 9): a frame where no face is found never disappears
from the sequence. It gets a fallback crop, ``found=False``, and a recorded
``source``, so downstream code can see exactly what happened and MRS can score
it as unreliable. The detector is never silently swapped for another one -- the
fallback chain is explicit and every observation carries the source it came from.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from .utils import resolve_path

# BlazeFace keypoint order, per the MediaPipe face detector model card.
# "right"/"left" are from the subject's point of view.
KEYPOINT_NAMES = ("right_eye", "left_eye", "nose_tip", "mouth_center",
                  "right_ear_tragion", "left_ear_tragion")

FACE_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)


@dataclass
class FaceDetection:
    found: bool
    source: str                       # mediapipe_face_detector | landmark_bbox | center_fallback
    bbox: tuple[int, int, int, int]   # x0, y0, x1, y1 in pixels, clipped to the frame
    confidence: float | None
    area_fraction: float              # bbox area / frame area
    keypoints: dict[str, tuple[float, float]] | None
    roll_deg: float | None            # eye-line angle used for alignment

    def as_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d


def _clip_box(x0: float, y0: float, x1: float, y1: float, w: int, h: int) -> tuple[int, int, int, int]:
    x0 = int(max(0, min(w - 1, round(x0))))
    y0 = int(max(0, min(h - 1, round(y0))))
    x1 = int(max(x0 + 1, min(w, round(x1))))
    y1 = int(max(y0 + 1, min(h, round(y1))))
    return x0, y0, x1, y1


def _pad_box(box: tuple[int, int, int, int], padding: float, w: int, h: int):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    return _clip_box(x0 - padding * bw, y0 - padding * bh,
                     x1 + padding * bw, y1 + padding * bh, w, h)


class FaceDetector:
    """Thin wrapper over the MediaPipe Tasks face detector.

    The wrapper exists so that (a) the model path and thresholds are recorded in
    one place, and (b) an import or initialisation failure surfaces as a clear error at
    construction time rather than as a mysterious empty result later.
    """

    def __init__(self, model_path: str | Path, min_confidence: float = 0.3):
        self.model_path = str(resolve_path(model_path))
        self.min_confidence = float(min_confidence)
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(
                f"MediaPipe face detector bundle not found at {self.model_path}. "
                f"Run: python scripts/fetch_models.py"
            )
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self.version = getattr(mp, "__version__", "unknown")
        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=self.min_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def describe(self) -> dict:
        return {
            "detector": "mediapipe.tasks.vision.FaceDetector (blaze_face_short_range)",
            "mediapipe_version": self.version,
            "model_path": self.model_path,
            "min_detection_confidence": self.min_confidence,
        }

    def detect(self, frame_bgr: np.ndarray) -> FaceDetection:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._detector.detect(image)
        detections = getattr(result, "detections", None) or []
        if not detections:
            return FaceDetection(False, "none", (0, 0, w, h), None, 1.0, None, None)

        # Largest face: DAiSEE clips are single-subject webcam recordings, so the
        # biggest box is the participant if a bystander is ever picked up.
        def area(d):
            bb = d.bounding_box
            return bb.width * bb.height

        det = max(detections, key=area)
        bb = det.bounding_box
        box = _clip_box(bb.origin_x, bb.origin_y,
                        bb.origin_x + bb.width, bb.origin_y + bb.height, w, h)
        score = None
        cats = getattr(det, "categories", None)
        if cats:
            score = float(cats[0].score)

        keypoints = None
        roll = None
        kps = getattr(det, "keypoints", None)
        if kps:
            keypoints = {}
            for name, kp in zip(KEYPOINT_NAMES, kps):
                keypoints[name] = (float(kp.x) * w, float(kp.y) * h)
            if "right_eye" in keypoints and "left_eye" in keypoints:
                rx, ry = keypoints["right_eye"]
                lx, ly = keypoints["left_eye"]
                roll = float(np.degrees(np.arctan2(ly - ry, lx - rx)))

        area_fraction = ((box[2] - box[0]) * (box[3] - box[1])) / float(w * h)
        return FaceDetection(True, "mediapipe_face_detector", box, score,
                             area_fraction, keypoints, roll)

    def close(self) -> None:
        try:
            self._detector.close()
        except Exception:
            pass


def bbox_from_landmarks(landmarks_xy: np.ndarray, w: int, h: int,
                        margin: float = 0.0) -> tuple[int, int, int, int]:
    """Tight box around normalised (x, y) landmark coordinates."""
    xs = landmarks_xy[:, 0] * w
    ys = landmarks_xy[:, 1] * h
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    if margin:
        bw, bh = x1 - x0, y1 - y0
        x0, x1 = x0 - margin * bw, x1 + margin * bw
        y0, y1 = y0 - margin * bh, y1 + margin * bh
    return _clip_box(x0, y0, x1, y1, w, h)


def center_square(w: int, h: int, fraction: float = 0.6) -> tuple[int, int, int, int]:
    """Deterministic fallback region when no face is located at all."""
    side = int(min(w, h) * fraction)
    cx, cy = w // 2, h // 2
    return _clip_box(cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2, w, h)


def crop_face(frame_bgr: np.ndarray,
              detection: FaceDetection,
              image_size: int,
              padding: float = 0.25,
              align: bool = True) -> np.ndarray:
    """Crop, optionally roll-align, and resize to a square image_size crop.

    Alignment rotates the whole frame about the face centre so the inter-ocular
    line is horizontal, then crops. Rotating before cropping avoids the black
    corners that rotating a tight crop would introduce.
    """
    h, w = frame_bgr.shape[:2]
    box = _pad_box(detection.bbox, padding, w, h)
    src = frame_bgr

    if align and detection.roll_deg is not None and abs(detection.roll_deg) > 1e-3:
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        matrix = cv2.getRotationMatrix2D((cx, cy), detection.roll_deg, 1.0)
        src = cv2.warpAffine(frame_bgr, matrix, (w, h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    crop = src[box[1]:box[3], box[0]:box[2]]
    if crop.size == 0:
        crop = src
    return cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
