"""MediaPipe Face Landmarker branch: geometry, gaze, head pose and blendshapes.

Why the Tasks Face Landmarker rather than raw Face Mesh coordinates: besides the
478 landmarks (which include the 10 iris points), it also returns

  * 52 **blendshape** activations -- semantic, action-unit-like scores such as
    ``eyeBlinkLeft``, ``browDownRight``, ``jawOpen``, ``mouthSmileLeft``;
  * a 4x4 **facial transformation matrix** from which head pose is decomposed.

The blendshapes come from a pretrained head that was trained for exactly this
kind of expression read-out, so they are far more informative per dimension than
concatenated raw coordinates -- which spec section 12 explicitly warns against.

Every geometric feature is normalised by a face-scale reference (inter-ocular
distance or face width) so absolute pixel size cannot dominate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .utils import resolve_path

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# Canonical MediaPipe / ARKit blendshape ordering. Used as the authoritative
# column order so the feature vector has a stable layout across runs. The
# extractor *verifies* the model actually returns this exact set and raises if
# it does not -- the order is checked, not assumed.
BLENDSHAPE_NAMES: tuple[str, ...] = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
    "noseSneerRight",
)

# Canonical face-mesh vertex indices (MediaPipe 478-point topology).
# "left"/"right" follow MediaPipe's naming, which is the *image* side.
RIGHT_EYE = dict(outer=33, inner=133, top=(159, 160, 158), bottom=(145, 144, 153))
LEFT_EYE = dict(outer=263, inner=362, top=(386, 385, 387), bottom=(374, 380, 373))
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473
MOUTH = dict(left=61, right=291, top=13, bottom=14, upper_outer=0, lower_outer=17)
RIGHT_BROW = 105
LEFT_BROW = 334
NOSE_TIP = 1
FACE_LEFT = 234
FACE_RIGHT = 454
FACE_TOP = 10
FACE_BOTTOM = 152

MIN_LANDMARKS = 478

GEOMETRIC_FEATURE_NAMES: tuple[str, ...] = (
    "ear_right", "ear_left", "ear_mean", "ear_asymmetry",
    "eye_open_right", "eye_open_left",
    "iris_dx_right", "iris_dy_right", "iris_dx_left", "iris_dy_left",
    "iris_dx_mean", "iris_dy_mean", "gaze_vergence",
    "mouth_aspect_ratio", "mouth_width_norm", "mouth_corner_lift",
    "brow_eye_right", "brow_eye_left", "brow_eye_mean", "brow_asymmetry",
    "head_yaw_norm", "head_pitch_norm", "head_roll_norm",
    "face_area_fraction", "face_aspect_ratio",
    "nose_offset_x", "nose_offset_y", "inter_ocular_norm",
)

FEATURE_NAMES: tuple[str, ...] = GEOMETRIC_FEATURE_NAMES + tuple(
    f"blendshape_{n}" for n in BLENDSHAPE_NAMES
)
FEATURE_DIM = len(FEATURE_NAMES)

# Groups used for SHAP attribution reporting (spec section 32).
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "eye_openness": ("ear_right", "ear_left", "ear_mean", "ear_asymmetry",
                     "eye_open_right", "eye_open_left",
                     "blendshape_eyeBlinkLeft", "blendshape_eyeBlinkRight",
                     "blendshape_eyeSquintLeft", "blendshape_eyeSquintRight",
                     "blendshape_eyeWideLeft", "blendshape_eyeWideRight"),
    "iris_gaze": ("iris_dx_right", "iris_dy_right", "iris_dx_left", "iris_dy_left",
                  "iris_dx_mean", "iris_dy_mean", "gaze_vergence",
                  "blendshape_eyeLookDownLeft", "blendshape_eyeLookDownRight",
                  "blendshape_eyeLookInLeft", "blendshape_eyeLookInRight",
                  "blendshape_eyeLookOutLeft", "blendshape_eyeLookOutRight",
                  "blendshape_eyeLookUpLeft", "blendshape_eyeLookUpRight"),
    "head_pose": ("head_yaw_norm", "head_pitch_norm", "head_roll_norm",
                  "nose_offset_x", "nose_offset_y"),
    "brow": ("brow_eye_right", "brow_eye_left", "brow_eye_mean", "brow_asymmetry",
             "blendshape_browDownLeft", "blendshape_browDownRight",
             "blendshape_browInnerUp", "blendshape_browOuterUpLeft",
             "blendshape_browOuterUpRight"),
    "mouth_jaw": ("mouth_aspect_ratio", "mouth_width_norm", "mouth_corner_lift",
                  "blendshape_jawOpen", "blendshape_jawForward", "blendshape_jawLeft",
                  "blendshape_jawRight", "blendshape_mouthClose",
                  "blendshape_mouthSmileLeft", "blendshape_mouthSmileRight",
                  "blendshape_mouthFrownLeft", "blendshape_mouthFrownRight",
                  "blendshape_mouthPucker", "blendshape_mouthFunnel",
                  "blendshape_mouthDimpleLeft", "blendshape_mouthDimpleRight",
                  "blendshape_mouthLeft", "blendshape_mouthRight",
                  "blendshape_mouthLowerDownLeft", "blendshape_mouthLowerDownRight",
                  "blendshape_mouthPressLeft", "blendshape_mouthPressRight",
                  "blendshape_mouthRollLower", "blendshape_mouthRollUpper",
                  "blendshape_mouthShrugLower", "blendshape_mouthShrugUpper",
                  "blendshape_mouthStretchLeft", "blendshape_mouthStretchRight",
                  "blendshape_mouthUpperUpLeft", "blendshape_mouthUpperUpRight"),
    "cheek_nose": ("blendshape_cheekPuff", "blendshape_cheekSquintLeft",
                   "blendshape_cheekSquintRight", "blendshape_noseSneerLeft",
                   "blendshape_noseSneerRight"),
    "face_scale": ("face_area_fraction", "face_aspect_ratio", "inter_ocular_norm"),
    "neutral": ("blendshape__neutral",),
}

# Every per-frame feature must belong to exactly one group, otherwise SHAP
# attribution silently pools the leftovers into an uninterpretable bucket -- which
# is what happened on the first run, with the single highest-attribution feature
# landing in it. Checked at import so a new feature cannot be added without also
# being classified.
_grouped = [name for members in FEATURE_GROUPS.values() for name in members]
_missing = set(FEATURE_NAMES) - set(_grouped)
_duplicated = {n for n in _grouped if _grouped.count(n) > 1}
if _missing or _duplicated:  # pragma: no cover - import-time contract
    raise RuntimeError(
        "FEATURE_GROUPS must partition FEATURE_NAMES exactly.\n"
        f"  ungrouped: {sorted(_missing)}\n"
        f"  in more than one group: {sorted(_duplicated)}"
    )


@dataclass
class LandmarkObservation:
    found: bool
    landmarks: np.ndarray | None       # (N, 3) normalised x, y, z
    blendshapes: np.ndarray | None     # (52,) in [0, 1]
    head_pose_deg: tuple[float, float, float] | None   # yaw, pitch, roll
    features: np.ndarray               # (FEATURE_DIM,) float32, zeros when not found


def _euler_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    """Decompose the 3x3 rotation block into (yaw, pitch, roll) degrees.

    Convention: rotation about Y is yaw, about X is pitch, about Z is roll, taken
    from the standard ZYX decomposition. Signs are consistent across frames,
    which is all the reliability term and the temporal features require; they are
    not calibrated against a ground-truth head-pose benchmark.
    """
    r = matrix[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:  # gimbal-locked
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def _mean_point(pts: np.ndarray, idx) -> np.ndarray:
    return pts[list(idx)].mean(axis=0)


def _eye_aspect_ratio(pts: np.ndarray, eye: dict) -> float:
    """Vertical lid separation over horizontal corner separation.

    A low EAR means the eye is closed. Note that a blink is normal behaviour, not
    an unreliable observation -- see src/mrs.py, which deliberately does not
    penalise closed eyes (spec section 10.4).
    """
    width = _dist(pts[eye["outer"]], pts[eye["inner"]])
    if width < 1e-8:
        return 0.0
    top = _mean_point(pts, eye["top"])
    bottom = _mean_point(pts, eye["bottom"])
    return float(_dist(top, bottom) / width)


def compute_geometric_features(pts: np.ndarray,
                               head_pose: tuple[float, float, float] | None,
                               frame_wh: tuple[int, int]) -> dict[str, float]:
    """All geometry is expressed relative to face scale, never in raw pixels."""
    w, h = frame_wh
    aspect = w / float(h) if h else 1.0
    # Work in aspect-corrected normalised units so ratios are not skewed by a
    # non-square frame.
    p = pts.copy()
    p[:, 0] *= aspect

    r_eye_c = (p[RIGHT_EYE["outer"]] + p[RIGHT_EYE["inner"]]) / 2.0
    l_eye_c = (p[LEFT_EYE["outer"]] + p[LEFT_EYE["inner"]]) / 2.0
    iod = _dist(r_eye_c, l_eye_c)
    if iod < 1e-8:
        iod = 1e-8

    face_w = _dist(p[FACE_LEFT], p[FACE_RIGHT])
    face_h = _dist(p[FACE_TOP], p[FACE_BOTTOM])
    face_scale = max(face_w, 1e-8)

    ear_r = _eye_aspect_ratio(p, RIGHT_EYE)
    ear_l = _eye_aspect_ratio(p, LEFT_EYE)

    lid_r = _dist(_mean_point(p, RIGHT_EYE["top"]), _mean_point(p, RIGHT_EYE["bottom"])) / iod
    lid_l = _dist(_mean_point(p, LEFT_EYE["top"]), _mean_point(p, LEFT_EYE["bottom"])) / iod

    # Iris offset from the eye centre, normalised by that eye's own width: a
    # scale-free proxy for gaze direction.
    r_eye_w = max(_dist(p[RIGHT_EYE["outer"]], p[RIGHT_EYE["inner"]]), 1e-8)
    l_eye_w = max(_dist(p[LEFT_EYE["outer"]], p[LEFT_EYE["inner"]]), 1e-8)
    r_iris = p[RIGHT_IRIS_CENTER]
    l_iris = p[LEFT_IRIS_CENTER]
    iris_dx_r = float((r_iris[0] - r_eye_c[0]) / r_eye_w)
    iris_dy_r = float((r_iris[1] - r_eye_c[1]) / r_eye_w)
    iris_dx_l = float((l_iris[0] - l_eye_c[0]) / l_eye_w)
    iris_dy_l = float((l_iris[1] - l_eye_c[1]) / l_eye_w)

    mouth_w = _dist(p[MOUTH["left"]], p[MOUTH["right"]])
    mouth_h = _dist(p[MOUTH["top"]], p[MOUTH["bottom"]])
    mar = float(mouth_h / max(mouth_w, 1e-8))
    corner_mid_y = (p[MOUTH["left"]][1] + p[MOUTH["right"]][1]) / 2.0
    lip_mid_y = (p[MOUTH["top"]][1] + p[MOUTH["bottom"]][1]) / 2.0
    corner_lift = float((lip_mid_y - corner_mid_y) / iod)

    brow_r = float(_dist(p[RIGHT_BROW], r_eye_c) / iod)
    brow_l = float(_dist(p[LEFT_BROW], l_eye_c) / iod)

    yaw, pitch, roll = head_pose if head_pose else (0.0, 0.0, 0.0)

    nose = p[NOSE_TIP]
    face_c = (p[FACE_LEFT] + p[FACE_RIGHT]) / 2.0

    return {
        "ear_right": ear_r,
        "ear_left": ear_l,
        "ear_mean": (ear_r + ear_l) / 2.0,
        "ear_asymmetry": abs(ear_r - ear_l),
        "eye_open_right": float(lid_r),
        "eye_open_left": float(lid_l),
        "iris_dx_right": iris_dx_r,
        "iris_dy_right": iris_dy_r,
        "iris_dx_left": iris_dx_l,
        "iris_dy_left": iris_dy_l,
        "iris_dx_mean": (iris_dx_r + iris_dx_l) / 2.0,
        "iris_dy_mean": (iris_dy_r + iris_dy_l) / 2.0,
        "gaze_vergence": iris_dx_r - iris_dx_l,
        "mouth_aspect_ratio": mar,
        "mouth_width_norm": float(mouth_w / face_scale),
        "mouth_corner_lift": corner_lift,
        "brow_eye_right": brow_r,
        "brow_eye_left": brow_l,
        "brow_eye_mean": (brow_r + brow_l) / 2.0,
        "brow_asymmetry": abs(brow_r - brow_l),
        "head_yaw_norm": float(np.clip(yaw / 90.0, -2.0, 2.0)),
        "head_pitch_norm": float(np.clip(pitch / 90.0, -2.0, 2.0)),
        "head_roll_norm": float(np.clip(roll / 90.0, -2.0, 2.0)),
        "face_area_fraction": float(face_w * face_h),
        "face_aspect_ratio": float(face_h / max(face_w, 1e-8)),
        "nose_offset_x": float((nose[0] - face_c[0]) / face_scale),
        "nose_offset_y": float((nose[1] - face_c[1]) / face_scale),
        "inter_ocular_norm": float(iod / face_scale),
    }


class LandmarkExtractor:
    """MediaPipe Tasks FaceLandmarker wrapper producing a fixed-width feature row."""

    def __init__(self, cfg_landmarks: dict):
        self.model_path = str(resolve_path(cfg_landmarks["model_bundle"]))
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(
                f"MediaPipe face landmarker bundle not found at {self.model_path}. "
                f"Run: python scripts/fetch_models.py"
            )
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self.version = getattr(mp, "__version__", "unknown")
        self._blendshape_order_checked = False
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=int(cfg_landmarks.get("num_faces", 1)),
            min_face_detection_confidence=float(cfg_landmarks.get("min_face_detection_confidence", 0.3)),
            min_face_presence_confidence=float(cfg_landmarks.get("min_face_presence_confidence", 0.3)),
            min_tracking_confidence=float(cfg_landmarks.get("min_tracking_confidence", 0.3)),
            output_face_blendshapes=bool(cfg_landmarks.get("output_blendshapes", True)),
            output_facial_transformation_matrixes=bool(
                cfg_landmarks.get("output_transformation_matrix", True)),
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def describe(self) -> dict:
        return {
            "landmarker": "mediapipe.tasks.vision.FaceLandmarker",
            "mediapipe_version": self.version,
            "model_path": self.model_path,
            "num_landmarks_expected": MIN_LANDMARKS,
            "num_blendshapes": len(BLENDSHAPE_NAMES),
            "feature_dim": FEATURE_DIM,
        }

    def _blendshape_vector(self, categories) -> np.ndarray:
        by_name = {c.category_name: float(c.score) for c in categories}
        if not self._blendshape_order_checked:
            returned = set(by_name)
            expected = set(BLENDSHAPE_NAMES)
            if returned != expected:
                raise RuntimeError(
                    "MediaPipe returned a blendshape set that does not match the "
                    "canonical 52-name list this code indexes by.\n"
                    f"  missing from model : {sorted(expected - returned)}\n"
                    f"  unexpected extras  : {sorted(returned - expected)}\n"
                    "Update BLENDSHAPE_NAMES in src/landmarks.py rather than letting "
                    "the feature columns shift silently."
                )
            self._blendshape_order_checked = True
        return np.array([by_name[n] for n in BLENDSHAPE_NAMES], dtype=np.float32)

    def extract(self, frame_bgr: np.ndarray) -> LandmarkObservation:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect(image)

        face_landmarks = getattr(result, "face_landmarks", None) or []
        if not face_landmarks:
            return LandmarkObservation(False, None, None, None,
                                       np.zeros(FEATURE_DIM, dtype=np.float32))

        lm = face_landmarks[0]
        if len(lm) < MIN_LANDMARKS:
            raise RuntimeError(
                f"FaceLandmarker returned {len(lm)} landmarks; this code indexes iris "
                f"points and needs at least {MIN_LANDMARKS}. Check the model bundle."
            )
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

        head_pose = None
        mats = getattr(result, "facial_transformation_matrixes", None) or []
        if mats:
            head_pose = _euler_from_matrix(np.asarray(mats[0], dtype=np.float64))

        blend = None
        bs = getattr(result, "face_blendshapes", None) or []
        if bs:
            blend = self._blendshape_vector(bs[0])

        geo = compute_geometric_features(pts, head_pose, (w, h))
        vec = np.empty(FEATURE_DIM, dtype=np.float32)
        for i, name in enumerate(GEOMETRIC_FEATURE_NAMES):
            vec[i] = geo[name]
        offset = len(GEOMETRIC_FEATURE_NAMES)
        vec[offset:] = blend if blend is not None else 0.0

        if not np.all(np.isfinite(vec)):
            bad = [FEATURE_NAMES[i] for i in np.where(~np.isfinite(vec))[0]]
            raise RuntimeError(f"non-finite landmark features: {bad}")

        return LandmarkObservation(True, pts, blend, head_pose, vec)

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass


def aggregate_clip_features(per_frame: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """[T, L] -> [3L] via mean, std and mean absolute temporal difference.

    Spec section 17: the aggregation is simple and stated, and the resulting
    dimension is 3 * L, printed and recorded rather than assumed.
    """
    x = np.asarray(per_frame, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected [T, L], got {x.shape}")
    if valid_mask is not None and valid_mask.any():
        # Only average over frames that actually had a face, so a run of
        # undetected frames does not drag every statistic toward zero.
        sel = x[valid_mask.astype(bool)]
    else:
        sel = x
    if sel.shape[0] == 0:
        sel = x
    mean = sel.mean(axis=0)
    std = sel.std(axis=0)
    delta = np.abs(np.diff(x, axis=0)).mean(axis=0) if x.shape[0] > 1 else np.zeros_like(mean)
    return np.concatenate([mean, std, delta]).astype(np.float32)


CLIP_FEATURE_NAMES: tuple[str, ...] = (
    tuple(f"mean_{n}" for n in FEATURE_NAMES)
    + tuple(f"std_{n}" for n in FEATURE_NAMES)
    + tuple(f"delta_{n}" for n in FEATURE_NAMES)
)
CLIP_FEATURE_DIM = len(CLIP_FEATURE_NAMES)
