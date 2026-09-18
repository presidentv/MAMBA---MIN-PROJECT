"""Motion Reliability Score (MRS).

MRS answers "how much should I trust what I can see in this frame?", not "how
engaged is this student?". It is computed purely from observation quality and
never touches the engagement label (spec Rules 9 and 10).

    MRS_t = w_b*B_t + w_f*F_t + w_h*H_t + w_e*E_t + w_m*M_t

with every component in [0, 1] and the weights summing to 1. The baseline
0.2 each is a starting point, not a tuned or optimal setting.

Two deliberate design choices, both from spec section 10/11:

* **A closed eye is not unreliable.** Blinking is normal behaviour, so E scores
  whether the eye region was *localisable*, not whether it was open. Penalising
  low eye-aspect-ratio here would leak behaviour into the reliability signal.
* **A turned head is not disengagement.** H says the frontal appearance evidence
  is weaker, nothing more.

Low-MRS frames are never deleted. They keep their temporal position and are
down-weighted (spec section 11), which is what preserves the sequence for Mamba.

**Calibration.** Two of the five signals -- blur sharpness and face size -- have
no dataset-independent scale. A variance-of-Laplacian of 40 means nothing until
you know what this camera and this framing usually produce, and "a face filling
15% of the frame" is arbitrary for a seated webcam recording. Both are therefore
mapped through percentiles fitted on the **training split only** (Rule 8), by
scripts/fit_mrs_stats.py. Hard-coded thresholds would have made both components
close to constant on DAiSEE, wasting two fifths of the score's dynamic range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import cv2
import numpy as np

COMPONENTS = ("blur", "face_visibility", "head_pose", "eye_visibility", "motion_consistency")


@dataclass
class MRSResult:
    mrs: float
    blur: float
    face_visibility: float
    head_pose: float
    eye_visibility: float
    motion_consistency: float
    # Provenance, so a value of 1.0 that means "no evidence" can be told apart
    # from a value of 1.0 that means "measured and perfect".
    blur_raw: float | None = None
    face_area_fraction: float | None = None
    motion_defined: bool = True
    motion_source: str = "landmark_displacement"

    def as_dict(self) -> dict:
        return asdict(self)

    def component_vector(self) -> np.ndarray:
        return np.array([self.blur, self.face_visibility, self.head_pose,
                         self.eye_visibility, self.motion_consistency], dtype=np.float32)


class MRSCalibration:
    """Percentile mappings for the two scale-dependent MRS signals.

    Blur is mapped in log space because variance-of-Laplacian spans orders of
    magnitude across clips; a linear map would pile almost every frame at one end.
    Face area is mapped in linear space on the area *fraction*.

    Both are fitted on training frames only.
    """

    def __init__(self, blur_log_low: float, blur_log_high: float,
                 area_low: float, area_high: float,
                 calibrated: bool = True,
                 percentiles: tuple[float, float] = (5.0, 95.0),
                 n_frames: int | None = None):
        if not blur_log_high > blur_log_low:
            raise ValueError(f"blur bounds need high > low, got {blur_log_low}/{blur_log_high}")
        if not area_high > area_low:
            raise ValueError(f"area bounds need high > low, got {area_low}/{area_high}")
        self.blur_log_low = float(blur_log_low)
        self.blur_log_high = float(blur_log_high)
        self.area_low = float(area_low)
        self.area_high = float(area_high)
        self.calibrated = bool(calibrated)
        self.percentiles = tuple(percentiles)
        self.n_frames = n_frames

    # ------------------------------------------------------------ construction
    @classmethod
    def uncalibrated(cls) -> "MRSCalibration":
        """Fallback used only before scripts/fit_mrs_stats.py has run.

        The blur bounds are the conventional OpenCV rule-of-thumb range from
        "obviously blurred" to "sharp"; the area bounds are a generic webcam
        guess. Anything produced with this is flagged calibrated=False so it can
        never be mistaken for a fitted mapping.
        """
        return cls(math.log(11.0), math.log(501.0), 0.01, 0.15, calibrated=False)

    @classmethod
    def fit(cls, blur_raw, area_fractions,
            percentiles: tuple[float, float] = (5.0, 95.0)) -> "MRSCalibration":
        blur = np.asarray([v for v in np.ravel(blur_raw) if np.isfinite(v)], dtype=np.float64)
        area = np.asarray([v for v in np.ravel(area_fractions) if np.isfinite(v)], dtype=np.float64)
        if blur.size < 10 or area.size < 10:
            raise ValueError(f"need >= 10 samples to fit; got blur={blur.size} area={area.size}")

        logs = np.log(np.maximum(blur, 0.0) + 1.0)
        b_low, b_high = np.percentile(logs, percentiles)
        a_low, a_high = np.percentile(area, percentiles)

        if b_high - b_low < 1e-6:
            raise ValueError("training blur values are nearly constant; B would be "
                             "uninformative. Inspect artifacts/mrs_calibration.json.")
        if a_high - a_low < 1e-9:
            # A fixed-camera corpus can genuinely have near-constant face size.
            # Widen slightly rather than divide by zero, and say so.
            a_high = a_low + 1e-6
        return cls(float(b_low), float(b_high), float(a_low), float(a_high),
                   True, percentiles, int(blur.size))

    # ------------------------------------------------------------------ scoring
    def blur_score(self, raw: float) -> float:
        x = math.log(max(float(raw), 0.0) + 1.0)
        return float(np.clip((x - self.blur_log_low) /
                             (self.blur_log_high - self.blur_log_low), 0.0, 1.0))

    def blur_score_array(self, raw: np.ndarray) -> np.ndarray:
        x = np.log(np.maximum(np.asarray(raw, dtype=np.float64), 0.0) + 1.0)
        return np.clip((x - self.blur_log_low) / (self.blur_log_high - self.blur_log_low), 0.0, 1.0)

    def area_score(self, fraction: float) -> float:
        return float(np.clip((float(fraction) - self.area_low) /
                             (self.area_high - self.area_low), 0.0, 1.0))

    def area_score_array(self, fraction: np.ndarray) -> np.ndarray:
        f = np.asarray(fraction, dtype=np.float64)
        return np.clip((f - self.area_low) / (self.area_high - self.area_low), 0.0, 1.0)

    # -------------------------------------------------------------- (de)serialise
    def to_dict(self) -> dict:
        return {"blur_log_low": self.blur_log_low, "blur_log_high": self.blur_log_high,
                "area_low": self.area_low, "area_high": self.area_high,
                "calibrated": self.calibrated, "percentiles": list(self.percentiles),
                "n_frames": self.n_frames}

    @classmethod
    def from_dict(cls, d: dict) -> "MRSCalibration":
        return cls(d["blur_log_low"], d["blur_log_high"], d["area_low"], d["area_high"],
                   d.get("calibrated", True), tuple(d.get("percentiles", (5.0, 95.0))),
                   d.get("n_frames"))


def laplacian_variance(face_crop_bgr: np.ndarray, contrast_normalised: bool = False) -> float:
    """Sharpness proxy: variance of the Laplacian on the face crop.

    ``contrast_normalised=True`` divides by the intensity variance, which is
    algebraically identical to standardising the crop before the Laplacian
    (the operator is linear and annihilates constants, so
    Var(Lap((I-mu)/sigma)) == Var(Lap(I)) / Var(I)).

    It defaults to **False**. The normalisation does what it claims -- on real
    DAiSEE frames it drops the correlation between the blur score and image
    contrast from r = 0.505 to r = -0.016 -- but measured end to end against
    known degradation severity it made the score worse overall, taking MRS from
    rho = -0.862 to -0.762. Two failures outside its reach are why:

      * darkening re-quantises to 8 bits and leaves a high-frequency noise floor
        in the Laplacian. Var(I) falls 9.5x while Var(Lap) falls only 5.8x, so
        the ratio *inflates* and a dark frame reads as sharper (1.61x clean).
      * an occluder adds hard edges while being internally uniform, raising the
        numerator and lowering the denominator at once (2.06x clean).

    Neither is a contrast-scaling effect, so rescaling cannot fix them. The
    option is kept because the measurement is real and a future blur term may
    want it; see scripts/test_mrs.py and the README for the numbers.
    """
    gray = (cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
            if face_crop_bgr.ndim == 3 else face_crop_bgr)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if not contrast_normalised:
        return lap_var
    intensity_var = float(np.asarray(gray, dtype=np.float64).var())
    return 100.0 * lap_var / (intensity_var + 1e-6)


# --------------------------------------------------------------------------- #
# Individual components
# --------------------------------------------------------------------------- #
def face_visibility_score(detector_confidence: float | None, area_fraction: float,
                          calibration: MRSCalibration) -> float:
    """Detector confidence combined with how much of the frame the face fills.

    A tiny face and a large clear face must not score the same (spec 10.2), so
    the two terms are multiplied: high confidence on a 20-pixel face still gives
    a low reliability. A frame with no detection at all has confidence None and
    scores 0.
    """
    if detector_confidence is None or not np.isfinite(detector_confidence):
        return 0.0
    conf = float(np.clip(detector_confidence, 0.0, 1.0))
    return float(conf * calibration.area_score(area_fraction))


def head_pose_score(head_pose_deg, cfg_mrs: dict) -> float:
    """Reliability of the frontal appearance evidence given head rotation.

    Roll is excluded because the crop is roll-aligned, so it costs no appearance
    information. Yaw and pitch do occlude facial structure, so the worse of the
    two drives the score.

    This is explicitly NOT an engagement signal (spec 10.3).
    """
    if head_pose_deg is None:
        return 0.5   # unknown pose: neither trusted nor discarded
    yaw, pitch = float(head_pose_deg[0]), float(head_pose_deg[1])
    if not (math.isfinite(yaw) and math.isfinite(pitch)):
        return 0.5
    worst = max(abs(yaw), abs(pitch))
    full = float(cfg_mrs["head_pose_full_reliability_deg"])
    zero = float(cfg_mrs["head_pose_zero_reliability_deg"])
    if worst <= full:
        return 1.0
    if worst >= zero:
        return 0.0
    return float(1.0 - (worst - full) / (zero - full))


def eye_visibility_score(landmark_obs) -> float:
    """Whether the eye and iris regions were successfully localised.

    Deliberately independent of eye openness: a blink is valid behaviour, not a
    bad observation (spec 10.4). What lowers this score is the eye region falling
    outside the frame or the iris points not being resolved.
    """
    from .landmarks import LEFT_EYE, LEFT_IRIS_CENTER, RIGHT_EYE, RIGHT_IRIS_CENTER

    if landmark_obs is None or not landmark_obs.found or landmark_obs.landmarks is None:
        return 0.0
    pts = landmark_obs.landmarks
    idx = [RIGHT_EYE["outer"], RIGHT_EYE["inner"], LEFT_EYE["outer"], LEFT_EYE["inner"],
           RIGHT_IRIS_CENTER, LEFT_IRIS_CENTER]
    if pts.shape[0] <= max(idx):
        return 0.0
    sel = pts[idx, :2]
    if not np.all(np.isfinite(sel)):
        return 0.0
    inside = np.logical_and(np.all(sel >= 0.0, axis=1), np.all(sel <= 1.0, axis=1))
    in_frame = float(inside.mean())

    # Iris resolved distinctly from the eye centre (a degenerate iris fit
    # collapses onto the eye centre and carries no gaze information).
    r_c = (pts[RIGHT_EYE["outer"], :2] + pts[RIGHT_EYE["inner"], :2]) / 2.0
    l_c = (pts[LEFT_EYE["outer"], :2] + pts[LEFT_EYE["inner"], :2]) / 2.0
    r_w = float(np.linalg.norm(pts[RIGHT_EYE["outer"], :2] - pts[RIGHT_EYE["inner"], :2]))
    l_w = float(np.linalg.norm(pts[LEFT_EYE["outer"], :2] - pts[LEFT_EYE["inner"], :2]))
    iris_ok = 0.0
    for c, w, iris in ((r_c, r_w, pts[RIGHT_IRIS_CENTER, :2]), (l_c, l_w, pts[LEFT_IRIS_CENTER, :2])):
        if w > 1e-6 and np.linalg.norm(iris - c) / w < 1.5:
            iris_ok += 0.5
    return float(np.clip(0.5 * in_frame + 0.5 * iris_ok, 0.0, 1.0))


def motion_consistency_score(landmark_obs, prev_landmark_obs,
                             face_crop_gray: np.ndarray | None,
                             prev_face_crop_gray: np.ndarray | None,
                             dt_seconds: float,
                             cfg_mrs: dict,
                             detection_found: bool | None = None) -> tuple[float, bool, str]:
    """How stable the observation is relative to the previous sampled frame.

    Primary measure: median landmark displacement between the two frames,
    normalised by inter-ocular distance and by the elapsed time, giving a
    face-widths-per-second rate that is comparable across different sampling
    rates T.

    Fallback when landmarks are missing on either side: zero-mean normalised
    cross-correlation (ZNCC) of the grayscale face crops. ZNCC is used rather
    than a raw pixel difference precisely because raw differences respond to
    lighting and global camera shifts as strongly as to subject motion
    (spec 10.5).

    Returns (score, defined, source). ``defined`` is False for the first frame
    of a clip, where 1.0 means "no evidence of inconsistency" rather than a
    measurement.
    """
    if prev_landmark_obs is None and prev_face_crop_gray is None:
        return 1.0, False, "first_frame"

    # No face located at all: the crop is a fallback region, not a face, so there
    # is no motion to be consistent about. The ZNCC fallback below would compare
    # two near-uniform fallback crops and return a high correlation, which is how
    # heavy occlusion previously RAISED motion reliability from 0.447 to 0.790 --
    # exactly backwards. Absence of a face is unreliable by definition.
    if detection_found is False:
        return 0.0, True, "no_face_detected"

    ref = float(cfg_mrs["motion_ref_displacement"])
    dt = max(float(dt_seconds), 1e-3)

    if (landmark_obs is not None and landmark_obs.found and landmark_obs.landmarks is not None
            and prev_landmark_obs is not None and prev_landmark_obs.found
            and prev_landmark_obs.landmarks is not None):
        from .landmarks import LEFT_EYE, RIGHT_EYE

        a = landmark_obs.landmarks[:, :2]
        b = prev_landmark_obs.landmarks[:, :2]
        n = min(a.shape[0], b.shape[0])
        r_c = (a[RIGHT_EYE["outer"]] + a[RIGHT_EYE["inner"]]) / 2.0
        l_c = (a[LEFT_EYE["outer"]] + a[LEFT_EYE["inner"]]) / 2.0
        iod = float(np.linalg.norm(r_c - l_c))
        if iod > 1e-6:
            disp = np.linalg.norm(a[:n] - b[:n], axis=1)
            rate = float(np.median(disp) / iod / dt)
            return float(np.clip(1.0 - rate / ref, 0.0, 1.0)), True, "landmark_displacement"

    if face_crop_gray is not None and prev_face_crop_gray is not None:
        a = face_crop_gray.astype(np.float32)
        b = prev_face_crop_gray.astype(np.float32)
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom > 1e-6:
            return float(np.clip(float((a * b).sum() / denom), 0.0, 1.0)), True, "zncc_face_crop"

    return 0.5, False, "unmeasurable"


# --------------------------------------------------------------------------- #
# Raw measurement / combination
# --------------------------------------------------------------------------- #
def compute_raw_components(face_crop_bgr: np.ndarray,
                           detection,
                           landmark_obs,
                           prev_landmark_obs,
                           prev_face_crop_bgr: np.ndarray | None,
                           dt_seconds: float,
                           cfg_mrs: dict) -> dict:
    """Measure the reliability signals in their *uncalibrated* form.

    Blur is left as raw variance-of-Laplacian and face size as a raw area
    fraction, because both need percentiles that are not known until the
    training split has been swept. Head pose is left in degrees. Caching this
    raw form means re-fitting the calibration, or changing the MRS weights,
    costs nothing -- no video is decoded again and no ViT pass is repeated.

    Every input is an observation-quality input. No engagement label, prediction,
    or label-derived statistic is involved (Rules 9 and 10).
    """
    gray = (cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
            if face_crop_bgr.ndim == 3 else face_crop_bgr)
    prev_gray = None
    if prev_face_crop_bgr is not None:
        prev_gray = (cv2.cvtColor(prev_face_crop_bgr, cv2.COLOR_BGR2GRAY)
                     if prev_face_crop_bgr.ndim == 3 else prev_face_crop_bgr)

    m, m_defined, m_source = motion_consistency_score(
        landmark_obs, prev_landmark_obs, gray, prev_gray, dt_seconds, cfg_mrs,
        detection_found=(detection.found if detection is not None else None))

    head = landmark_obs.head_pose_deg if landmark_obs is not None else None
    return {
        "blur_raw": laplacian_variance(face_crop_bgr),
        "face_area_fraction": float(detection.area_fraction) if detection is not None else 0.0,
        "detector_confidence": (float(detection.confidence)
                                if (detection is not None and detection.confidence is not None)
                                else float("nan")),
        "head_pose_deg": tuple(float(v) for v in head) if head is not None else (np.nan,) * 3,
        "eye_visibility": eye_visibility_score(landmark_obs),
        "motion_consistency": m,
        "motion_defined": m_defined,
        "motion_source": m_source,
    }


def combine_components(raw: dict, cfg_mrs: dict, calibration: MRSCalibration) -> MRSResult:
    """Turn raw measurements into the weighted MRS, validating every range."""
    weights = cfg_mrs["weights"]
    total_w = sum(float(weights[c]) for c in COMPONENTS)
    if total_w <= 0:
        raise ValueError("MRS weights must sum to a positive value")

    conf = raw.get("detector_confidence")
    parts = {
        "blur": calibration.blur_score(raw["blur_raw"]),
        "face_visibility": face_visibility_score(conf, raw["face_area_fraction"], calibration),
        "head_pose": head_pose_score(raw.get("head_pose_deg"), cfg_mrs),
        "eye_visibility": float(raw["eye_visibility"]),
        "motion_consistency": float(raw["motion_consistency"]),
    }
    for name, value in parts.items():
        if not math.isfinite(value) or not (-1e-6 <= value <= 1.0 + 1e-6):
            raise RuntimeError(f"MRS component {name} out of range: {value}")

    mrs = sum(float(weights[c]) * parts[c] for c in COMPONENTS) / total_w
    return MRSResult(
        mrs=float(np.clip(mrs, 0.0, 1.0)),
        blur_raw=raw.get("blur_raw"),
        face_area_fraction=raw.get("face_area_fraction"),
        motion_defined=bool(raw.get("motion_defined", True)),
        motion_source=str(raw.get("motion_source", "landmark_displacement")),
        **parts,
    )


def compute_mrs(face_crop_bgr, detection, landmark_obs, prev_landmark_obs,
                prev_face_crop_bgr, dt_seconds, cfg_mrs, calibration) -> MRSResult:
    """Single-shot convenience wrapper: measure, then combine."""
    raw = compute_raw_components(face_crop_bgr, detection, landmark_obs, prev_landmark_obs,
                                 prev_face_crop_bgr, dt_seconds, cfg_mrs)
    return combine_components(raw, cfg_mrs, calibration)


def components_from_arrays(blur_raw, face_area_fraction, detector_confidence,
                           head_pose_deg, eye_visibility, motion_consistency,
                           cfg_mrs: dict, calibration: MRSCalibration) -> dict[str, np.ndarray]:
    """Vectorised per-component scoring for a whole cached clip: [T] -> [T] each."""
    conf = np.asarray(detector_confidence, dtype=np.float64)
    area = np.asarray(face_area_fraction, dtype=np.float64)
    f = np.where(np.isfinite(conf), np.clip(conf, 0.0, 1.0), 0.0) * calibration.area_score_array(area)

    pose = np.asarray(head_pose_deg, dtype=np.float64)
    worst = np.max(np.abs(pose[:, :2]), axis=1)
    full = float(cfg_mrs["head_pose_full_reliability_deg"])
    zero = float(cfg_mrs["head_pose_zero_reliability_deg"])
    h = np.clip(1.0 - (worst - full) / (zero - full), 0.0, 1.0)
    h = np.where(np.isfinite(worst), h, 0.5)

    return {
        "blur": calibration.blur_score_array(blur_raw),
        "face_visibility": f,
        "head_pose": h,
        "eye_visibility": np.asarray(eye_visibility, dtype=np.float64),
        "motion_consistency": np.asarray(motion_consistency, dtype=np.float64),
    }


def mrs_from_arrays(blur_raw, face_area_fraction, detector_confidence, head_pose_deg,
                    eye_visibility, motion_consistency, cfg_mrs: dict,
                    calibration: MRSCalibration) -> np.ndarray:
    """Vectorised combine for a whole cached clip: [T] arrays -> MRS [T]."""
    weights = cfg_mrs["weights"]
    total_w = sum(float(weights[c]) for c in COMPONENTS)
    parts = components_from_arrays(blur_raw, face_area_fraction, detector_confidence,
                                   head_pose_deg, eye_visibility, motion_consistency,
                                   cfg_mrs, calibration)
    mrs = sum(float(weights[c]) * parts[c] for c in COMPONENTS) / total_w
    return np.clip(mrs, 0.0, 1.0).astype(np.float32)


def component_matrix_from_arrays(blur_raw, face_area_fraction, detector_confidence,
                                 head_pose_deg, eye_visibility, motion_consistency,
                                 cfg_mrs: dict, calibration: MRSCalibration) -> np.ndarray:
    """[T] raw arrays -> [T, 5] calibrated components in COMPONENTS order.

    This is what a learnable-weight model consumes: the five components stay
    label-free and fixed, and only their combination is learned.
    """
    parts = components_from_arrays(blur_raw, face_area_fraction, detector_confidence,
                                   head_pose_deg, eye_visibility, motion_consistency,
                                   cfg_mrs, calibration)
    return np.stack([parts[c] for c in COMPONENTS], axis=-1).astype(np.float32)
