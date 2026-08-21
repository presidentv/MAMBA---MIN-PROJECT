"""Phase 4 input -- landmark-based geometric features.

The PDF's Phase 4 fuses the Vision Mamba embedding with "landmark-based
geometric features". Those come from the Phase 2 landmark tables and split into
two groups:

  * **behavioural descriptors** already engineered in ``src/features.py``
    (gaze direction, fixation proxy, dispersion, EAR/blink, off-screen, head
    stability) -- these correspond directly to the four behavioural categories
    the PDF names for Phase 3: eye gaze, blink patterns, head movement, facial
    dynamics;
  * **raw facial geometry** computed here: normalised inter-landmark distances
    and ratios that describe face shape and its per-frame deformation, which the
    gaze features do not capture.

Both are per-frame sequences, so they align 1:1 with the Vision Mamba frame
embeddings by ``sample_index``.

Nothing here re-runs MediaPipe; it reads the Phase 2 parquet only.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("mrgvm.geometric")

# Mesh landmark indices used for the geometric descriptors. All are stable
# points on the 468-point topology.
IDX_NOSE_TIP = 1
IDX_CHIN = 152
IDX_FOREHEAD = 10
IDX_LEFT_EYE_OUTER = 33
IDX_LEFT_EYE_INNER = 133
IDX_RIGHT_EYE_OUTER = 362
IDX_RIGHT_EYE_INNER = 263
IDX_MOUTH_LEFT = 61
IDX_MOUTH_RIGHT = 291
IDX_UPPER_LIP = 13
IDX_LOWER_LIP = 14
IDX_LEFT_BROW = 105
IDX_RIGHT_BROW = 334
IDX_LEFT_CHEEK = 234
IDX_RIGHT_CHEEK = 454

GEOMETRIC_FEATURE_COLUMNS: Tuple[str, ...] = (
    "geo_face_width",
    "geo_face_height",
    "geo_aspect_ratio",
    "geo_mouth_openness",
    "geo_mouth_width",
    "geo_brow_eye_left",
    "geo_brow_eye_right",
    "geo_brow_asymmetry",
    "geo_eye_openness_asymmetry",
    "geo_nose_chin_dist",
    "geo_landmark_energy",
    "geo_landmark_velocity",
)


def _point(frame: pd.DataFrame, index: int) -> np.ndarray:
    """Fetch the (N, 3) trajectory of one landmark from a Phase 2 table."""
    return frame[[f"lm_{index:03d}_x", f"lm_{index:03d}_y", f"lm_{index:03d}_z"]].to_numpy(
        dtype=np.float64
    )


def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, :2] - b[:, :2], axis=1)


def compute_geometric_features(landmarks: pd.DataFrame) -> pd.DataFrame:
    """Per-frame geometric descriptors for one clip's Phase 2 table.

    Every distance is divided by the inter-ocular distance of the same frame, so
    the features are invariant to residual scale differences between subjects
    and to any drift in the Phase 1 alignment.
    """
    df = landmarks.sort_values("sample_index").reset_index(drop=True)

    left_outer = _point(df, IDX_LEFT_EYE_OUTER)
    right_outer = _point(df, IDX_RIGHT_EYE_OUTER)
    interocular = _distance(left_outer, right_outer)
    scale = np.where(interocular > 1e-9, interocular, 1.0)

    cheek_left = _point(df, IDX_LEFT_CHEEK)
    cheek_right = _point(df, IDX_RIGHT_CHEEK)
    forehead = _point(df, IDX_FOREHEAD)
    chin = _point(df, IDX_CHIN)
    nose = _point(df, IDX_NOSE_TIP)
    mouth_left = _point(df, IDX_MOUTH_LEFT)
    mouth_right = _point(df, IDX_MOUTH_RIGHT)
    upper_lip = _point(df, IDX_UPPER_LIP)
    lower_lip = _point(df, IDX_LOWER_LIP)
    brow_left = _point(df, IDX_LEFT_BROW)
    brow_right = _point(df, IDX_RIGHT_BROW)
    eye_left_inner = _point(df, IDX_LEFT_EYE_INNER)
    eye_right_inner = _point(df, IDX_RIGHT_EYE_INNER)

    out = pd.DataFrame(index=range(len(df)))
    out["geo_face_width"] = _distance(cheek_left, cheek_right) / scale
    out["geo_face_height"] = _distance(forehead, chin) / scale
    out["geo_aspect_ratio"] = out["geo_face_height"] / np.maximum(out["geo_face_width"], 1e-9)
    out["geo_mouth_openness"] = _distance(upper_lip, lower_lip) / scale
    out["geo_mouth_width"] = _distance(mouth_left, mouth_right) / scale
    out["geo_brow_eye_left"] = _distance(brow_left, left_outer) / scale
    out["geo_brow_eye_right"] = _distance(brow_right, right_outer) / scale
    out["geo_brow_asymmetry"] = np.abs(out["geo_brow_eye_left"] - out["geo_brow_eye_right"])
    out["geo_nose_chin_dist"] = _distance(nose, chin) / scale

    left_eye_span = _distance(left_outer, eye_left_inner) / scale
    right_eye_span = _distance(right_outer, eye_right_inner) / scale
    out["geo_eye_openness_asymmetry"] = np.abs(left_eye_span - right_eye_span)

    # Whole-mesh deformation energy and inter-frame velocity: cheap proxies for
    # "facial dynamics" that need no per-region indexing.
    coordinate_columns = [c for c in df.columns if c.startswith("lm_") and c.endswith(("_x", "_y"))]
    mesh = df[coordinate_columns].to_numpy(dtype=np.float64)
    centred = mesh - mesh.mean(axis=0, keepdims=True)
    out["geo_landmark_energy"] = np.sqrt((centred**2).mean(axis=1)) / scale
    velocity = np.zeros(len(df))
    if len(df) > 1:
        velocity[1:] = np.linalg.norm(np.diff(mesh, axis=0), axis=1) / np.maximum(scale[1:], 1e-9)
    out["geo_landmark_velocity"] = velocity

    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_clip_geometry(
    landmark_path,
    gaze_path=None,
    gaze_columns: Sequence[str] = (),
) -> Tuple[pd.DataFrame, List[str]]:
    """Assemble the full per-frame geometric matrix for one clip.

    Returns ``(frame_table, column_names)`` where ``frame_table`` also carries
    ``sample_index`` and ``mrs`` so the caller can align with the frame images.
    """
    landmarks = pd.read_parquet(landmark_path)
    geometry = compute_geometric_features(landmarks)
    geometry.insert(0, "sample_index", landmarks.sort_values("sample_index")["sample_index"].to_numpy())
    geometry["mrs"] = landmarks.sort_values("sample_index")["mrs"].to_numpy()

    columns = list(GEOMETRIC_FEATURE_COLUMNS)
    if gaze_path is not None and gaze_columns:
        gaze = pd.read_parquet(gaze_path)
        keep = ["sample_index"] + [c for c in gaze_columns if c in gaze.columns]
        geometry = geometry.merge(gaze[keep], on="sample_index", how="inner", validate="1:1")
        columns += [c for c in gaze_columns if c in gaze.columns]

    return geometry, columns
