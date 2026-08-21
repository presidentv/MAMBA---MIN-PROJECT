"""Phase 2 -- facial landmark, iris and head-pose extraction.

Runs the MediaPipe Tasks FaceLandmarker over the reliable frames retained by
Phase 1 and writes one row per frame per clip.

It reads the ALIGNED CROPS rather than re-decoding the source video. That keeps
Phase 2 independent of the dataset (it only needs the Phase 1 output tree) and
avoids paying for video decoding twice. The consequence is that landmark
coordinates are normalised to the aligned crop; the Phase 1 affine is carried
into every row so any point can be projected back to original-frame pixels via
``alignment.invert_affine``, and the in-plane rotation removed by alignment is
added back so the reported roll refers to the original frame.

Outputs (all under --output-root):
    phase2_landmarks/<split>/<subject>/<clip>/landmarks.parquet
    phase2_manifest.csv
    logs/phase2.log, logs/phase2_warnings.csv, logs/phase2_config.json

Example:
    python pipeline/phase2_landmarks.py \
        --input-root Datasets/DAiSEE_Small \
        --output-root outputs
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engagement_pipeline import alignment, faces, io_utils, pose  # noqa: E402
from engagement_pipeline.config import Phase2Config, dump_config, load_config  # noqa: E402
from engagement_pipeline.dataset import LABEL_COLUMNS, label_lookup, load_labels  # noqa: E402
from engagement_pipeline.logging_utils import setup_logging  # noqa: E402
from engagement_pipeline.models import ensure_model  # noqa: E402

logger = logging.getLogger("phase2")

LANDMARKS_STEM = "landmarks"
CARRY_FROM_PHASE1 = (
    "mrs", "blur", "face_visibility", "head_rotation", "eye_visibility",
    "motion_consistency", "detection_confidence", "interocular_px",
    "align_rotation_deg", "align_scale",
    "affine_a00", "affine_a01", "affine_a02", "affine_a10", "affine_a11", "affine_a12",
)


def landmark_column_names(n_points: int = faces.NUM_MESH_LANDMARKS) -> List[str]:
    """``lm_000_x, lm_000_y, lm_000_z, lm_001_x, ...`` in a stable order."""
    names: List[str] = []
    for i in range(n_points):
        names.extend([f"lm_{i:03d}_x", f"lm_{i:03d}_y", f"lm_{i:03d}_z"])
    return names


def _derived_eye_features(points: np.ndarray) -> Dict[str, float]:
    """Named convenience columns so downstream code need not know index magic.

    Everything here is recomputable from the raw ``lm_*`` columns; it is
    materialised because the gaze-feature stage reads these on every frame.
    """
    left_iris = faces.iris_centre(points, faces.LEFT_IRIS_IDX)
    right_iris = faces.iris_centre(points, faces.RIGHT_IRIS_IDX)
    left_ear = faces.eye_aspect_ratio(points, faces.LEFT_EYE_EAR_IDX)
    right_ear = faces.eye_aspect_ratio(points, faces.RIGHT_EYE_EAR_IDX)

    features: Dict[str, float] = {
        "left_iris_x": float(left_iris[0]),
        "left_iris_y": float(left_iris[1]),
        "left_iris_z": float(left_iris[2]),
        "right_iris_x": float(right_iris[0]),
        "right_iris_y": float(right_iris[1]),
        "right_iris_z": float(right_iris[2]),
        "left_ear": float(left_ear),
        "right_ear": float(right_ear),
        "mean_ear": float((left_ear + right_ear) / 2.0),
    }

    # Iris position expressed inside its own eye socket: 0 = outer corner,
    # 1 = inner corner. This is the scale-free horizontal gaze proxy.
    for side, corners, iris in (
        ("left", faces.LEFT_EYE_CORNERS, left_iris),
        ("right", faces.RIGHT_EYE_CORNERS, right_iris),
    ):
        outer = points[corners[0]][:2]
        inner = points[corners[1]][:2]
        span = inner - outer
        width = float(np.linalg.norm(span))
        if width > 1e-9:
            offset = iris[:2] - outer
            unit = span / width
            rel = float(offset @ span / (width * width))
            # 2-D scalar cross product (numpy 2.x dropped the 2-vector form of
            # np.cross): signed perpendicular offset of the iris from the eye axis.
            perp = float(unit[0] * offset[1] - unit[1] * offset[0]) / width
        else:
            rel, perp = float("nan"), float("nan")
        features[f"{side}_iris_rel_x"] = float(rel)
        features[f"{side}_iris_rel_y"] = float(perp)
        features[f"{side}_eye_width"] = width
    return features


def process_clip(
    clip_dir_phase1: Path,
    split: str,
    subject_id: str,
    clip_id: str,
    cfg: Phase2Config,
    output_root: Path,
    landmarker: faces.FaceLandmarkerWrapper,
    labels: Dict[str, object],
) -> Dict[str, object]:
    """Extract landmarks for every retained frame of one clip."""
    started = time.perf_counter()
    scores_path = io_utils.find_table(clip_dir_phase1, "mrs_scores")
    if scores_path is None:
        logger.error("No Phase 1 score table in %s", clip_dir_phase1)
        return _manifest_row(clip_id, subject_id, split, 0, 0, 0, started, "missing_phase1", labels)

    scores = io_utils.read_table(scores_path)
    retained = scores[scores["retained"].astype(bool)].sort_values("sample_index")
    if retained.empty:
        logger.warning("No retained frames for %s/%s/%s", split, subject_id, clip_id)
        return _manifest_row(clip_id, subject_id, split, 0, 0, 0, started, "no_retained_frames", labels)

    column_names = landmark_column_names()
    rows: List[dict] = []
    previous_points: Optional[np.ndarray] = None
    previous_sample: Optional[int] = None
    track_id = 0
    track_breaks = 0
    missed = 0

    for record in retained.to_dict(orient="records"):
        image_path = clip_dir_phase1 / str(record["frame_file"]).replace("/", "\\")
        if not image_path.is_file():
            image_path = clip_dir_phase1 / str(record["frame_file"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Could not read aligned frame %s", image_path)
            missed += 1
            continue

        mesh = landmarker.detect(image)
        if mesh is None:
            # A face that cleared the MRS gate but that the mesh cannot fit is a
            # genuine anomaly, so it is counted and reported rather than hidden.
            missed += 1
            previous_points = None
            continue

        height, width = image.shape[:2]
        points = mesh.points

        # --- head pose ----------------------------------------------------- #
        matrix_pose = pose.pose_from_transformation_matrix(mesh.transformation_matrix)
        pnp_pose = pose.pose_from_mesh(points, width, height)
        align_rotation = float(record.get("align_rotation_deg", 0.0) or 0.0)

        # --- tracking ------------------------------------------------------ #
        displacement = float("nan")
        broke = False
        if previous_points is not None:
            displacement = float(
                np.mean(np.linalg.norm(points[:, :2] - previous_points[:, :2], axis=1))
            )
            gap = int(record["sample_index"]) - int(previous_sample or 0)
            broke = displacement > cfg.track_break_threshold or gap > 1
        elif rows:
            broke = True
        if broke:
            track_id += 1
            track_breaks += 1

        row: Dict[str, object] = {
            "ClipID": clip_id,
            "SubjectID": subject_id,
            "split": split,
            "sample_index": int(record["sample_index"]),
            "frame_index": int(record["frame_index"]),
            "timestamp": float(record["timestamp"]),
            "n_landmarks": int(points.shape[0]),
            "track_id": track_id,
            "landmark_displacement": displacement,
            "track_break": bool(broke),
            # Pose in the ALIGNED CROP frame (roll is ~0 there by construction).
            "yaw": matrix_pose.yaw,
            "pitch": matrix_pose.pitch,
            "roll_aligned": matrix_pose.roll,
            # Roll referred back to the ORIGINAL video frame.
            "roll": matrix_pose.roll + align_rotation,
            "pose_source": "mesh_transformation_matrix",
            # Independent solvePnP estimate, kept as a cross-check.
            "pnp_yaw": pnp_pose.yaw,
            "pnp_pitch": pnp_pose.pitch,
            "pnp_roll": pnp_pose.roll,
            "pnp_reprojection_error": pnp_pose.reprojection_error,
        }
        row.update(_derived_eye_features(points))
        row.update(dict(zip(column_names, points.reshape(-1).astype(np.float32))))
        for key in CARRY_FROM_PHASE1:
            if key in record:
                row[key] = record[key]
        row.update(labels)
        rows.append(row)

        previous_points = points
        previous_sample = int(record["sample_index"])

    if not rows:
        logger.error("Mesh failed on every retained frame of %s/%s", subject_id, clip_id)
        return _manifest_row(
            clip_id, subject_id, split, len(retained), 0, missed, started, "mesh_failed", labels
        )

    out_dir = io_utils.ensure_dir(
        io_utils.clip_output_dir(output_root / "phase2_landmarks", split, subject_id, clip_id)
    )
    frame_df = pd.DataFrame(rows)
    written = io_utils.write_table(frame_df, out_dir / LANDMARKS_STEM)
    logger.info(
        "%s/%s/%s: %d/%d frames landmarked, %d track break(s) -> %s",
        split, subject_id, clip_id, len(rows), len(retained), track_breaks, written.name,
    )
    return _manifest_row(
        clip_id, subject_id, split, len(retained), len(rows), missed, started, "ok", labels,
        track_breaks=track_breaks,
        landmarks_file=str(written.relative_to(output_root)),
        mean_ear=float(frame_df["mean_ear"].mean()),
        mean_yaw=float(frame_df["yaw"].mean()),
    )


def _manifest_row(
    clip_id: str, subject_id: str, split: str, retained: int, landmarked: int,
    missed: int, started: float, status: str, labels: Dict[str, object],
    track_breaks: int = 0, landmarks_file: str = "",
    mean_ear: float = float("nan"), mean_yaw: float = float("nan"),
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "ClipID": clip_id,
        "SubjectID": subject_id,
        "split": split,
        "status": status,
        "frames_retained_phase1": retained,
        "frames_landmarked": landmarked,
        "frames_mesh_failed": missed,
        "landmark_success_ratio": landmarked / retained if retained else 0.0,
        "track_breaks": track_breaks,
        "mean_ear": mean_ear,
        "mean_yaw": mean_yaw,
        "landmarks_file": landmarks_file,
        "seconds": round(time.perf_counter() - started, 3),
    }
    row.update(labels)
    row["has_label"] = bool(labels) and labels.get("Engagement") is not None
    return row


def discover_phase1_clips(phase1_root: Path, splits) -> List[tuple]:
    """Walk the Phase 1 output tree -- the same layout-driven discovery as Phase 1."""
    found: List[tuple] = []
    for split in splits:
        split_dir = phase1_root / split
        if not split_dir.is_dir():
            logger.warning("No Phase 1 output for split %s at %s", split, split_dir)
            continue
        for subject_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for clip_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
                found.append((clip_dir, split, subject_dir.name, clip_dir.name))
    return found


def run(input_root: Path, output_root: Path, cfg: Phase2Config, model_dir: Optional[Path]) -> pd.DataFrame:
    output_root = io_utils.ensure_dir(Path(output_root))
    logs_dir = io_utils.ensure_dir(output_root / "logs")
    dump_config(cfg, logs_dir / "phase2_config.json")

    phase1_root = output_root / "phase1_reliable_frames"
    if not phase1_root.is_dir():
        raise SystemExit(
            f"Phase 1 output not found at {phase1_root}. Run phase1_preprocessing.py first."
        )

    clips = discover_phase1_clips(phase1_root, cfg.splits)
    if cfg.limit_clips:
        clips = clips[: cfg.limit_clips]
    if not clips:
        raise SystemExit(f"No Phase 1 clip directories under {phase1_root}")
    logger.info("Found %d Phase 1 clip directories", len(clips))

    lookup = label_lookup(load_labels(input_root, cfg.splits))

    landmarker = faces.FaceLandmarkerWrapper(
        ensure_model("face_landmarker", model_dir),
        cfg.min_face_detection_confidence,
        cfg.min_face_presence_confidence,
        cfg.min_tracking_confidence,
        cfg.output_blendshapes,
    )

    manifest_rows: List[Dict[str, object]] = []
    try:
        for index, (clip_dir, split, subject_id, clip_id) in enumerate(clips, start=1):
            logger.info("[%d/%d] %s/%s/%s", index, len(clips), split, subject_id, clip_id)
            labels = lookup.get((split, clip_id), {c: None for c in LABEL_COLUMNS})
            try:
                row = process_clip(
                    clip_dir, split, subject_id, clip_id, cfg, output_root, landmarker, labels
                )
            except Exception:
                logger.exception("Clip failed: %s/%s/%s", split, subject_id, clip_id)
                row = _manifest_row(clip_id, subject_id, split, 0, 0, 0, time.perf_counter(), "error", labels)
            manifest_rows.append(row)
    finally:
        landmarker.close()

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_root / "phase2_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    logger.info("Manifest written: %s", manifest_path)

    warnings = manifest[
        (manifest["status"] != "ok")
        | (manifest["landmark_success_ratio"] < 1.0)
        | (~manifest["has_label"])
    ]
    warnings.to_csv(logs_dir / "phase2_warnings.csv", index=False)

    logger.info("=" * 78)
    logger.info("PHASE 2 SUMMARY")
    logger.info("  clips processed          : %d", len(manifest))
    logger.info("  clips ok                 : %d", int((manifest["status"] == "ok").sum()))
    logger.info("  frames landmarked (total): %d", int(manifest["frames_landmarked"].sum()))
    logger.info("  frames mesh-failed       : %d", int(manifest["frames_mesh_failed"].sum()))
    logger.info("  clips missing a label    : %d", int((~manifest["has_label"]).sum()))
    logger.info("  total track breaks       : %d", int(manifest["track_breaks"].sum()))
    for split, group in manifest.groupby("split"):
        logger.info(
            "  %-11s: %2d clips, %4d frames, success %.1f%%",
            split, len(group), int(group["frames_landmarked"].sum()),
            100 * float(group["landmark_success_ratio"].mean()),
        )
    if len(warnings):
        logger.warning("  %d clip(s) flagged -> logs/phase2_warnings.csv", len(warnings))
    logger.info("=" * 78)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-root", type=Path, required=True, help="DAiSEE root (for Labels/).")
    p.add_argument("--output-root", type=Path, required=True, help="Same --output-root as Phase 1.")
    p.add_argument("--config", type=Path, default=None, help="JSON config file (phase2 section).")
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--min-face-detection-confidence", type=float, default=None)
    p.add_argument("--track-break-threshold", type=float, default=None)
    p.add_argument("--output-blendshapes", action="store_true", default=None,
                   help="Also store the 52 ARKit blendshape scores per frame.")
    p.add_argument("--limit-clips", type=int, default=None)
    p.add_argument("--splits", nargs="+", default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(
        Phase2Config,
        args.config,
        section="phase2",
        min_face_detection_confidence=args.min_face_detection_confidence,
        track_break_threshold=args.track_break_threshold,
        output_blendshapes=args.output_blendshapes,
        limit_clips=args.limit_clips,
        splits=tuple(args.splits) if args.splits else None,
    )
    output_root = Path(args.output_root)
    io_utils.ensure_dir(output_root / "logs")
    setup_logging(output_root / "logs" / "phase2.log")
    logger.info("Phase 2 configuration: %s", cfg)
    run(Path(args.input_root), output_root, cfg, args.model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
