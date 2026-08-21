"""Phase 1 -- video preprocessing and Motion Reliability Score (MRS).

For every discovered clip this script:
  1. samples frames at a configurable rate (default 5 fps),
  2. detects the subject's face (MediaPipe Tasks BlazeFace) and tracks it across
     frames so bystanders cannot hijack the clip,
  3. aligns the face to a canonical eye line and normalises it to a fixed size,
  4. scores each frame with the five MRS components, and
  5. keeps only frames whose MRS clears the threshold.

Outputs (all under --output-root):
    phase1_reliable_frames/<split>/<subject>/<clip>/frames/*.png   retained crops
    phase1_reliable_frames/<split>/<subject>/<clip>/mrs_scores.parquet
    phase1_manifest.csv          one row per clip
    logs/phase1.log              full run log
    logs/phase1_warnings.csv     clips needing attention
    logs/phase1_config.json      the fully-resolved configuration

Example:
    python pipeline/phase1_preprocessing.py \
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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engagement_pipeline import alignment, faces, io_utils, mrs, pose, video  # noqa: E402
from engagement_pipeline.config import Phase1Config, dump_config, load_config  # noqa: E402
from engagement_pipeline.dataset import (  # noqa: E402
    ClipRecord,
    LABEL_COLUMNS,
    discover_clips,
    label_lookup,
    load_labels,
)
from engagement_pipeline.logging_utils import setup_logging  # noqa: E402
from engagement_pipeline.models import ensure_model  # noqa: E402

logger = logging.getLogger("phase1")

FRAMES_SUBDIR = "frames"
SCORES_STEM = "mrs_scores"


# --------------------------------------------------------------------------- #
# Per-clip processing
# --------------------------------------------------------------------------- #
def process_clip(
    record: ClipRecord,
    cfg: Phase1Config,
    output_root: Path,
    detector: faces.FaceDetectorWrapper,
    landmarker: Optional[faces.FaceLandmarkerWrapper],
) -> Dict[str, object]:
    """Run the full Phase 1 pipeline for one clip and return its manifest row."""
    started = time.perf_counter()
    clip_dir = io_utils.clip_output_dir(
        output_root / "phase1_reliable_frames", record.split, record.subject_id, record.clip_id
    )
    frames_dir = io_utils.ensure_dir(clip_dir / FRAMES_SUBDIR)

    if cfg.overwrite:
        for stale in frames_dir.glob("*"):
            if stale.is_file():
                stale.unlink()

    sampled, meta = video.read_sampled_frames(
        record.video_path, cfg.sample_fps, cfg.max_frames_per_clip
    )
    if not sampled:
        logger.error("No frames decoded from %s", record.video_path)
        return _manifest_row(
            record, cfg, meta, rows=[], retained=0, duration=time.perf_counter() - started,
            status="decode_failed",
        )

    rows: List[dict] = []
    previous_aligned: Optional[np.ndarray] = None
    previous_centre: Optional[np.ndarray] = None
    reference_width: Optional[float] = None
    retained = 0

    for frame in sampled:
        height, width = frame.image.shape[:2]
        detections = detector.detect_all(frame.image)
        detection = faces.select_primary_face(
            detections, previous_centre, reference_width, cfg.max_subject_jump_ratio
        )

        row: Dict[str, object] = {
            "sample_index": frame.sample_index,
            "frame_index": frame.frame_index,
            "timestamp": round(frame.timestamp, 6),
            "n_faces_detected": len(detections),
            "face_found": detection is not None,
        }

        if detection is None:
            # Score the frame anyway (all-zero components) so the CSV has a row
            # per sampled frame and retention arithmetic stays honest.
            components = mrs.MRSComponents()
            row.update(_empty_geometry_columns())
            row.update(components.as_dict())
            row["mrs"] = mrs.combine(components, cfg.weights.as_dict())
            row["retained"] = False
            row["frame_file"] = ""
            rows.append(row)
            previous_aligned = None  # break the optical-flow chain
            continue

        aligned = alignment.align_face(
            frame.image,
            right_eye=detection.right_eye,
            left_eye=detection.left_eye,
            output_size=cfg.output_size,
            desired_left_eye_x=cfg.desired_left_eye_x,
            desired_eye_y=cfg.desired_eye_y,
        )

        # --- head pose + eye openness ------------------------------------- #
        mesh_openness: Optional[float] = None
        head_pose = pose.pose_from_keypoints(detection.keypoints, width, height)
        pose_source = "blazeface_keypoints_pnp"

        if landmarker is not None:
            mesh = landmarker.detect(aligned.image)
            if mesh is not None:
                mesh_pose = pose.pose_from_transformation_matrix(mesh.transformation_matrix)
                if mesh_pose.success:
                    # Alignment removed in-plane rotation; add it back so the
                    # reported roll refers to the ORIGINAL frame.
                    head_pose = pose.HeadPose(
                        yaw=mesh_pose.yaw,
                        pitch=mesh_pose.pitch,
                        roll=mesh_pose.roll + aligned.rotation_deg,
                        success=True,
                    )
                    pose_source = "mesh_transformation_matrix"
                left_ear = faces.eye_aspect_ratio(mesh.points, faces.LEFT_EYE_EAR_IDX)
                right_ear = faces.eye_aspect_ratio(mesh.points, faces.RIGHT_EYE_EAR_IDX)
                mesh_openness = float((left_ear + right_ear) / 2.0)

        # --- MRS components ------------------------------------------------ #
        motion_score, median_flow, flow_ref = mrs.motion_consistency_score(
            aligned.image, previous_aligned, cfg.max_flow_magnitude_px
        )
        roi = mrs.face_roi(frame.image, detection.bbox_xyxy)
        components = mrs.MRSComponents(
            blur=mrs.blur_score(roi, cfg.blur_var_reference),
            face_visibility=mrs.face_visibility_score(detection.confidence),
            head_rotation=mrs.head_rotation_score(
                head_pose.deviation, cfg.max_head_deviation_deg
            ),
            eye_visibility=mrs.eye_visibility_score(
                detection.right_eye,
                detection.left_eye,
                detection.bbox_xyxy,
                frame.image.shape,
                margin=cfg.eye_bbox_margin,
                mesh_eye_openness=mesh_openness,
            ),
            motion_consistency=motion_score,
        )
        score = mrs.combine(components, cfg.weights.as_dict())
        keep = score >= cfg.mrs_threshold

        row.update(
            {
                "detection_confidence": float(detection.confidence),
                "bbox_x1": detection.bbox_xyxy[0],
                "bbox_y1": detection.bbox_xyxy[1],
                "bbox_x2": detection.bbox_xyxy[2],
                "bbox_y2": detection.bbox_xyxy[3],
                "interocular_px": aligned.interocular_px,
                "align_rotation_deg": aligned.rotation_deg,
                "align_scale": aligned.scale,
                "yaw": head_pose.yaw,
                "pitch": head_pose.pitch,
                "roll": head_pose.roll,
                "head_deviation_deg": head_pose.deviation,
                "pose_source": pose_source,
                "mesh_eye_openness": mesh_openness if mesh_openness is not None else float("nan"),
                "laplacian_variance": mrs.laplacian_variance(roi),
                "roi_height_px": int(roi.shape[0]),
                "median_flow_px": median_flow,
                "motion_reference_available": flow_ref,
            }
        )
        row.update(alignment.affine_to_dict(aligned.affine))
        row.update(components.as_dict())
        row["mrs"] = score
        row["retained"] = bool(keep)

        if keep:
            filename = io_utils.frame_filename(
                frame.sample_index, frame.frame_index, cfg.image_format
            )
            io_utils.save_frame(
                aligned.image, frames_dir / filename, cfg.image_format, cfg.jpeg_quality
            )
            row["frame_file"] = f"{FRAMES_SUBDIR}/{filename}"
            if cfg.save_normalized_npy:
                tensor = alignment.normalize_image(aligned.image, cfg.norm_mean, cfg.norm_std)
                np.save(frames_dir / f"{Path(filename).stem}.npy", tensor.astype(np.float32))
            retained += 1
        else:
            row["frame_file"] = ""

        rows.append(row)
        previous_aligned = aligned.image
        previous_centre = faces.bbox_centre(detection)
        box_w, _ = detection.bbox_wh
        reference_width = box_w if reference_width is None else 0.7 * reference_width + 0.3 * box_w

    scores_df = pd.DataFrame(rows)
    written = io_utils.write_table(scores_df, clip_dir / SCORES_STEM)
    logger.info(
        "%s/%s/%s: %d sampled, %d retained (%.0f%%), mean MRS %.3f -> %s",
        record.split,
        record.subject_id,
        record.clip_id,
        len(rows),
        retained,
        100.0 * retained / max(len(rows), 1),
        float(scores_df["mrs"].mean()) if len(scores_df) else float("nan"),
        written.name,
    )
    return _manifest_row(
        record, cfg, meta, rows=rows, retained=retained,
        duration=time.perf_counter() - started, status="ok",
        scores_file=str(written.relative_to(output_root)),
    )


def _empty_geometry_columns() -> Dict[str, object]:
    """Placeholder geometry columns for frames with no detected face."""
    nan = float("nan")
    row: Dict[str, object] = {
        "detection_confidence": 0.0,
        "bbox_x1": nan, "bbox_y1": nan, "bbox_x2": nan, "bbox_y2": nan,
        "interocular_px": nan, "align_rotation_deg": nan, "align_scale": nan,
        "yaw": nan, "pitch": nan, "roll": nan, "head_deviation_deg": nan,
        "pose_source": "none", "mesh_eye_openness": nan,
        "laplacian_variance": nan, "roi_height_px": 0, "median_flow_px": nan,
        "motion_reference_available": False,
    }
    row.update({k: nan for k in alignment.affine_to_dict(np.zeros((2, 3)))})
    return row


def _manifest_row(
    record: ClipRecord,
    cfg: Phase1Config,
    meta: Optional[video.VideoMetadata],
    rows: List[dict],
    retained: int,
    duration: float,
    status: str,
    scores_file: str = "",
) -> Dict[str, object]:
    sampled = len(rows)
    frames_with_face = sum(1 for r in rows if r.get("face_found"))
    mrs_values = [float(r["mrs"]) for r in rows if "mrs" in r]
    return {
        "ClipID": record.clip_id,
        "SubjectID": record.subject_id,
        "split": record.split,
        "video_path": str(record.video_path),
        "status": status,
        "native_fps": meta.native_fps if meta else float("nan"),
        "native_frame_count": meta.frame_count if meta else 0,
        "width": meta.width if meta else 0,
        "height": meta.height if meta else 0,
        "decoder": meta.decoder if meta else "none",
        "sample_fps": cfg.sample_fps,
        "frames_sampled": sampled,
        "frames_with_face": frames_with_face,
        "face_detection_ratio": frames_with_face / sampled if sampled else 0.0,
        "frames_retained": retained,
        "retention_ratio": retained / sampled if sampled else 0.0,
        "mean_mrs": float(np.mean(mrs_values)) if mrs_values else float("nan"),
        "median_mrs": float(np.median(mrs_values)) if mrs_values else float("nan"),
        "min_mrs": float(np.min(mrs_values)) if mrs_values else float("nan"),
        "max_mrs": float(np.max(mrs_values)) if mrs_values else float("nan"),
        "mrs_threshold": cfg.mrs_threshold,
        "scores_file": scores_file,
        "seconds": round(duration, 3),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(input_root: Path, output_root: Path, cfg: Phase1Config, model_dir: Optional[Path]) -> pd.DataFrame:
    output_root = io_utils.ensure_dir(Path(output_root))
    logs_dir = io_utils.ensure_dir(output_root / "logs")
    dump_config(cfg, logs_dir / "phase1_config.json")

    if not video.ffmpeg_available():
        logger.warning(
            "ffmpeg is not on PATH. OpenCV's bundled decoder handles the DAiSEE "
            "AVIs, so this only matters if a clip fails to open."
        )
    else:
        logger.info("ffmpeg available at %s (decode fallback enabled)", video.ffmpeg_path())

    clips = discover_clips(input_root, cfg.splits, cfg.video_extensions, cfg.limit_clips)
    if not clips:
        raise SystemExit(f"No clips discovered under {input_root}")

    labels = load_labels(input_root, cfg.splits)
    lookup = label_lookup(labels)

    detector_model = ensure_model("face_detector", model_dir)
    detector = faces.FaceDetectorWrapper(detector_model, cfg.min_detection_confidence)
    landmarker = None
    if cfg.use_mesh_for_pose:
        landmarker = faces.FaceLandmarkerWrapper(ensure_model("face_landmarker", model_dir))
        logger.info("Mesh-based pose enabled (slower, more accurate yaw/pitch/roll)")

    manifest_rows: List[Dict[str, object]] = []
    try:
        for index, record in enumerate(clips, start=1):
            logger.info("[%d/%d] %s", index, len(clips), record.relative_dir)
            try:
                row = process_clip(record, cfg, output_root, detector, landmarker)
            except Exception:
                logger.exception("Clip failed: %s", record.relative_dir)
                row = _manifest_row(record, cfg, None, [], 0, 0.0, "error")
            row.update(lookup.get((record.split, record.clip_id), {c: None for c in LABEL_COLUMNS}))
            row["has_label"] = (record.split, record.clip_id) in lookup
            manifest_rows.append(row)
    finally:
        detector.close()
        if landmarker is not None:
            landmarker.close()

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_root / "phase1_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    logger.info("Manifest written: %s", manifest_path)

    _write_warnings(manifest, cfg, logs_dir / "phase1_warnings.csv")
    _log_summary(manifest, cfg)
    return manifest


def _write_warnings(manifest: pd.DataFrame, cfg: Phase1Config, path: Path) -> None:
    """Flag clips that need a human look rather than dropping them silently."""
    warnings: List[dict] = []
    for row in manifest.to_dict(orient="records"):
        reasons = []
        if row["status"] != "ok":
            reasons.append(f"status={row['status']}")
        if row["frames_sampled"] == 0:
            reasons.append("no frames decoded")
        elif row["face_detection_ratio"] == 0.0:
            reasons.append("no face detected in any sampled frame")
        elif row["retention_ratio"] < cfg.low_retention_warn_ratio:
            reasons.append(
                f"low retention {row['retention_ratio']:.0%} "
                f"(< {cfg.low_retention_warn_ratio:.0%})"
            )
        if not row.get("has_label", False):
            reasons.append("no label row in Labels CSV")
        if reasons:
            warnings.append(
                {
                    "ClipID": row["ClipID"],
                    "SubjectID": row["SubjectID"],
                    "split": row["split"],
                    "frames_sampled": row["frames_sampled"],
                    "frames_retained": row["frames_retained"],
                    "retention_ratio": round(float(row["retention_ratio"]), 4),
                    "face_detection_ratio": round(float(row["face_detection_ratio"]), 4),
                    "mean_mrs": row["mean_mrs"],
                    "reasons": "; ".join(reasons),
                }
            )
    pd.DataFrame(warnings).to_csv(path, index=False)
    if warnings:
        logger.warning("%d clip(s) flagged in %s", len(warnings), path)
    else:
        logger.info("No clips flagged; warnings log written empty: %s", path)


def _log_summary(manifest: pd.DataFrame, cfg: Phase1Config) -> None:
    logger.info("=" * 78)
    logger.info("PHASE 1 SUMMARY")
    logger.info("  clips processed        : %d", len(manifest))
    logger.info("  clips ok               : %d", int((manifest["status"] == "ok").sum()))
    logger.info("  frames sampled (total) : %d", int(manifest["frames_sampled"].sum()))
    logger.info("  frames retained (total): %d", int(manifest["frames_retained"].sum()))
    overall = manifest["frames_retained"].sum() / max(manifest["frames_sampled"].sum(), 1)
    logger.info("  overall retention      : %.1f%%", 100 * overall)
    logger.info("  mean per-clip MRS      : %.3f", float(manifest["mean_mrs"].mean()))
    for split, group in manifest.groupby("split"):
        logger.info(
            "  %-11s: %2d clips, %4d/%4d frames retained (%.1f%%), mean MRS %.3f",
            split,
            len(group),
            int(group["frames_retained"].sum()),
            int(group["frames_sampled"].sum()),
            100 * group["frames_retained"].sum() / max(group["frames_sampled"].sum(), 1),
            float(group["mean_mrs"].mean()),
        )
    dead = manifest[manifest["face_detection_ratio"] == 0.0]
    if len(dead):
        logger.warning("  clips with NO face detected at all: %s", list(dead["ClipID"]))
    logger.info("=" * 78)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-root", type=Path, required=True, help="DAiSEE root (holds DataSet/ and Labels/).")
    p.add_argument("--output-root", type=Path, required=True, help="Directory for all outputs.")
    p.add_argument("--config", type=Path, default=None, help="JSON config file (phase1 section).")
    p.add_argument("--model-dir", type=Path, default=None, help="MediaPipe model cache directory.")
    p.add_argument("--sample-fps", type=float, default=None, help="Frame sampling rate (default 5).")
    p.add_argument("--mrs-threshold", type=float, default=None, help="Retain frames with MRS >= this.")
    p.add_argument("--output-size", type=int, default=None, help="Aligned crop size in pixels.")
    p.add_argument("--min-detection-confidence", type=float, default=None)
    p.add_argument("--image-format", choices=["png", "jpg"], default=None)
    p.add_argument("--limit-clips", type=int, default=None, help="Process only the first N clips.")
    p.add_argument("--max-frames-per-clip", type=int, default=None)
    p.add_argument("--splits", nargs="+", default=None, help="Subset of Train/Validation/Test.")
    p.add_argument("--no-mesh-pose", dest="use_mesh_for_pose", action="store_false", default=None,
                   help="Skip the face mesh in Phase 1 (~6x faster, less accurate pose).")
    p.add_argument("--save-normalized-npy", action="store_true", default=None)
    p.add_argument(
        "--weight", action="append", default=[], metavar="NAME=VALUE",
        help="Override one MRS weight, e.g. --weight blur=2.0. Repeatable.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    weight_overrides = {}
    for item in args.weight:
        if "=" not in item:
            raise SystemExit(f"--weight expects NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        if name not in mrs.COMPONENT_NAMES:
            raise SystemExit(f"Unknown MRS component {name!r}; valid: {list(mrs.COMPONENT_NAMES)}")
        weight_overrides[name] = float(value)

    cfg = load_config(
        Phase1Config,
        args.config,
        section="phase1",
        sample_fps=args.sample_fps,
        mrs_threshold=args.mrs_threshold,
        output_size=args.output_size,
        min_detection_confidence=args.min_detection_confidence,
        image_format=args.image_format,
        limit_clips=args.limit_clips,
        max_frames_per_clip=args.max_frames_per_clip,
        splits=tuple(args.splits) if args.splits else None,
        use_mesh_for_pose=args.use_mesh_for_pose,
        save_normalized_npy=args.save_normalized_npy,
        weights=weight_overrides or None,
    )

    output_root = Path(args.output_root)
    io_utils.ensure_dir(output_root / "logs")
    setup_logging(output_root / "logs" / "phase1.log")
    logger.info("Phase 1 configuration: %s", cfg)

    run(Path(args.input_root), output_root, cfg, args.model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
