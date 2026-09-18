"""Stage A preprocessing and caching (spec section 24), split into two stages.

    Stage 1  video -> sampled frames -> face crops -> landmarks -> raw MRS
    Stage 2  cached crops -> ViT features

Why the split: Stage 1 is dominated by video decoding and MediaPipe, runs on the
CPU, and is *independent of the backbone*. Stage 2 is a GPU forward pass. Keeping
them apart means a different ViT backbone costs only Stage 2 -- which is what
makes an honest backbone comparison affordable instead of a five-fold repeat of
the expensive part.

What Stage 1 caches, and why in this form:

* face crops           JPEG-encoded at ``CROP_STORE_SIZE`` px, so each backbone
                       can resize to its own native input, and so the cache
                       stays ~15x smaller than raw arrays
* landmark features    [T, L]  MediaPipe geometry + blendshapes
* raw MRS signals      [T]     five per-frame signals, blur left *raw*

Caching the raw MRS signals rather than a finished score means the blur
calibration can be refitted, or the five MRS weights changed, without decoding a
single video again. That matters because the calibration must be fitted on the
training split only, which is not knowable until the training split has been
swept.

Cache keys cover everything that would invalidate their contents, so a stale
cache cannot be silently reused. Nothing here reads an engagement label except to
copy it into the record.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .face_processing import FaceDetection, FaceDetector, center_square, crop_face
from .landmarks import CLIP_FEATURE_DIM, FEATURE_DIM, LandmarkExtractor, aggregate_clip_features
from .mrs import compute_raw_components
from .utils import ensure_dir, resolve_path, save_json
from .video_sampling import read_frames

# Bump when a step changes in a way that alters what is cached.
STAGE1_VERSION = "1.3.0"
STAGE2_VERSION = "1.1.0"

# Crops are stored at this resolution and resized per-backbone at Stage 2.
# 256 is comfortably above the ~200-300 px a face actually occupies in a
# 640x480 DAiSEE frame, so nothing real is thrown away; a backbone whose native
# input is larger (DINOv2 at 518) receives an upsampled crop, which is stated
# rather than hidden.
CROP_STORE_SIZE = 256
CROP_JPEG_QUALITY = 95


# --------------------------------------------------------------------------- #
# Cache keys
# --------------------------------------------------------------------------- #
def _digest(payload: dict) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]


def stage1_key(cfg, corruption: str | None = None) -> str:
    payload = {
        "stage1_version": STAGE1_VERSION,
        "num_frames": int(cfg["video"]["num_frames"]),
        "sampling": str(cfg["video"]["sampling"]),
        "crop_store_size": CROP_STORE_SIZE,
        "crop_jpeg_quality": CROP_JPEG_QUALITY,
        "face": {
            "detector": str(cfg["face"]["detector"]),
            "min_detection_confidence": float(cfg["face"]["min_detection_confidence"]),
            "crop_padding": float(cfg["face"]["crop_padding"]),
            "align": bool(cfg["face"]["align"]),
            "missing_face_policy": str(cfg["face"]["missing_face_policy"]),
        },
        "landmarks": {
            "output_blendshapes": bool(cfg["landmarks"]["output_blendshapes"]),
            "output_transformation_matrix": bool(cfg["landmarks"]["output_transformation_matrix"]),
            "min_face_detection_confidence": float(cfg["landmarks"]["min_face_detection_confidence"]),
        },
        # MRS *weights* are deliberately excluded: raw signals are cached, so
        # changing the weights invalidates nothing.
        "mrs_geometry": {
            k: float(cfg["mrs"][k]) for k in (
                "head_pose_full_reliability_deg", "head_pose_zero_reliability_deg",
                "motion_ref_displacement")
        },
        "corruption": corruption,
    }
    tag = f"T{payload['num_frames']}"
    if corruption:
        tag += f"_{corruption}"
    return f"faces_{tag}_{_digest(payload)}"


def stage2_key(cfg, s1_key: str, backbone_name: str, input_size: int) -> str:
    payload = {"stage2_version": STAGE2_VERSION, "stage1_key": s1_key,
               "backbone": backbone_name, "input_size": int(input_size)}
    safe = backbone_name.replace(":", "_").replace("/", "_")
    return f"vit_{safe}_{_digest(payload)}"


def stage1_dir(cfg, key: str) -> Path:
    return resolve_path(Path(cfg["paths"]["cache_dir"]) / key)


def stage2_dir(cfg, key: str) -> Path:
    return resolve_path(Path(cfg["paths"]["cache_dir"]) / key)


def stage1_path(cfg, key: str, split: str, stem: str) -> Path:
    return stage1_dir(cfg, key) / split / f"{stem}.npz"


def stage2_path(cfg, key: str, split: str, stem: str) -> Path:
    return stage2_dir(cfg, key) / split / f"{stem}.npy"


# --------------------------------------------------------------------------- #
# JPEG packing (pickle-free)
# --------------------------------------------------------------------------- #
def pack_crops(crops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[T, S, S, 3] uint8 -> (concatenated JPEG bytes, offsets [T+1])."""
    blobs = []
    for crop in crops:
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), CROP_JPEG_QUALITY])
        if not ok:
            raise RuntimeError("cv2.imencode failed on a face crop")
        blobs.append(buf.reshape(-1))
    offsets = np.zeros(len(blobs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([b.size for b in blobs])
    return np.concatenate(blobs).astype(np.uint8), offsets


def unpack_crops(data: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Inverse of pack_crops -> [T, S, S, 3] uint8 BGR."""
    out = []
    for i in range(len(offsets) - 1):
        blob = data[offsets[i]:offsets[i + 1]]
        img = cv2.imdecode(blob, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cv2.imdecode failed for cached crop {i}")
        out.append(img)
    return np.stack(out)


# --------------------------------------------------------------------------- #
# Failure reporting
# --------------------------------------------------------------------------- #
@dataclass
class ClipFailure:
    clip_id: str
    split: str
    stage: str
    error: str
    exception: str

    def as_dict(self) -> dict:
        return {"video_id": self.clip_id, "split": self.split, "stage": self.stage,
                "error": self.error, "exception": self.exception}


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #
class FacePreprocessor:
    """Holds the detector and landmarker open across many clips."""

    def __init__(self, cfg, frame_transform=None, corruption_name: str | None = None):
        self.cfg = cfg
        self.frame_transform = frame_transform
        self.corruption_name = corruption_name
        self.detector = FaceDetector(
            Path(cfg["paths"]["models_dir"]) / "blaze_face_short_range.tflite",
            min_confidence=float(cfg["face"]["min_detection_confidence"]))
        self.landmarker = LandmarkExtractor(dict(cfg["landmarks"]))
        self.key = stage1_key(cfg, corruption_name)

    def describe(self) -> dict:
        return {
            "stage1_version": STAGE1_VERSION,
            "cache_key": self.key,
            "num_frames": int(self.cfg["video"]["num_frames"]),
            "crop_store_size": CROP_STORE_SIZE,
            "corruption": self.corruption_name,
            "face": self.detector.describe() | {
                "crop_padding": float(self.cfg["face"]["crop_padding"]),
                "align": bool(self.cfg["face"]["align"]),
                "missing_face_policy": str(self.cfg["face"]["missing_face_policy"]),
            },
            "landmarks": self.landmarker.describe(),
        }

    def process_clip(self, record) -> dict:
        cfg = self.cfg
        num_frames = int(cfg["video"]["num_frames"])

        frames, indices, probe = read_frames(record.path, num_frames)
        if self.frame_transform is not None:
            frames = np.stack([self.frame_transform(f) for f in frames])
        fps = probe.fps or 30.0
        h, w = frames.shape[1:3]

        crops = np.empty((num_frames, CROP_STORE_SIZE, CROP_STORE_SIZE, 3), dtype=np.uint8)
        landmark_rows = np.zeros((num_frames, FEATURE_DIM), dtype=np.float32)
        face_found = np.zeros(num_frames, dtype=bool)
        landmarks_found = np.zeros(num_frames, dtype=bool)
        head_pose = np.full((num_frames, 3), np.nan, dtype=np.float32)
        det_conf = np.full(num_frames, np.nan, dtype=np.float32)
        raw = {k: np.zeros(num_frames, dtype=np.float32)
               for k in ("blur_raw", "face_area_fraction", "eye_visibility",
                         "motion_consistency")}
        bbox_source: list[str] = []
        motion_source: list[str] = []

        prev_lm = None
        prev_crop = None
        for t in range(num_frames):
            frame = frames[t]
            det = self.detector.detect(frame)
            lm = self.landmarker.extract(frame)

            # Explicit fallback chain; each step is recorded, never silent.
            if not det.found:
                if lm.found and lm.landmarks is not None:
                    from .face_processing import bbox_from_landmarks

                    box = bbox_from_landmarks(lm.landmarks[:, :2], w, h)
                    src = "landmark_bbox"
                else:
                    box = center_square(w, h)
                    src = "center_fallback"
                det = FaceDetection(False, src, box, None,
                                    ((box[2]-box[0])*(box[3]-box[1]))/float(w*h), None, None)

            crop = crop_face(frame, det, CROP_STORE_SIZE,
                             padding=float(cfg["face"]["crop_padding"]),
                             align=bool(cfg["face"]["align"]))
            crops[t] = crop
            face_found[t] = det.source == "mediapipe_face_detector"
            bbox_source.append(det.source)
            landmarks_found[t] = bool(lm.found)
            landmark_rows[t] = lm.features
            if lm.head_pose_deg is not None:
                head_pose[t] = lm.head_pose_deg

            dt = ((indices[t] - indices[t - 1]) / fps) if t > 0 else 0.0
            comps = compute_raw_components(crop, det, lm, prev_lm, prev_crop, dt,
                                           dict(cfg["mrs"]))
            for k in raw:
                raw[k][t] = comps[k]
            det_conf[t] = comps["detector_confidence"]
            motion_source.append(comps["motion_source"])

            prev_lm = lm
            prev_crop = crop

        if not np.all(np.isfinite(landmark_rows)):
            raise RuntimeError("non-finite landmark features in clip")

        clip_landmarks = aggregate_clip_features(landmark_rows, landmarks_found)
        if clip_landmarks.shape != (CLIP_FEATURE_DIM,):
            raise RuntimeError(
                f"aggregated landmark features {clip_landmarks.shape}, expected {(CLIP_FEATURE_DIM,)}")

        crops_data, crops_offsets = pack_crops(crops)
        return {
            "crops_data": crops_data,
            "crops_offsets": crops_offsets,
            "crop_size": np.int32(CROP_STORE_SIZE),
            "landmark_features": landmark_rows,
            "landmark_clip_features": clip_landmarks,
            "blur_raw": raw["blur_raw"],
            "face_area_fraction": raw["face_area_fraction"],
            "mrs_eye_visibility": raw["eye_visibility"],
            "mrs_motion_consistency": raw["motion_consistency"],
            "face_found": face_found,
            "landmarks_found": landmarks_found,
            "detector_confidence": det_conf,
            "bbox_source": np.array(bbox_source),
            "motion_source": np.array(motion_source),
            "head_pose_deg": head_pose,
            "frame_indices": np.array(indices, dtype=np.int32),
            "label": np.int64(record.engagement),
            "subject_id": np.str_(record.subject_id),
            "clip_id": np.str_(record.clip_id),
            "split": np.str_(record.split),
            "fps": np.float32(fps),
            "source_frame_count": np.int32(probe.frame_count or 0),
            "source_resolution": np.array([w, h], dtype=np.int32),
        }

    def close(self) -> None:
        self.detector.close()
        self.landmarker.close()


def _atomic_savez(out_path: Path, data: dict) -> None:
    """Write an .npz so that it either exists complete or does not exist at all.

    The sweep treats "file exists" as "clip done". A process killed half-way
    through np.savez_compressed would otherwise leave a truncated archive that
    every later run skips as cached and that only fails, hours later, when the
    training loader opens it. Writing to a temporary name and renaming makes the
    final path appear in one step.
    """
    # The process id keeps two writers from ever sharing a temporary file, e.g.
    # overlapping shard runs, or a new run starting while an old one is alive.
    tmp = out_path.with_name(f"{out_path.name}.{os.getpid()}.partial")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **data)
    os.replace(tmp, out_path)


def _remove_stale_partials(out_path: Path) -> None:
    """Delete leftover temporaries of *this clip* from interrupted runs.

    Scoped to one clip on purpose. A sweep over the whole split directory would
    reach into other shards' in-progress writes: measured here, it deletes them
    on Linux and raises PermissionError on Windows. A file still held open by a
    live process is left alone.
    """
    for stale in out_path.parent.glob(f"{out_path.name}.*.partial"):
        try:
            stale.unlink()
        except OSError:
            pass


def run_stage1(cfg, index, splits=("train", "val", "test"), limit: int | None = None,
               force: bool = False, logger=None, frame_transform=None,
               corruption_name: str | None = None,
               shard: tuple[int, int] | None = None) -> dict:
    """Sweep the requested splits, writing one .npz per clip plus a manifest.

    Failures never stop the sweep and are never hidden: each is logged with
    video_id, stage, error and exception, and the summary reports total /
    successful / failed / failure percentage (spec Rule 5).

    ``shard=(i, n)`` processes only every n-th clip starting at i, so n
    independent processes can split the corpus between them. Each clip is
    written to its own file, so shards never contend, and a clip picked up by
    two runs is simply found cached by the second.
    """
    if shard is not None:
        i_shard, n_shard = shard
        if not (0 <= i_shard < n_shard):
            raise ValueError(f"shard index {i_shard} out of range for {n_shard} shards")
    pre = FacePreprocessor(cfg, frame_transform=frame_transform, corruption_name=corruption_name)
    root = ensure_dir(stage1_dir(cfg, pre.key))
    failures: list[ClipFailure] = []
    counts = {"total": 0, "cached": 0, "processed": 0, "failed": 0}
    per_clip: list[dict] = []
    t_start = time.perf_counter()

    try:
        for split in splits:
            ensure_dir(root / split)
            records = index.clips.get(split, [])
            if limit:
                records = records[:limit]
            if shard is not None:
                records = records[i_shard::n_shard]
            for i, record in enumerate(records, 1):
                counts["total"] += 1
                out_path = stage1_path(cfg, pre.key, split, record.stem)
                if out_path.exists() and not force:
                    counts["cached"] += 1
                    continue
                # A .partial for this clip is the remains of an interrupted
                # write; the clip has no complete entry and is redone now.
                _remove_stale_partials(out_path)
                stage = "read_frames"
                try:
                    data = pre.process_clip(record)
                    stage = "write_cache"
                    _atomic_savez(out_path, data)
                    counts["processed"] += 1
                    per_clip.append({
                        "clip_id": record.clip_id, "split": split,
                        "subject_id": record.subject_id, "label": int(record.engagement),
                        "face_detected_frames": int(data["face_found"].sum()),
                        "landmark_frames": int(data["landmarks_found"].sum()),
                        "num_frames": int(len(data["frame_indices"])),
                    })
                except Exception as exc:
                    counts["failed"] += 1
                    failures.append(ClipFailure(record.clip_id, split, stage,
                                                f"{type(exc).__name__}: {exc}",
                                                traceback.format_exc(limit=6)))
                    if logger:
                        logger.error("FAILED %s/%s at %s: %s", split, record.clip_id, stage, exc)
                if logger and (i % 10 == 0 or i == len(records)):
                    done = counts["processed"] + counts["failed"]
                    elapsed = time.perf_counter() - t_start
                    rate = done / elapsed if elapsed > 0 and done else 0.0
                    remaining = len(records) - i
                    eta = f"{remaining / rate / 60:.1f} min" if rate else "n/a"
                    logger.info("stage1 %s: %d/%d  (%.2f clips/s, %s left in this split)",
                                split, i, len(records), rate, eta)
    finally:
        pre.close()

    attempted = counts["total"] - counts["cached"]
    summary = {
        "cache_key": pre.key,
        "cache_dir": str(root),
        "counts": counts,
        "failure_percentage": round(100.0 * counts["failed"] / attempted, 2) if attempted else 0.0,
        "failures": [f.as_dict() for f in failures],
        "settings": pre.describe(),
        "per_clip": per_clip,
        "shard": list(shard) if shard is not None else None,
    }
    # Parallel shards would overwrite a single manifest; each gets its own.
    name = (f"manifest_shard{shard[0]}of{shard[1]}.json" if shard is not None
            else "manifest.json")
    save_json(summary, root / name)
    return summary


# --------------------------------------------------------------------------- #
# Stage 2
# --------------------------------------------------------------------------- #
def run_stage2(cfg, index, vit_encoder, s1_key: str,
               splits=("train", "val", "test"), limit: int | None = None,
               force: bool = False, device: str = "cpu", logger=None) -> dict:
    """Cached crops -> ViT features. One .npy of shape [T, D] per clip."""
    import torch

    from .vit_encoder import preprocess_crops

    spec = vit_encoder.spec
    key = stage2_key(cfg, s1_key, spec.name, spec.input_size)
    root = ensure_dir(stage2_dir(cfg, key))
    failures: list[ClipFailure] = []
    counts = {"total": 0, "cached": 0, "processed": 0, "failed": 0, "missing_stage1": 0}
    shapes: set[tuple[int, int]] = set()

    for split in splits:
        ensure_dir(root / split)
        records = index.clips.get(split, [])
        if limit:
            records = records[:limit]
        for i, record in enumerate(records, 1):
            counts["total"] += 1
            out_path = stage2_path(cfg, key, split, record.stem)
            if out_path.exists() and not force:
                counts["cached"] += 1
                continue
            src = stage1_path(cfg, s1_key, split, record.stem)
            if not src.exists():
                counts["missing_stage1"] += 1
                failures.append(ClipFailure(record.clip_id, split, "stage1_lookup",
                                            f"no Stage 1 cache at {src}", ""))
                continue
            try:
                data = np.load(src)
                crops = unpack_crops(data["crops_data"], data["crops_offsets"])
                batch = preprocess_crops(crops, spec).unsqueeze(0).to(device)
                with torch.no_grad():
                    feats = vit_encoder(batch)
                arr = feats.squeeze(0).float().cpu().numpy().astype(np.float32)
                if not np.all(np.isfinite(arr)):
                    raise RuntimeError("ViT produced non-finite features")
                if arr.shape[1] != spec.embed_dim:
                    raise RuntimeError(
                        f"ViT output dim {arr.shape[1]} != reported {spec.embed_dim}")
                shapes.add(arr.shape)
                np.save(out_path, arr)
                counts["processed"] += 1
            except Exception as exc:
                counts["failed"] += 1
                failures.append(ClipFailure(record.clip_id, split, "vit_forward",
                                            f"{type(exc).__name__}: {exc}",
                                            traceback.format_exc(limit=6)))
                if logger:
                    logger.error("FAILED %s/%s at vit_forward: %s", split, record.clip_id, exc)
            if logger and i % 20 == 0:
                logger.info("stage2 %s: %d/%d", split, i, len(records))

    attempted = counts["total"] - counts["cached"]
    summary = {
        "cache_key": key,
        "stage1_key": s1_key,
        "cache_dir": str(root),
        "counts": counts,
        "failure_percentage": round(100.0 * counts["failed"] / attempted, 2) if attempted else 0.0,
        "failures": [f.as_dict() for f in failures],
        "backbone": spec.as_dict(),
        "observed_feature_shapes": sorted(shapes),
    }
    save_json(summary, root / "manifest.json")
    return summary
