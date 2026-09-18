"""TEST 2 + TEST 3 + CHECKPOINT 3 - video reading and the face pipeline.

Decodes real clips, runs the detector, validates every crop, and saves a visual
montage so the crops can actually be looked at rather than trusted.

Writes artifacts/face_crop_preview.png and artifacts/face_pipeline_report.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_index import build_index  # noqa: E402
from src.face_processing import (  # noqa: E402
    FaceDetection, FaceDetector, bbox_from_landmarks, center_square, crop_face,
)
from src.landmarks import LandmarkExtractor  # noqa: E402
from src.utils import ensure_dir, get_logger, load_config, resolve_path, save_json  # noqa: E402
from src.video_sampling import read_frames  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=6, help="clips to run per split")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    log = get_logger("face", "logs/test_face_pipeline.log")
    cfg = load_config(args.config)
    index = build_index(cfg)
    num_frames = int(cfg["video"]["num_frames"])
    image_size = int(cfg["face"]["image_size"])

    detector = FaceDetector(
        Path(cfg["paths"]["models_dir"]) / "blaze_face_short_range.tflite",
        min_confidence=float(cfg["face"]["min_detection_confidence"]))
    landmarker = LandmarkExtractor(dict(cfg["landmarks"]))

    print("=" * 72)
    print("TEST 2 / TEST 3 / CHECKPOINT 3 - VIDEO + FACE PIPELINE")
    print("=" * 72)
    print("detector settings:")
    for k, v in detector.describe().items():
        print(f"  {k}: {v}")
    print(f"  crop_padding: {cfg['face']['crop_padding']}")
    print(f"  align: {cfg['face']['align']}")
    print(f"  input_resolution: {image_size}")

    selected = []
    for split in ("train", "val", "test"):
        selected += index.clips.get(split, [])[: args.clips]

    montage_rows = []
    stats = {"frames": 0, "detector_hits": 0, "landmark_bbox": 0, "center_fallback": 0,
             "landmark_hits": 0}
    problems: list[str] = []
    per_clip = []

    for rec in selected:
        frames, indices, probe = read_frames(rec.path, num_frames)
        print(f"\n[{rec.split}] {rec.clip_id}")
        print(f"  fps={probe.fps}  frames_decoded={probe.frame_count}  "
              f"resolution={probe.width}x{probe.height}  duration={probe.duration_s:.2f}s")
        print(f"  sampled indices: {indices}")

        h, w = frames.shape[1:3]
        crops = []
        sources = []
        confidences = []
        for t in range(num_frames):
            frame = frames[t]
            det = detector.detect(frame)
            lm = landmarker.extract(frame)
            stats["frames"] += 1
            if lm.found:
                stats["landmark_hits"] += 1

            if det.found:
                stats["detector_hits"] += 1
            elif lm.found and lm.landmarks is not None:
                box = bbox_from_landmarks(lm.landmarks[:, :2], w, h)
                det = FaceDetection(False, "landmark_bbox", box, None,
                                    ((box[2]-box[0])*(box[3]-box[1]))/float(w*h), None, None)
                stats["landmark_bbox"] += 1
            else:
                box = center_square(w, h)
                det = FaceDetection(False, "center_fallback", box, None,
                                    ((box[2]-box[0])*(box[3]-box[1]))/float(w*h), None, None)
                stats["center_fallback"] += 1

            # --- CHECKPOINT 3 validations ---
            x0, y0, x1, y1 = det.bbox
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                problems.append(f"{rec.clip_id} frame {t}: bbox {det.bbox} outside {w}x{h}")
            crop = crop_face(frame, det, image_size,
                             padding=float(cfg["face"]["crop_padding"]),
                             align=bool(cfg["face"]["align"]))
            if crop.size == 0:
                problems.append(f"{rec.clip_id} frame {t}: empty crop")
            if crop.shape != (image_size, image_size, 3):
                problems.append(f"{rec.clip_id} frame {t}: crop shape {crop.shape}")
            if crop.std() < 1.0:
                problems.append(f"{rec.clip_id} frame {t}: crop is nearly uniform "
                                f"(std={crop.std():.2f}) - likely a bad region")
            crops.append(crop)
            sources.append(det.source)
            if det.confidence is not None:
                confidences.append(det.confidence)

        hits = sum(1 for s in sources if s == "mediapipe_face_detector")
        print(f"  detector found a face in {hits}/{num_frames} sampled frames"
              f"  (mean confidence {np.mean(confidences):.3f})" if confidences
              else f"  detector found a face in {hits}/{num_frames} sampled frames")
        print(f"  bbox sources: {dict((s, sources.count(s)) for s in set(sources))}")
        per_clip.append({"clip_id": rec.clip_id, "split": rec.split,
                         "detector_hits": hits, "num_frames": num_frames,
                         "mean_confidence": float(np.mean(confidences)) if confidences else None,
                         "bbox_sources": {s: sources.count(s) for s in set(sources)}})

        # 8 evenly spaced crops from this clip for the montage.
        step = max(1, num_frames // 8)
        montage_rows.append(np.concatenate(crops[::step][:8], axis=1))

    detector.close()
    landmarker.close()

    # ----------------------------------------------------------- montage
    width = min(r.shape[1] for r in montage_rows)
    montage = np.concatenate([r[:, :width] for r in montage_rows], axis=0)
    out = ensure_dir("artifacts") / "face_crop_preview.png"
    cv2.imwrite(str(out), montage)
    print(f"\nwrote {out}  ({montage.shape[1]}x{montage.shape[0]}, "
          f"{len(montage_rows)} clips x 8 crops)")

    rate = 100.0 * stats["detector_hits"] / max(stats["frames"], 1)
    lm_rate = 100.0 * stats["landmark_hits"] / max(stats["frames"], 1)
    print(f"\nface detection rate : {stats['detector_hits']}/{stats['frames']} ({rate:.1f}%)")
    print(f"landmark hit rate   : {stats['landmark_hits']}/{stats['frames']} ({lm_rate:.1f}%)")
    print(f"landmark-bbox fallback : {stats['landmark_bbox']}")
    print(f"centre-crop fallback   : {stats['center_fallback']}")

    report = {
        "settings": detector.describe() | {
            "crop_padding": float(cfg["face"]["crop_padding"]),
            "align": bool(cfg["face"]["align"]),
            "input_resolution": image_size,
            "missing_face_policy": str(cfg["face"]["missing_face_policy"]),
        },
        "frames_examined": stats["frames"],
        "detector_hit_rate_pct": round(rate, 2),
        "landmark_hit_rate_pct": round(lm_rate, 2),
        "fallback_counts": {"landmark_bbox": stats["landmark_bbox"],
                            "center_fallback": stats["center_fallback"]},
        "per_clip": per_clip,
        "validation_problems": problems,
        "preview_image": str(out),
    }
    save_json(report, "artifacts/face_pipeline_report.json")
    print("wrote artifacts/face_pipeline_report.json")

    print("\n" + "=" * 72)
    if problems:
        print(f"CHECKPOINT 3 FAILED - {len(problems)} validation problems:")
        for p in problems[:20]:
            print(f"  - {p}")
        print("=" * 72)
        return 1
    if rate < 50.0:
        print(f"CHECKPOINT 3 FAILED - detector hit rate {rate:.1f}% is too low to proceed.")
        print("Inspect artifacts/face_crop_preview.png before changing anything.")
        print("=" * 72)
        return 1
    print("CHECKPOINT 3 PASSED (inspect artifacts/face_crop_preview.png visually as well)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
