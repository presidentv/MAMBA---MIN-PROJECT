"""Download the MediaPipe Tasks model bundles used by Phase 1 and Phase 2.

Usage:
    python pipeline/fetch_models.py [--model-dir DIR] [--force]

Downloads (from https://storage.googleapis.com/mediapipe-models, Google's
official MediaPipe model host):
    blaze_face_short_range.tflite  ~0.2 MB  face detection (Phase 1)
    face_landmarker.task           ~3.8 MB  478-point iris-refined mesh (Phase 2)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engagement_pipeline.logging_utils import setup_logging  # noqa: E402
from engagement_pipeline.models import MODELS, default_model_dir, ensure_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=None, help="Where to cache the bundles.")
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    args = parser.parse_args()

    logger = setup_logging()
    target_dir = args.model_dir or default_model_dir()
    logger.info("Model cache directory: %s", target_dir)
    for spec in MODELS.values():
        logger.info("  %-28s %s", spec.filename, spec.description)

    paths = ensure_all(args.model_dir, force=args.force)
    for key, path in paths.items():
        logger.info("Ready: %-16s -> %s (%.2f MB)", key, path, path.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
