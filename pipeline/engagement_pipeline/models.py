"""Resolution of the MediaPipe Tasks model bundles.

Why this module exists
----------------------
MediaPipe removed the legacy ``mediapipe.solutions`` API (``solutions.face_mesh``,
``solutions.face_detection``) from every wheel that supports Python 3.13+, which
is what this machine runs. The replacement Tasks API is functionally a superset
-- it exposes the same BlazeFace detector and the 478-point (iris-refined) face
mesh, plus a facial transformation matrix that gives head pose directly -- but it
does not bundle the model weights inside the wheel. They are downloaded once from
Google's official MediaPipe model storage and cached on disk.

Run ``python fetch_models.py`` once before the first pipeline run, or let the
pipeline fetch them lazily on demand.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

_BASE = "https://storage.googleapis.com/mediapipe-models"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    url: str
    approx_bytes: int
    description: str


MODELS: Dict[str, ModelSpec] = {
    "face_detector": ModelSpec(
        key="face_detector",
        filename="blaze_face_short_range.tflite",
        url=f"{_BASE}/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        approx_bytes=230_000,
        description="BlazeFace short-range detector: bbox, confidence and 6 keypoints.",
    ),
    "face_landmarker": ModelSpec(
        key="face_landmarker",
        filename="face_landmarker.task",
        url=f"{_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        approx_bytes=3_800_000,
        description="Face Mesh v2 bundle: 478 landmarks (incl. iris) + transformation matrix.",
    ),
}


def default_model_dir() -> Path:
    """Model cache location. Override with the ``DAISEE_MODEL_DIR`` env var."""
    env = os.environ.get("DAISEE_MODEL_DIR")
    if env:
        return Path(env)
    # <repo>/pipeline/engagement_pipeline/models.py -> <repo>/models
    return Path(__file__).resolve().parents[2] / "models"


def model_path(key: str, model_dir: Path | None = None) -> Path:
    spec = MODELS[key]
    return Path(model_dir or default_model_dir()) / spec.filename


def ensure_model(key: str, model_dir: Path | None = None, force: bool = False) -> Path:
    """Return a local path to the model bundle, downloading it if necessary."""
    spec = MODELS[key]
    target = model_path(key, model_dir)
    if target.is_file() and target.stat().st_size > 0 and not force:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading MediaPipe model '%s' (~%.1f MB) from %s",
        spec.filename,
        spec.approx_bytes / 1e6,
        spec.url,
    )
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(spec.url, timeout=120) as response, open(tmp, "wb") as handle:
            handle.write(response.read())
    except (urllib.error.URLError, OSError) as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Failed to download {spec.filename} from {spec.url}: {exc}. "
            "Download it manually and place it in " + str(target.parent)
        ) from exc
    tmp.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()[:16]
    logger.info("Saved %s (%d bytes, sha256:%s...)", target, target.stat().st_size, digest)
    return target


def ensure_all(model_dir: Path | None = None, force: bool = False) -> Dict[str, Path]:
    return {key: ensure_model(key, model_dir, force) for key in MODELS}
