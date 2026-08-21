"""Frame sampling with an OpenCV primary path and an ffmpeg fallback.

OpenCV's bundled FFmpeg build decodes the DAiSEE MPEG-4 AVIs directly, so the
external ``ffmpeg`` binary is only needed as a fallback for containers OpenCV
cannot open. Availability is *checked*, never assumed: if ffmpeg is absent the
fallback is disabled with a warning rather than crashing at use time.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_FPS = 30.0


@lru_cache(maxsize=1)
def ffmpeg_path() -> Optional[str]:
    """Return the ffmpeg executable path, or None if it is not on PATH."""
    return shutil.which("ffmpeg")


@lru_cache(maxsize=1)
def ffmpeg_available() -> bool:
    path = ffmpeg_path()
    if path is None:
        return False
    try:
        subprocess.run([path, "-version"], capture_output=True, timeout=15, check=True)
        return True
    except (subprocess.SubprocessError, OSError):
        return False


@dataclass
class SampledFrame:
    """One decoded frame plus the bookkeeping needed to trace it back to the video."""

    sample_index: int
    """0-based index within the sampled sequence (not the native frame number)."""
    frame_index: int
    """0-based index in the source video's native frame sequence."""
    timestamp: float
    """Seconds from the start of the clip."""
    image: np.ndarray
    """BGR uint8 frame as decoded."""


@dataclass
class VideoMetadata:
    path: Path
    native_fps: float
    frame_count: int
    width: int
    height: int
    decoder: str

    @property
    def duration(self) -> float:
        return self.frame_count / self.native_fps if self.native_fps > 0 else 0.0


def probe(video_path: Path) -> Optional[VideoMetadata]:
    """Read container metadata without decoding the whole file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            logger.warning(
                "Invalid FPS reported for %s; assuming %.1f", video_path.name, DEFAULT_FALLBACK_FPS
            )
            fps = DEFAULT_FALLBACK_FPS
        return VideoMetadata(
            path=video_path,
            native_fps=fps,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            decoder="opencv",
        )
    finally:
        cap.release()


def sample_step(native_fps: float, target_fps: float) -> int:
    """Frames to advance between samples, at least 1.

    At the default 5 fps target on 30 fps DAiSEE clips this is 6, i.e. ~50 frames
    from a 10 s clip. If the source is slower than the target, every frame is kept.
    """
    if target_fps <= 0:
        return 1
    return max(1, int(round(native_fps / target_fps)))


def iter_sampled_frames(
    video_path: Path, target_fps: float, max_frames: int = 0
) -> Iterator[SampledFrame]:
    """Yield frames sampled at ``target_fps``, decoding sequentially.

    Sequential decode + modulo selection is used rather than ``CAP_PROP_POS_FRAMES``
    seeking: on MPEG-4 AVIs seeking is both slower and prone to landing on the
    wrong frame, and we need exact native frame indices for traceability.

    Falls back to ffmpeg extraction only if OpenCV cannot open the container.
    """
    meta = probe(video_path)
    if meta is None:
        yield from _iter_sampled_frames_ffmpeg(video_path, target_fps, max_frames)
        return

    step = sample_step(meta.native_fps, target_fps)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        yield from _iter_sampled_frames_ffmpeg(video_path, target_fps, max_frames)
        return

    try:
        frame_index = 0
        sample_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                yield SampledFrame(
                    sample_index=sample_index,
                    frame_index=frame_index,
                    timestamp=frame_index / meta.native_fps,
                    image=frame,
                )
                sample_index += 1
                if max_frames and sample_index >= max_frames:
                    break
            frame_index += 1
    finally:
        cap.release()


def _iter_sampled_frames_ffmpeg(
    video_path: Path, target_fps: float, max_frames: int = 0
) -> Iterator[SampledFrame]:
    """Decode via the external ffmpeg binary into a temp directory.

    Only reached when OpenCV refuses the container. Timestamps here are derived
    from the requested output rate, so ``frame_index`` is the *resampled* index
    rather than a native one; the manifest records the decoder used so this
    difference stays visible downstream.
    """
    if not ffmpeg_available():
        logger.error(
            "OpenCV could not open %s and ffmpeg is not on PATH; skipping clip.", video_path
        )
        return

    with tempfile.TemporaryDirectory(prefix="daisee_ffmpeg_") as tmpdir:
        tmp = Path(tmpdir)
        cmd = [
            ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps={target_fps}",
        ]
        if max_frames:
            cmd += ["-frames:v", str(max_frames)]
        cmd += [str(tmp / "frame_%06d.png")]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            logger.error("ffmpeg failed on %s: %s", video_path, exc.stderr[:400])
            return

        for sample_index, png in enumerate(sorted(tmp.glob("frame_*.png"))):
            image = cv2.imread(str(png), cv2.IMREAD_COLOR)
            if image is None:
                continue
            yield SampledFrame(
                sample_index=sample_index,
                frame_index=sample_index,
                timestamp=sample_index / target_fps if target_fps > 0 else 0.0,
                image=image,
            )


def read_sampled_frames(
    video_path: Path, target_fps: float, max_frames: int = 0
) -> Tuple[List[SampledFrame], Optional[VideoMetadata]]:
    """Eager variant of :func:`iter_sampled_frames`.

    ~50 frames of 640x480 BGR is ~44 MB, which is fine per clip and keeps the
    Phase 1 driver simple; the lazy iterator remains available for larger clips.
    """
    meta = probe(video_path)
    frames = list(iter_sampled_frames(video_path, target_fps, max_frames))
    return frames, meta
