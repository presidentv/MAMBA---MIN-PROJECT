"""Video probing and deterministic uniform temporal sampling.

Sampling is *computational subsampling*, not a claim that the skipped frames
carry no information (spec section 8). The frame count T is a config value, and
scripts/run_sampling_experiment.py measures what it costs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoProbe:
    path: str
    decodable: bool
    fps: float | None
    frame_count: int | None
    width: int | None
    height: int | None
    duration_s: float | None
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def probe_video(path: str | Path) -> VideoProbe:
    """Read container metadata and verify at least one frame actually decodes.

    ``CAP_PROP_FRAME_COUNT`` is a container hint and is known to be wrong for
    some encoders, so a decodable video whose reported count is unusable is
    still flagged rather than trusted.
    """
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return VideoProbe(path, False, None, None, None, None, None, "cv2.VideoCapture could not open file")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = cap.read()
        if not ok:
            return VideoProbe(path, False, fps or None, n or None, w or None, h or None, None,
                              "opened but first frame did not decode")
        duration = (n / fps) if (fps and fps > 0 and n and n > 0) else None
        return VideoProbe(path, True, fps if fps > 0 else None, n if n > 0 else None,
                          w or None, h or None, duration)
    except Exception as exc:  # pragma: no cover - defensive
        return VideoProbe(path, False, None, None, None, None, None, repr(exc))
    finally:
        cap.release()


def uniform_indices(num_available: int, num_requested: int) -> list[int]:
    """i_k = round(k(N-1)/(T-1)) for k = 0..T-1  (spec section 8).

    Degenerate cases are handled explicitly rather than by luck:
      * T == 1        -> the middle frame
      * N < T         -> indices repeat; the temporal length stays T so that
                         every clip yields the same tensor shape
    """
    if num_available <= 0:
        raise ValueError("num_available must be positive")
    if num_requested <= 0:
        raise ValueError("num_requested must be positive")
    if num_requested == 1:
        return [num_available // 2]
    step = (num_available - 1) / (num_requested - 1)
    return [int(round(k * step)) for k in range(num_requested)]


def read_frames(path: str | Path, num_frames: int) -> tuple[np.ndarray, list[int], VideoProbe]:
    """Decode ``num_frames`` uniformly spaced BGR frames.

    Returns (frames [T,H,W,3] uint8, source indices, probe).

    Sequential decoding is used rather than per-frame ``CAP_PROP_POS_FRAMES``
    seeking: seeking in these MPEG-4 AVIs lands on the nearest keyframe and
    silently returns the wrong frame, which would corrupt the temporal spacing
    without raising anything.
    """
    probe = probe_video(path)
    if not probe.decodable:
        raise RuntimeError(f"cannot decode {path}: {probe.error}")

    cap = cv2.VideoCapture(str(path))
    try:
        decoded: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded.append(frame)
        if not decoded:
            raise RuntimeError(f"no frames decoded from {path}")

        n_actual = len(decoded)
        idx = uniform_indices(n_actual, num_frames)
        frames = np.stack([decoded[i] for i in idx], axis=0)
        # Record what actually decoded, which may differ from the container hint.
        probe.frame_count = n_actual
        if probe.fps and probe.fps > 0:
            probe.duration_s = n_actual / probe.fps
        return frames, idx, probe
    finally:
        cap.release()
