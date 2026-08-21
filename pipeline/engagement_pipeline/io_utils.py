"""Mirrored output paths and tabular writing.

Every phase writes into a tree that mirrors the input layout
(``<root>/<split>/<subject>/<clip>/``) so that an output file can always be
traced back to its source clip by path alone.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clip_output_dir(output_root: Path, split: str, subject_id: str, clip_id: str) -> Path:
    return Path(output_root) / split / subject_id / clip_id


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def frame_filename(sample_index: int, frame_index: int, extension: str = "png") -> str:
    """Name encodes both indices so a file is self-describing on disk."""
    return f"s{sample_index:04d}_f{frame_index:06d}.{extension.lstrip('.')}"


def save_frame(image: np.ndarray, path: Path, image_format: str = "png", jpeg_quality: int = 95) -> bool:
    params: List[int] = []
    if image_format.lower() in ("jpg", "jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    elif image_format.lower() == "png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    ok = cv2.imwrite(str(path), image, params)
    if not ok:
        logger.error("Failed to write frame image: %s", path)
    return bool(ok)


def write_table(df: pd.DataFrame, path: Path, prefer_parquet: bool = True) -> Path:
    """Write a DataFrame as parquet, falling back to CSV if pyarrow is missing.

    Returns the path actually written, which the caller records in the manifest
    so downstream consumers never have to guess the format.
    """
    path = Path(path)
    ensure_dir(path.parent)
    if prefer_parquet:
        try:
            parquet_path = path.with_suffix(".parquet")
            df.to_parquet(parquet_path, index=False)
            return parquet_path
        except Exception as exc:  # pyarrow absent or a column type it dislikes
            logger.warning("Parquet write failed for %s (%s); falling back to CSV", path, exc)
    csv_path = path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def find_table(directory: Path, stem: str) -> Optional[Path]:
    """Locate ``<stem>.parquet`` or ``<stem>.csv`` in a directory."""
    directory = Path(directory)
    for suffix in (".parquet", ".csv"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def append_rows(rows: Iterable[dict]) -> pd.DataFrame:
    rows = list(rows)
    return pd.DataFrame(rows) if rows else pd.DataFrame()
