"""Discovery of DAiSEE clips and loading of the label CSVs.

Nothing here knows about specific subject IDs or clip counts: the split list, the
subjects and the clips are all discovered by walking the directory tree, so the
same code runs unchanged on the 108-clip sample and on the full 9,068-clip DAiSEE.

Expected (official DAiSEE) layout::

    <input_root>/DataSet/<Split>/<SubjectID>/<ClipID>/<ClipID>.avi
    <input_root>/Labels/<Split>Labels.csv
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

LABEL_COLUMNS = ["Boredom", "Engagement", "Confusion", "Frustration"]


@dataclass(frozen=True)
class ClipRecord:
    """One video clip and everything needed to locate its inputs and outputs."""

    clip_id: str
    subject_id: str
    split: str
    video_path: Path

    @property
    def relative_dir(self) -> Path:
        """Path fragment mirrored into every output tree."""
        return Path(self.split) / self.subject_id / self.clip_id


def find_dataset_dir(input_root: Path) -> Path:
    """Return the directory holding the split folders.

    Accepts either the dataset root (which contains ``DataSet/``) or the
    ``DataSet`` directory itself, so the caller can pass whichever they have.
    """
    input_root = Path(input_root)
    candidates = [input_root / "DataSet", input_root / "Dataset", input_root]
    for candidate in candidates:
        if candidate.is_dir() and any(
            (candidate / s).is_dir() for s in ("Train", "Test", "Validation")
        ):
            return candidate
    raise FileNotFoundError(
        f"Could not locate a DataSet directory with Train/Test/Validation under {input_root}"
    )


def find_labels_dir(input_root: Path) -> Optional[Path]:
    input_root = Path(input_root)
    for candidate in (input_root / "Labels", input_root.parent / "Labels"):
        if candidate.is_dir():
            return candidate
    return None


def discover_clips(
    input_root: Path,
    splits: Sequence[str] = ("Train", "Validation", "Test"),
    video_extensions: Sequence[str] = (".avi", ".mp4", ".mov", ".mkv", ".webm"),
    limit: int = 0,
) -> List[ClipRecord]:
    """Walk the dataset tree and return every clip found, sorted deterministically.

    A clip directory is expected to contain exactly one video file; if it holds
    several, the first by sorted name wins and a warning is emitted. Directories
    with no video at all are reported and skipped rather than raising, because on
    the full dataset a handful of corrupt downloads should not abort the run.
    """
    dataset_dir = find_dataset_dir(input_root)
    extensions = {e.lower() for e in video_extensions}
    records: List[ClipRecord] = []

    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            logger.warning("Split directory missing, skipping: %s", split_dir)
            continue

        for subject_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for clip_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
                videos = sorted(
                    p
                    for p in clip_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in extensions
                )
                if not videos:
                    logger.warning("No video file in clip directory: %s", clip_dir)
                    continue
                if len(videos) > 1:
                    logger.warning(
                        "%d video files in %s, using %s", len(videos), clip_dir, videos[0].name
                    )
                records.append(
                    ClipRecord(
                        clip_id=clip_dir.name,
                        subject_id=subject_dir.name,
                        split=split,
                        video_path=videos[0],
                    )
                )

    records.sort(key=lambda r: (r.split, r.subject_id, r.clip_id))
    if limit:
        records = records[:limit]
    logger.info(
        "Discovered %d clips across %d splits under %s",
        len(records),
        len({r.split for r in records}),
        dataset_dir,
    )
    return records


def load_labels(input_root: Path, splits: Iterable[str]) -> Dict[str, pd.DataFrame]:
    """Load ``<Split>Labels.csv`` for each split, keyed by split name.

    The shipped DAiSEE CSVs have a trailing space in the ``Frustration`` header
    and store ClipID with the file extension attached (``1100011002.avi``), so
    both are normalised here: columns are stripped and a bare ``ClipID`` stem
    column is produced for joining against discovered clip directory names.
    """
    labels_dir = find_labels_dir(input_root)
    out: Dict[str, pd.DataFrame] = {}
    if labels_dir is None:
        logger.warning("No Labels directory found under %s; clips will be unlabelled", input_root)
        return out

    for split in splits:
        csv_path = labels_dir / f"{split}Labels.csv"
        if not csv_path.is_file():
            logger.warning("Missing label file: %s", csv_path)
            continue
        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip() for c in df.columns]
        if "ClipID" not in df.columns:
            logger.warning("No ClipID column in %s (found %s)", csv_path, list(df.columns))
            continue
        df["ClipID"] = df["ClipID"].astype(str).str.strip()
        # Strip any video extension so the key matches the clip directory name.
        df["ClipID"] = df["ClipID"].str.replace(
            r"\.(avi|mp4|mov|mkv|webm)$", "", regex=True, case=False
        )
        for col in LABEL_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            else:
                logger.warning("Label column %s missing from %s", col, csv_path)
        df = df.drop_duplicates(subset="ClipID", keep="first")
        out[split] = df
        logger.info("Loaded %d label rows for split %s", len(df), split)
    return out


def label_lookup(labels: Dict[str, pd.DataFrame]) -> Dict[tuple, Dict[str, object]]:
    """Flatten the per-split label frames into a ``(split, clip_id) -> dict`` map."""
    table: Dict[tuple, Dict[str, object]] = {}
    for split, df in labels.items():
        for row in df.to_dict(orient="records"):
            clip_id = str(row["ClipID"])
            table[(split, clip_id)] = {
                col: (None if pd.isna(row.get(col)) else int(row[col]))
                for col in LABEL_COLUMNS
                if col in row
            }
    return table
