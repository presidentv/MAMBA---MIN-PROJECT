"""Discovery and validation of the DAiSEE directory tree.

This module *inspects* the dataset. It never assumes a path layout, a clip
count, an FPS, or a label encoding -- everything is read off disk and reported.
If something required is absent, the caller is given a structured problem
description so it can stop (spec section 4, Rule 1, Rule 2).

DAiSEE layout, as documented in the dataset's own README.txt::

    DataSet/<Split>/<SubjectID>/<ClipID>/<ClipID>.avi
    Labels/<Split>Labels.csv    -> ClipID,Boredom,Engagement,Confusion,Frustration

The subject identifier is the directory one level above the clip directory,
which is what makes subject-disjointness checkable.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from .utils import resolve_path

SPLITS = ("train", "val", "test")


@dataclass
class ClipRecord:
    clip_id: str          # file name including extension, e.g. "1100011002.avi"
    stem: str             # "1100011002"
    split: str
    subject_id: str
    path: str
    labels: dict[str, int] = field(default_factory=dict)

    @property
    def engagement(self) -> int | None:
        return self.labels.get("Engagement")


@dataclass
class IndexProblem:
    kind: str
    detail: str
    items: list[str] = field(default_factory=list)


@dataclass
class DatasetIndex:
    root: str
    clips: dict[str, list[ClipRecord]]
    problems: list[IndexProblem]

    def all_clips(self) -> Iterable[ClipRecord]:
        for split in SPLITS:
            yield from self.clips.get(split, [])

    def subjects(self, split: str) -> set[str]:
        return {c.subject_id for c in self.clips.get(split, [])}

    def is_usable(self) -> bool:
        """True when nothing fatal was found. Non-fatal problems (a handful of
        label rows without a video file, which is normal for a subset copy) do
        not block the pipeline but are always reported."""
        fatal = {"missing_split_dir", "missing_label_file", "no_videos", "label_parse_error"}
        return not any(p.kind in fatal for p in self.problems)


def _find_videos(split_dir: Path, extensions: tuple[str, ...]) -> list[tuple[str, str, Path]]:
    """Return (subject_id, clip_file_name, path) for every video under split_dir.

    Walks the tree rather than assuming exactly two nesting levels, but derives
    the subject from the path component directly under the split directory --
    that is the anonymised user id per the DAiSEE README.
    """
    found: list[tuple[str, str, Path]] = []
    for path in sorted(split_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        rel = path.relative_to(split_dir)
        subject = rel.parts[0] if len(rel.parts) > 1 else "UNKNOWN"
        found.append((subject, path.name, path))
    return found


def _read_labels(csv_path: Path) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Parse a DAiSEE label CSV.

    The shipped files have a trailing space in the final header
    ("Frustration ") and the ClipID column carries the file extension. Both are
    handled here rather than silently mangled elsewhere.
    """
    labels: dict[str, dict[str, int]] = {}
    errors: list[str] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return labels, [f"{csv_path.name}: no header row"]
        headers = [h.strip() for h in reader.fieldnames]
        if headers[0] != "ClipID":
            errors.append(f"{csv_path.name}: first column is {headers[0]!r}, expected 'ClipID'")
        for lineno, row in enumerate(reader, start=2):
            clean = {k.strip(): (v.strip() if isinstance(v, str) else v)
                     for k, v in row.items() if k is not None}
            clip_id = clean.get("ClipID", "")
            if not clip_id:
                errors.append(f"{csv_path.name}:{lineno}: empty ClipID")
                continue
            values: dict[str, int] = {}
            for key in ("Boredom", "Engagement", "Confusion", "Frustration"):
                raw = clean.get(key)
                if raw is None or raw == "":
                    errors.append(f"{csv_path.name}:{lineno}: missing {key} for {clip_id}")
                    continue
                try:
                    values[key] = int(raw)
                except ValueError:
                    errors.append(f"{csv_path.name}:{lineno}: non-integer {key}={raw!r} for {clip_id}")
            labels[clip_id] = values
    return labels, errors


def build_index(cfg) -> DatasetIndex:
    """Scan the dataset root described by cfg and join videos with label rows."""
    root = resolve_path(cfg.paths["dataset_root"])
    extensions = tuple(e.lower() for e in cfg.dataset["video_extensions"])
    problems: list[IndexProblem] = []
    clips: dict[str, list[ClipRecord]] = {s: [] for s in SPLITS}

    if not root.exists():
        problems.append(IndexProblem("missing_dataset_root", f"dataset_root does not exist: {root}"))
        return DatasetIndex(str(root), clips, problems)

    for split in SPLITS:
        split_dir = root / cfg.dataset["split_dirs"][split]
        label_file = root / cfg.dataset["label_files"][split]

        if not split_dir.is_dir():
            problems.append(IndexProblem("missing_split_dir", f"{split}: {split_dir} not found"))
            continue
        if not label_file.is_file():
            problems.append(IndexProblem("missing_label_file", f"{split}: {label_file} not found"))
            continue

        labels, label_errors = _read_labels(label_file)
        if label_errors:
            problems.append(IndexProblem("label_parse_error", f"{split}: label CSV issues",
                                         label_errors))

        videos = _find_videos(split_dir, extensions)
        if not videos:
            problems.append(IndexProblem("no_videos", f"{split}: no video files under {split_dir}"))
            continue

        # Index label rows by both full file name and stem: the CSV and the
        # directory tree disagree on extension for at least one clip in the
        # shipped data, so match on stem when the exact name is absent.
        labels_by_stem = {Path(k).stem: (k, v) for k, v in labels.items()}
        matched_keys: set[str] = set()

        for subject, file_name, path in videos:
            entry = labels.get(file_name)
            key = file_name
            if entry is None:
                by_stem = labels_by_stem.get(Path(file_name).stem)
                if by_stem is not None:
                    key, entry = by_stem
            if entry is None:
                problems.append(IndexProblem(
                    "video_without_label", f"{split}: no label row for {file_name}", [str(path)]))
                continue
            matched_keys.add(key)
            clips[split].append(ClipRecord(
                clip_id=file_name,
                stem=Path(file_name).stem,
                split=split,
                subject_id=subject,
                path=str(path),
                labels=entry,
            ))

        orphan_labels = sorted(set(labels) - matched_keys)
        if orphan_labels:
            problems.append(IndexProblem(
                "label_without_video",
                f"{split}: {len(orphan_labels)} label rows have no video file on disk",
                orphan_labels))

    return DatasetIndex(str(root), clips, problems)


def label_distribution(records: list[ClipRecord], target: str = "Engagement") -> dict[int, int]:
    dist: dict[int, int] = {}
    for r in records:
        v = r.labels.get(target)
        if v is not None:
            dist[v] = dist.get(v, 0) + 1
    return dict(sorted(dist.items()))


def subject_overlaps(index: DatasetIndex) -> dict[str, list[str]]:
    tr, va, te = (index.subjects(s) for s in SPLITS)
    return {
        "train_val": sorted(tr & va),
        "train_test": sorted(tr & te),
        "val_test": sorted(va & te),
    }


def index_to_dict(index: DatasetIndex) -> dict:
    return {
        "root": index.root,
        "counts": {s: len(index.clips.get(s, [])) for s in SPLITS},
        "problems": [asdict(p) for p in index.problems],
    }
