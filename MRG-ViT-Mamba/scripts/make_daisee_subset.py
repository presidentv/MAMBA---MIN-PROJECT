"""Standalone DAiSEE subset builder. Copy this one file anywhere and run it.

Unlike scripts/build_subset.py this imports nothing from the project, so it
works on a machine where the repository has not been cloned. It does the same
job: undersample the common engagement classes by taking clips round-robin
across subjects, leave the evaluation splits alone, and write a real DAiSEE
root that the training pipeline reads unchanged.

Defaults are set for the machine holding the data:

    source  D:\\Mini project\\DAiSEE
    output  D:\\Mini project\\daisee_subset

Run it with no arguments to see the plan without writing:

    python make_daisee_subset.py --dry-run

Then, to build it:

    python make_daisee_subset.py

Then verify, which re-reads the output from scratch rather than trusting the
build:

    python make_daisee_subset.py --verify-only

--------------------------------------------------------------------------
What it does, and why
--------------------------------------------------------------------------
Counting is done on MERGED classes: DAiSEE engagement levels 0 and 1 are
treated as one class, because level 0 has only 34 training clips and 4 test
clips. The label CSVs written out keep the ORIGINAL 4-level values, so the
merge stays a decision your training config makes at load time and nothing is
baked irreversibly into the data.

Only the training split is subsampled. Validation and Test are copied whole.
Subsampling an evaluation split changes the baseline it is measured against
(balancing DAiSEE's test split moves majority-class accuracy from 0.494 to
0.333), throws away most of the statistical power, and makes the result
incomparable to published work.

The rare class is kept whole and only the common ones are capped. Matching
every class to the rarest would cap training at ~136 clips.

Selection is round-robin across subjects, not random. The training split is
5,357 clips from only 69 subjects and the clips per subject are very uneven, so
random undersampling drops whole subjects as a side effect. Subject diversity,
not clip count, is what limits generalisation on a 69-subject corpus.

Videos are placed by hard link, so the subset costs no extra disk space when it
is on the same drive as the dataset (D: -> D: here). Use --link copy to
duplicate the files instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- #
# Defaults for this machine
# --------------------------------------------------------------------------- #
SOURCE = r"D:\Mini project\DAiSEE"
OUTPUT = r"D:\Mini project\daisee_subset"

# How the four DAiSEE engagement levels are grouped for counting purposes.
# [[0, 1], [2], [3]] -> three classes: Low, High, Very High.
CLASS_MERGE = [[0, 1], [2], [3]]
CLASS_NAMES = ["Low", "High", "Very High"]

# Clips per merged class in the training split. A class with fewer is kept
# whole; it is never padded or duplicated. 1200 keeps ~2,647 of 5,357 training
# clips, which roughly halves the epoch time while leaving the class prior close
# to the real one.
DEFAULT_CAP = 1200

SPLITS = ("train", "val", "test")
VIDEO_EXTENSIONS = (".avi", ".mp4")
LABEL_COLUMNS = ("Boredom", "Engagement", "Confusion", "Frustration")

# Directory names DAiSEE has shipped under, lower-cased. The release uses
# Train / Validation / Test, but casing has varied between copies, so the
# layout is discovered rather than assumed.
SPLIT_DIR_CANDIDATES = {
    "train": ("train",),
    "val": ("validation", "val"),
    "test": ("test",),
}
LABEL_FILE_CANDIDATES = {
    "train": ("trainlabels.csv",),
    "val": ("validationlabels.csv", "vallabels.csv"),
    "test": ("testlabels.csv",),
}

# Published DAiSEE engagement counts, for comparison only. The script always
# uses what it measures on disk.
PUBLISHED = {"train": [34, 213, 2617, 2494],
             "val": [23, 143, 813, 450],
             "test": [4, 84, 882, 814]}


class Clip:
    __slots__ = ("split", "subject", "clip_id", "stem", "path", "engagement")

    def __init__(self, split, subject, clip_id, stem, path, engagement):
        self.split = split
        self.subject = subject
        self.clip_id = clip_id
        self.stem = stem
        self.path = path
        self.engagement = engagement          # the ORIGINAL 0..3 level


# --------------------------------------------------------------------------- #
# Reading the dataset
# --------------------------------------------------------------------------- #
def find_dir(parent: Path, names) -> Path | None:
    """Case-insensitive lookup of a child directory."""
    if not parent.is_dir():
        return None
    lowered = {p.name.lower(): p for p in parent.iterdir() if p.is_dir()}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def find_file(parent: Path, names) -> Path | None:
    if not parent.is_dir():
        return None
    lowered = {p.name.lower(): p for p in parent.iterdir() if p.is_file()}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def locate_layout(root: Path) -> dict:
    """Find the DataSet/<Split> directories and the label CSVs."""
    data_dir = find_dir(root, ("dataset", "data_set", "data"))
    if data_dir is None:
        sys.exit(f"no DataSet directory under {root}\n"
                 f"  expected <root>/DataSet/Train/<Subject>/<Clip>/<Clip>.avi")
    labels_dir = find_dir(root, ("labels", "label"))
    if labels_dir is None:
        sys.exit(f"no Labels directory under {root}")

    layout = {"data_dir": data_dir, "labels_dir": labels_dir,
              "split_dirs": {}, "label_files": {}}
    for split in SPLITS:
        split_dir = find_dir(data_dir, SPLIT_DIR_CANDIDATES[split])
        label_file = find_file(labels_dir, LABEL_FILE_CANDIDATES[split])
        if split_dir is None:
            sys.exit(f"no {split} directory under {data_dir}; found "
                     f"{sorted(p.name for p in data_dir.iterdir() if p.is_dir())}")
        if label_file is None:
            sys.exit(f"no {split} label CSV under {labels_dir}; found "
                     f"{sorted(p.name for p in labels_dir.iterdir() if p.is_file())}")
        layout["split_dirs"][split] = split_dir
        layout["label_files"][split] = label_file
    return layout


def read_labels(csv_path: Path) -> tuple[dict, list[str], list[str]]:
    """Parse a DAiSEE label CSV.

    Three quirks in the shipped files are handled rather than worked around:
    the final header has a trailing space ("Frustration "), the ClipID column
    carries the file extension, and the files may begin with a UTF-8 BOM.
    Returns {clip_id: {label: value}}, the raw header row, and any problems.
    """
    problems: list[str] = []
    labels: dict[str, dict[str, int]] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        sys.exit(f"{csv_path} is empty")
    header = rows[0]
    clean_header = [h.strip() for h in header]
    if clean_header[0] != "ClipID":
        problems.append(f"{csv_path.name}: first column is {clean_header[0]!r}, "
                        f"expected 'ClipID'")
    index = {name: i for i, name in enumerate(clean_header)}
    for lineno, row in enumerate(rows[1:], start=2):
        if not row or not row[0].strip():
            continue
        clip_id = row[0].strip()
        values = {}
        for name in LABEL_COLUMNS:
            if name not in index or index[name] >= len(row):
                problems.append(f"{csv_path.name}:{lineno}: missing {name} for {clip_id}")
                continue
            raw = row[index[name]].strip()
            try:
                values[name] = int(raw)
            except ValueError:
                problems.append(f"{csv_path.name}:{lineno}: "
                                f"non-integer {name}={raw!r} for {clip_id}")
        labels[clip_id] = values
    return labels, header, problems


def scan_split(split: str, split_dir: Path, label_file: Path) -> tuple[list[Clip], list[str]]:
    """Join the videos on disk with their label rows."""
    labels, _, problems = read_labels(label_file)
    # Index by stem as well: for at least one clip the CSV and the directory
    # tree disagree on the extension.
    by_stem = {Path(k).stem: (k, v) for k, v in labels.items()}

    clips: list[Clip] = []
    matched: set[str] = set()
    for path in sorted(split_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        rel = path.relative_to(split_dir)
        # The subject is the directory directly under the split directory --
        # that is the anonymised user id, and what makes the splits disjoint.
        subject = rel.parts[0] if len(rel.parts) > 1 else "UNKNOWN"
        key, entry = path.name, labels.get(path.name)
        if entry is None:
            found = by_stem.get(path.stem)
            if found is not None:
                key, entry = found
        if entry is None:
            problems.append(f"{split}: no label row for {path.name}")
            continue
        engagement = entry.get("Engagement")
        if engagement is None:
            problems.append(f"{split}: no Engagement value for {path.name}")
            continue
        matched.add(key)
        clips.append(Clip(split, subject, path.name, path.stem, path, int(engagement)))

    orphans = sorted(set(labels) - matched)
    if orphans:
        problems.append(f"{split}: {len(orphans)} label rows have no video file "
                        f"(normal for a partial copy)")
    return clips, problems


# --------------------------------------------------------------------------- #
# Choosing what to keep
# --------------------------------------------------------------------------- #
def merge_map(groups) -> dict[int, int]:
    mapping = {}
    for new, group in enumerate(groups):
        for raw in group:
            mapping[int(raw)] = new
    missing = sorted(set(range(4)) - set(mapping))
    if missing:
        sys.exit(f"CLASS_MERGE does not cover engagement levels {missing}")
    return mapping


def select_round_robin(clips: list[Clip], target: int, seed: int) -> list[Clip]:
    """Up to `target` clips, taken one per subject per pass.

    Each pass takes one clip from every subject that still has clips, so the
    selection is as subject-diverse as the data allows at any size.
    """
    rng = random.Random(seed)
    by_subject: dict[str, list[Clip]] = defaultdict(list)
    for clip in clips:
        by_subject[clip.subject].append(clip)
    for subject in by_subject:
        rng.shuffle(by_subject[subject])

    order = sorted(by_subject)                 # deterministic before shuffling
    rng.shuffle(order)
    chosen: list[Clip] = []
    cursor = 0
    while len(chosen) < target:
        took = False
        for subject in order:
            bucket = by_subject[subject]
            if cursor < len(bucket):
                chosen.append(bucket[cursor])
                took = True
                if len(chosen) >= target:
                    break
        if not took:
            break                              # every subject exhausted
        cursor += 1
    return chosen


def plan(clips: list[Clip], mapping: dict[int, int], num_classes: int,
         per_class: list[int] | None, cap: int | None, keep_all: set[int],
         seed: int, subsample: bool) -> tuple[list[Clip], dict]:
    by_class: dict[int, list[Clip]] = {c: [] for c in range(num_classes)}
    for clip in clips:
        by_class[mapping[clip.engagement]].append(clip)

    chosen: list[Clip] = []
    detail: dict[int, dict] = {}
    for c in range(num_classes):
        available = by_class[c]
        if not subsample:
            target, reason = len(available), "split left whole"
        elif c in keep_all:
            target, reason = len(available), "kept whole (--keep-all)"
        elif per_class is not None:
            target, reason = min(per_class[c], len(available)), f"--per-class {per_class[c]}"
        elif cap is not None:
            target, reason = min(cap, len(available)), f"--cap {cap}"
        else:
            target, reason = len(available), "no target given"

        picked = (available if target >= len(available)
                  else select_round_robin(available, target, seed + c))
        chosen.extend(picked)
        detail[c] = {
            "available": len(available), "selected": len(picked),
            "reason": reason,
            "subjects_available": len({x.subject for x in available}),
            "subjects_selected": len({x.subject for x in picked}),
        }
    return chosen, detail


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def place(clip: Clip, split_dir: Path, out_split_dir: Path, mode: str) -> None:
    rel = clip.path.relative_to(split_dir)
    dst = out_split_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(clip.path, dst)
    elif mode == "symlink":
        os.symlink(clip.path, dst)
    else:
        try:
            os.link(clip.path, dst)
        except OSError as exc:
            sys.exit(f"hard link failed: {exc}\n"
                     f"  The subset must be on the same drive as the dataset for "
                     f"--link hard.\n  Re-run with --link copy (uses full disk "
                     f"space) or --link symlink.")


def write_label_csv(src_label_file: Path, out_path: Path, keep: set[str]) -> int:
    """Copy the label CSV, filtered to the selected clips.

    Rows are written back byte-for-byte as they were read, including the
    original 4-level Engagement value and the header's trailing space, so the
    class merge remains something the training config applies at load time.
    """
    with open(src_label_file, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    kept = [r for r in body
            if r and r[0].strip() and (r[0].strip() in keep
                                       or Path(r[0].strip()).stem in keep)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(kept)
    return len(kept)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify(out_root: Path, mapping: dict[int, int], num_classes: int) -> int:
    """Re-read the output from scratch and check it, trusting nothing."""
    print(f"\nverifying {out_root}")
    layout = locate_layout(out_root)
    failures: list[str] = []

    for split in SPLITS:
        clips, problems = scan_split(split, layout["split_dirs"][split],
                                     layout["label_files"][split])
        labels, _, _ = read_labels(layout["label_files"][split])
        counts: dict[int, int] = {}
        for clip in clips:
            c = mapping[clip.engagement]
            counts[c] = counts.get(c, 0) + 1
        raw_levels = sorted({clip.engagement for clip in clips})
        subjects = {clip.subject for clip in clips}

        print(f"  {split:5s} {len(clips):5d} clips  {len(subjects):3d} subjects  "
              f"counts {dict(sorted(counts.items()))}  "
              f"original levels present {raw_levels}")

        if len(clips) != len(labels):
            failures.append(f"{split}: {len(clips)} videos but {len(labels)} label rows")
        for p in problems:
            if "no label row" in p or "no Engagement" in p:
                failures.append(p)

    # Subject disjointness is the property that makes the evaluation valid at
    # all, so it is checked on the output and not assumed to have survived.
    per_split = {}
    for split in SPLITS:
        clips, _ = scan_split(split, layout["split_dirs"][split],
                              layout["label_files"][split])
        per_split[split] = {c.subject for c in clips}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = per_split[a] & per_split[b]
        if shared:
            failures.append(f"subjects appear in both {a} and {b}: {sorted(shared)[:5]}")
    print(f"  subject overlap: "
          f"train/val {len(per_split['train'] & per_split['val'])}, "
          f"train/test {len(per_split['train'] & per_split['test'])}, "
          f"val/test {len(per_split['val'] & per_split['test'])}")

    # Open one video to confirm the links resolve to readable files.
    try:
        import cv2
        sample, _ = scan_split("train", layout["split_dirs"]["train"],
                               layout["label_files"]["train"])
        for clip in (sample[0], sample[-1]):
            cap = cv2.VideoCapture(str(clip.path))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, _ = cap.read()
            cap.release()
            print(f"  readable: {clip.clip_id}  {frames} frames  first frame {'OK' if ok else 'FAILED'}")
            if not ok:
                failures.append(f"{clip.clip_id} could not be decoded at its new path")
    except ImportError:
        print("  (opencv not installed here; skipped the decode check)")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nverification passed")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SOURCE, help=f"DAiSEE root (default: {SOURCE})")
    ap.add_argument("--out", default=OUTPUT, help=f"subset root (default: {OUTPUT})")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP,
                    help=f"max training clips per merged class (default: {DEFAULT_CAP})")
    ap.add_argument("--per-class", type=int, nargs="+", default=None,
                    help="explicit per-class training targets instead of --cap")
    ap.add_argument("--keep-all", type=int, nargs="*", default=None,
                    help="merged classes never subsampled (default: none; a class "
                         "below the cap is kept whole anyway)")
    ap.add_argument("--splits", nargs="+", default=["train"], choices=list(SPLITS),
                    help="splits to subsample (default: train only)")
    ap.add_argument("--link", choices=("hard", "symlink", "copy"), default="hard",
                    help="how videos are placed (default: hard link, no extra disk)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    ap.add_argument("--verify-only", action="store_true",
                    help="check an existing subset and exit")
    args = ap.parse_args()

    src_root = Path(args.source).expanduser()
    out_root = Path(args.out).expanduser()
    mapping = merge_map(CLASS_MERGE)
    num_classes = len(CLASS_MERGE)

    if args.verify_only:
        if not out_root.is_dir():
            sys.exit(f"{out_root} does not exist")
        return verify(out_root, mapping, num_classes)

    if not src_root.is_dir():
        sys.exit(f"source dataset not found: {src_root}")
    if args.per_class is not None and len(args.per_class) != num_classes:
        sys.exit(f"--per-class needs {num_classes} numbers, got {len(args.per_class)}")

    print(f"source : {src_root}")
    print(f"output : {out_root}")
    print(f"merge  : {CLASS_MERGE} -> {CLASS_NAMES}")

    layout = locate_layout(src_root)
    print(f"layout : {layout['data_dir'].name}/"
          f"{{{', '.join(layout['split_dirs'][s].name for s in SPLITS)}}}  +  "
          f"{layout['labels_dir'].name}/")

    all_problems: list[str] = []
    selected: dict[str, list[Clip]] = {}
    summary = {
        "source_root": str(src_root), "output_root": str(out_root),
        "seed": args.seed, "link_mode": args.link,
        "class_merge": CLASS_MERGE, "class_names": CLASS_NAMES,
        "cap": args.cap, "per_class": args.per_class,
        "subsampled_splits": list(args.splits), "splits": {},
    }

    for split in SPLITS:
        clips, problems = scan_split(split, layout["split_dirs"][split],
                                     layout["label_files"][split])
        all_problems.extend(problems)
        if not clips:
            sys.exit(f"{split}: no clips found under {layout['split_dirs'][split]}")

        raw_counts = [sum(1 for c in clips if c.engagement == lvl) for lvl in range(4)]
        if raw_counts != PUBLISHED[split]:
            print(f"\nnote: {split} counts {raw_counts} differ from the published "
                  f"{PUBLISHED[split]}; using what is on disk")

        chosen, detail = plan(clips, mapping, num_classes, args.per_class, args.cap,
                              set(args.keep_all or []), args.seed, split in args.splits)
        selected[split] = chosen

        after: dict[int, int] = {}
        for clip in chosen:
            c = mapping[clip.engagement]
            after[c] = after.get(c, 0) + 1
        total = sum(after.values()) or 1

        print(f"\n{split}: {len(clips)} -> {len(chosen)} clips, "
              f"{len({c.subject for c in clips})} -> "
              f"{len({c.subject for c in chosen})} subjects"
              f"{'' if split in args.splits else '   (left whole)'}")
        for c in range(num_classes):
            d = detail[c]
            print(f"   {c} {CLASS_NAMES[c]:<12} {d['available']:>5} -> {d['selected']:>5} clips"
                  f"   {d['subjects_available']:>3} -> {d['subjects_selected']:>3} subjects"
                  f"   [{d['reason']}]")
        print("   resulting class prior: " +
              "  ".join(f"{after.get(c, 0) / total:.3f}" for c in range(num_classes)))

        summary["splits"][split] = {
            "subsampled": split in args.splits,
            "clips_before": len(clips), "clips_after": len(chosen),
            "subjects_before": len({c.subject for c in clips}),
            "subjects_after": len({c.subject for c in chosen}),
            "original_level_counts_before": raw_counts,
            "merged_counts_after": {str(k): v for k, v in sorted(after.items())},
            "per_class": {str(k): v for k, v in detail.items()},
        }

    if all_problems:
        print(f"\n{len(all_problems)} dataset problem(s), first 5:")
        for p in all_problems[:5]:
            print(f"   {p}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    if out_root.exists() and any(out_root.iterdir()):
        sys.exit(f"\nrefusing to write into a non-empty directory: {out_root}\n"
                 f"  delete it or pass a different --out")
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\nwriting ({args.link} links)")
    for split in SPLITS:
        src_split = layout["split_dirs"][split]
        out_split = out_root / layout["data_dir"].name / src_split.name
        for clip in selected[split]:
            place(clip, src_split, out_split, args.link)
        keep = {c.clip_id for c in selected[split]} | {c.stem for c in selected[split]}
        rows = write_label_csv(
            layout["label_files"][split],
            out_root / layout["labels_dir"].name / layout["label_files"][split].name,
            keep)
        print(f"  {split:5s} {len(selected[split]):5d} clips, {rows:5d} label rows")

    # Counts per class only, never a clip-to-label mapping: that would
    # redistribute DAiSEE's per-clip annotations.
    with open(out_root / "subset_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out_root / 'subset_summary.json'}")

    code = verify(out_root, mapping, num_classes)
    if code == 0:
        print(f"\nTo train on it:\n"
              f"  set DAISEE_ROOT={out_root}\n"
              f"  python scripts/run_preprocessing.py --config configs/config_full_3class.yaml --stage 1\n"
              f"  python scripts/fit_mrs_stats.py     --config configs/config_full_3class.yaml\n"
              f"  python scripts/run_finetune.py      --config configs/config_full_3class.yaml "
              f"--run-name sub3class --logit-adjust --tta")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
