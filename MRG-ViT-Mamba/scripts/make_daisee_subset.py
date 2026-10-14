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

The default is 1500 training clips, allocated in proportion to the real class
distribution. --total N changes that.

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

Only the training split is subsampled. Validation and Test are copied whole --
every one of the 1,429 validation and 1,784 test clips, in both modes. So the
test measurement itself is untouched: the same clips, the same 0.494
majority-class baseline, and a number still comparable with published work.
Shrinking the TRAINING split can of course still cost accuracy; what these
modes control is whether it costs anything beyond the reduced sample size.
Subsampling an evaluation split changes the baseline it is measured against
(balancing DAiSEE's test split moves majority-class accuracy from 0.494 to
0.333), throws away most of the statistical power, and makes the result
incomparable to published work.

There are two ways to size the training split, and they answer different
questions.

--cap N      caps each common class at N and keeps the rare class whole. Use it
             when the goal is to spend less time per epoch and the rare class
             needs every clip it has. It shifts the class distribution towards
             the rare class, which costs accuracy unless the shift is corrected
             at inference (run_finetune.py --logit-adjust does that).

--total N    allocates N clips across the classes IN PROPORTION to the real
             class distribution, with a floor of --min-per-class on each. This
             is the setting to use when test accuracy is what matters. A
             proportional, subject-stratified sample is an unbiased smaller
             draw from the same distribution: the training prior still matches
             the test prior, so the decision boundary is calibrated for the
             distribution it will actually be evaluated on, and the only thing
             lost is sample size. Nothing about the evaluation changes.

             Measured on the real distribution, --total 1500 gives 69/733/698
             and a prior that differs from the full split's by 0.001. Nearly
             all of the budget lands on High vs Very High, which is 95% of the
             test split and therefore where accuracy is won or lost.

--min-per-class N applies a floor under --total, for when macro-F1 matters more
             than accuracy. It costs a measurable prior shift -- 241/640/619
             and a drift of 0.115 at --total 1500 -- so pair it with
             run_finetune.py --logit-adjust. The script prints the before and
             after priors and the size of the drift, so the trade is never
             invisible.

Matching every class to the rarest would cap training at ~136 clips, which is
why neither mode does that.

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

# Default sizing: a prior-matched training subset of this many clips. Set to
# None, with --cap given instead, to cap the common classes instead.
DEFAULT_TOTAL = 1500

# Floor applied to every class under --total. 0 means strict proportionality,
# which is the right default when test ACCURACY is the goal: it keeps the
# training prior equal to the real one, so the decision boundary stays
# calibrated for the distribution it will be evaluated on.
#
# The cost is that the merged rare class gets its proportional share and no
# more -- 69 clips at --total 1500 -- so its own recall will be poor. That
# bounds the accuracy loss at the rare class's share of the test split, 4.9%,
# while the 95% of test clips that are High or Very High are backed by ~700
# training clips each, which is where accuracy is actually won.
#
# Raise it (--min-per-class 200) when macro-F1 matters more than accuracy. That
# shifts the prior measurably -- 0.115 at --total 1500 -- so pair it with
# run_finetune.py --logit-adjust, which corrects the shift at inference.
DEFAULT_MIN_PER_CLASS = 0

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


def allocate_proportional(counts: list[int], total: int,
                          min_per_class: int) -> list[int]:
    """Split `total` across classes in proportion to `counts`.

    Each class gets at least ``min_per_class`` (or everything it has, if that is
    fewer), and no class is asked for more than it has. The remainder after the
    floors is handed out by largest remainder on the real proportions, then any
    clips left over because a class ran out are redistributed to the classes
    that still have headroom. Returns the per-class targets.
    """
    k = len(counts)
    available = list(counts)
    alloc = [min(min_per_class, available[c]) for c in range(k)]

    if sum(alloc) > total:
        # The floors alone exceed the budget: honour the budget and scale the
        # floors down proportionally rather than silently overshooting.
        scale = total / sum(alloc)
        alloc = [int(a * scale) for a in alloc]
        return alloc

    remaining = total - sum(alloc)
    while remaining > 0:
        headroom = [available[c] - alloc[c] for c in range(k)]
        if not any(headroom):
            break
        weight_total = sum(counts[c] for c in range(k) if headroom[c] > 0)
        if weight_total <= 0:
            break
        # Proportional share of what is left, floored, then largest-remainder
        # for the rounding so the total lands exactly on the budget.
        shares = []
        for c in range(k):
            if headroom[c] <= 0:
                shares.append((0.0, c))
                continue
            shares.append((remaining * counts[c] / weight_total, c))
        given = 0
        for share, c in shares:
            take = min(int(share), headroom[c])
            alloc[c] += take
            given += take
        remaining -= given
        if given == 0:
            # Rounding left less than one clip per class; hand the rest to the
            # classes with headroom, largest proportion first.
            for _, c in sorted(shares, reverse=True):
                if remaining <= 0:
                    break
                if available[c] - alloc[c] > 0:
                    alloc[c] += 1
                    remaining -= 1
    return alloc


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
         per_class: list[int] | None, cap: int | None, total: int | None,
         min_per_class: int, keep_all: set[int], seed: int,
         subsample: bool) -> tuple[list[Clip], dict]:
    by_class: dict[int, list[Clip]] = {c: [] for c in range(num_classes)}
    for clip in clips:
        by_class[mapping[clip.engagement]].append(clip)

    proportional = None
    if subsample and total is not None and per_class is None:
        proportional = allocate_proportional(
            [len(by_class[c]) for c in range(num_classes)], total, min_per_class)

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
        elif proportional is not None:
            target = min(proportional[c], len(available))
            reason = (f"--total {total}, prior-matched"
                      + (f", floored at {min_per_class}" if target == min_per_class else ""))
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
    ap.add_argument("--total", type=int, default=None,
                    help=f"total training clips, allocated in proportion to the real "
                         f"class distribution (default: {DEFAULT_TOTAL} when neither "
                         f"--cap nor --per-class is given). Use this when test "
                         f"accuracy is what matters.")
    ap.add_argument("--min-per-class", type=int, default=DEFAULT_MIN_PER_CLASS,
                    help=f"floor per class under --total (default: "
                         f"{DEFAULT_MIN_PER_CLASS})")
    ap.add_argument("--cap", type=int, default=None,
                    help="max training clips per merged class, rare class kept whole; "
                         "shifts the class prior, so pair it with "
                         "run_finetune.py --logit-adjust")
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

    # Exactly one sizing mode. Defaulting to --total means the recommended
    # behaviour is what you get by running the script with no arguments.
    modes = [args.total is not None, args.cap is not None, args.per_class is not None]
    if sum(modes) > 1:
        sys.exit("give only one of --total, --cap, --per-class")
    if not any(modes):
        args.total = DEFAULT_TOTAL

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
    if args.total is not None:
        print(f"sizing : {args.total} training clips, prior-matched, "
              f"floor {args.min_per_class} per class")
    elif args.cap is not None:
        print(f"sizing : common classes capped at {args.cap}, rare class kept whole")
    else:
        print(f"sizing : explicit per-class targets {args.per_class}")

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
        "cap": args.cap, "per_class": args.per_class, "total": args.total,
        "min_per_class": args.min_per_class,
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
                              args.total, args.min_per_class,
                              set(args.keep_all or []), args.seed,
                              split in args.splits)
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
        merged_before: dict[int, int] = {}
        for clip in clips:
            c = mapping[clip.engagement]
            merged_before[c] = merged_before.get(c, 0) + 1
        before_total = sum(merged_before.values()) or 1
        print("   class prior before: " +
              "  ".join(f"{merged_before.get(c, 0) / before_total:.3f}"
                        for c in range(num_classes)))
        print("   class prior after : " +
              "  ".join(f"{after.get(c, 0) / total:.3f}" for c in range(num_classes)))
        if split in args.splits:
            drift = max(abs(after.get(c, 0) / total
                            - merged_before.get(c, 0) / before_total)
                        for c in range(num_classes))
            # The training prior is what the decision boundary is calibrated to,
            # so drift away from the real distribution is what costs accuracy --
            # not the smaller clip count by itself.
            print(f"   largest prior shift: {drift:.3f}"
                  + ("   (negligible; no correction needed)" if drift < 0.02 else
                     "   (train with run_finetune.py --logit-adjust to correct it)"))

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
