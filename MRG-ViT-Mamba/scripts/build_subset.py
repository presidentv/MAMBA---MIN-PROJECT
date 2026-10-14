"""Build a smaller DAiSEE root by subject-stratified undersampling.

Why this is not just "take the first N clips". DAiSEE's training split has 5,357
clips from only 69 subjects, and the clips per subject are very uneven. Random
undersampling to 2,000 clips drops whole subjects as a side effect, and subject
diversity -- not clip count -- is what limits generalisation on a 69-subject
corpus: 2,000 clips from 69 students teaches more than 2,000 clips from 25. This
script therefore fills each class by taking clips round-robin across subjects,
so every subject that contributes to a class keeps contributing until its clips
run out.

What it does NOT do, by default:

*   It does not touch val or test. Subsampling an evaluation split changes the
    baseline it is measured against (balancing DAiSEE's test split moves
    majority-class accuracy from 0.494 to 0.333), discards most of the
    statistical power, and makes the result incomparable to published work.
    Use ``--splits`` to override, deliberately.
*   It does not equalise the classes. Class 0 has 34 training clips; matching
    everything to it leaves 136 clips. The sensible move is to keep the rare
    classes whole and cap only the common ones, which is what ``--cap`` does.

Output is a real DAiSEE root -- ``DataSet/<Split>/<Subject>/<Clip>/<Clip>.avi``
plus ``Labels/<Split>Labels.csv`` -- so the rest of the pipeline runs against it
unchanged by pointing ``DAISEE_ROOT`` at it. Label CSV rows are copied verbatim
from the originals, so all four affect labels survive and any class merge is
still applied downstream by the config, not baked into the data.

Examples::

    # cap the two common classes at 1200, keep the rare ones whole
    python scripts/build_subset.py --config configs/config_full_3class.yaml \\
        --out /data/DAiSEE_cap1200 --cap 1200

    # explicit per-class targets, on merged 3-class labels
    python scripts/build_subset.py --config configs/config_full_3class.yaml \\
        --out /data/DAiSEE_sub --per-class 247 900 850

    # see the plan without writing anything
    python scripts/build_subset.py --config configs/config_full.yaml --cap 1200 --dry-run

Linking mode matters for disk: ``--link hard`` (default) costs no extra space on
the same filesystem, ``--link symlink`` likewise but needs privileges on
Windows, ``--link copy`` duplicates the videos.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.class_merge import class_names as merged_class_names  # noqa: E402
from src.dataset_index import SPLITS, build_index, label_distribution  # noqa: E402
from src.utils import load_config, resolve_path, save_json  # noqa: E402


def select_round_robin(records, target: int, seed: int) -> list:
    """Up to ``target`` records, taken round-robin across subjects.

    Each pass takes one clip from every subject that still has clips, so the
    selection is as subject-diverse as the data allows at any size. Within a
    subject the order is shuffled with ``seed`` so the choice is reproducible
    but not an artefact of directory order.
    """
    import random

    rng = random.Random(seed)
    by_subject: dict[str, list] = defaultdict(list)
    for rec in records:
        by_subject[rec.subject_id].append(rec)
    for subject in by_subject:
        rng.shuffle(by_subject[subject])

    order = sorted(by_subject)                      # deterministic subject order
    rng.shuffle(order)
    chosen: list = []
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
            break                                   # every subject exhausted
        cursor += 1
    return chosen


def plan_split(records, num_classes: int, per_class: list[int] | None,
               cap: int | None, keep_all: set[int], seed: int) -> tuple[list, dict]:
    """Choose the records to keep for one split, and describe the decision."""
    by_class: dict[int, list] = {c: [] for c in range(num_classes)}
    for rec in records:
        label = rec.engagement
        if label is not None:
            by_class[int(label)].append(rec)

    chosen, detail = [], {}
    for c in range(num_classes):
        available = by_class[c]
        if c in keep_all:
            target = len(available)
            reason = "kept whole (--keep-all)"
        elif per_class is not None:
            target = min(int(per_class[c]), len(available))
            reason = f"--per-class {per_class[c]}"
        elif cap is not None:
            target = min(int(cap), len(available))
            reason = f"--cap {cap}"
        else:
            target = len(available)
            reason = "kept whole (no target given)"

        picked = (available if target >= len(available)
                  else select_round_robin(available, target, seed + c))
        chosen.extend(picked)
        detail[c] = {
            "available": len(available),
            "selected": len(picked),
            "requested": target,
            "reason": reason,
            "subjects_available": len({r.subject_id for r in available}),
            "subjects_selected": len({r.subject_id for r in picked}),
        }
    return chosen, detail


def link_clip(rec, src_root: Path, out_root: Path, split_dir: str, mode: str) -> None:
    """Reproduce one clip's DAiSEE path under the new root.

    ``split_dir`` comes from the config and already carries its "DataSet/"
    prefix, so it is joined to the output root directly. Adding another
    "DataSet" component here produced <out>/DataSet/DataSet/Train/... , which
    wrote every file successfully and then failed to be found by build_index.
    """
    src = Path(rec.path)
    rel = src.relative_to(src_root / split_dir)
    dst = out_root / split_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError as exc:            # different volume, or no hardlink support
            raise SystemExit(
                f"hard link failed ({exc}). The subset must sit on the same "
                f"filesystem as the dataset for --link hard; use --link copy "
                f"or --link symlink.") from exc


def write_labels(src_label_file: Path, out_root: Path, rel_out: str,
                 keep_ids: set[str]) -> int:
    """Copy the label CSV, filtered to the selected clips.

    Rows are written back with their original values -- including the raw
    4-level Engagement -- so any class merge stays a config decision applied at
    load time rather than something baked irreversibly into the subset.
    """
    out_path = out_root / rel_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(src_label_file, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    header, body = rows[0], rows[1:]
    id_col = 0
    kept = [r for r in body if r and (r[id_col].strip() in keep_ids
                                      or Path(r[id_col].strip()).stem in keep_ids)]
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(kept)
    return len(kept)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/config_full.yaml",
                    help="config giving the dataset root and any class_merge")
    ap.add_argument("--out", default=None,
                    help="directory for the new DAiSEE root (required unless --dry-run)")
    ap.add_argument("--cap", type=int, default=None,
                    help="maximum clips per class; classes with fewer are kept whole")
    ap.add_argument("--per-class", type=int, nargs="+", default=None,
                    help="explicit target per class, one number per class")
    ap.add_argument("--keep-all", type=int, nargs="*", default=None,
                    help="classes never subsampled (default: every class below --cap)")
    ap.add_argument("--splits", nargs="+", default=["train"], choices=list(SPLITS),
                    help="splits to subsample; val and test are left whole by default")
    ap.add_argument("--link", choices=("hard", "symlink", "copy"), default="hard",
                    help="how the videos are placed in the new root")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without writing")
    args = ap.parse_args()

    if args.cap is None and args.per_class is None:
        ap.error("give --cap or --per-class; otherwise the subset is the whole dataset")
    if not args.dry_run and not args.out:
        ap.error("--out is required unless --dry-run")

    cfg = load_config(resolve_path(args.config))
    src_root = resolve_path(cfg["paths"]["dataset_root"])
    print(f"source dataset : {src_root}")

    index = build_index(cfg)
    if not index.is_usable():
        print("dataset index has fatal problems:")
        for p in index.problems[:8]:
            print(f"  {p.kind}: {p.detail}")
        return 1

    num_classes = int(cfg["dataset"]["num_classes"])
    names = merged_class_names(cfg, num_classes)
    if index.class_merge:
        print(f"class_merge    : {index.class_merge['merge']['groups']} -> {list(names)}")
    if args.per_class is not None and len(args.per_class) != num_classes:
        ap.error(f"--per-class needs {num_classes} numbers, got {len(args.per_class)}")

    keep_all = set(args.keep_all or [])
    summary = {"source_root": str(src_root), "config": str(args.config),
               "seed": args.seed, "link_mode": args.link,
               "num_classes": num_classes, "class_names": list(names),
               "class_merge": index.class_merge,
               "cap": args.cap, "per_class": args.per_class,
               "keep_all": sorted(keep_all), "subsampled_splits": list(args.splits),
               "splits": {}}

    selected: dict[str, list] = {}
    for split in SPLITS:
        records = index.clips.get(split, [])
        if split in args.splits:
            chosen, detail = plan_split(records, num_classes, args.per_class,
                                        args.cap, keep_all, args.seed)
        else:
            chosen = list(records)
            detail = {c: {"available": n, "selected": n, "requested": n,
                          "reason": "split left whole",
                          "subjects_available": len({r.subject_id for r in records
                                                     if r.engagement == c}),
                          "subjects_selected": len({r.subject_id for r in records
                                                    if r.engagement == c})}
                      for c, n in label_distribution(records).items()}
        selected[split] = chosen

        before = label_distribution(records)
        after = label_distribution(chosen)
        summary["splits"][split] = {
            "subsampled": split in args.splits,
            "clips_before": len(records), "clips_after": len(chosen),
            "subjects_before": len({r.subject_id for r in records}),
            "subjects_after": len({r.subject_id for r in chosen}),
            "counts_before": {str(k): v for k, v in before.items()},
            "counts_after": {str(k): v for k, v in after.items()},
            "per_class": {str(k): v for k, v in detail.items()},
        }

        print(f"\n{split}: {len(records)} -> {len(chosen)} clips, "
              f"{len({r.subject_id for r in records})} -> "
              f"{len({r.subject_id for r in chosen})} subjects"
              f"{'' if split in args.splits else '   (left whole)'}")
        for c in range(num_classes):
            d = detail.get(c, {})
            if not d:
                continue
            print(f"   {c} {names[c]:<18} {d['available']:>5} -> {d['selected']:>5} clips"
                  f"   {d['subjects_available']:>3} -> {d['subjects_selected']:>3} subjects"
                  f"   [{d['reason']}]")
        total = sum(after.values()) or 1
        print("   resulting prior: " +
              "  ".join(f"{after.get(c, 0) / total:.3f}" for c in range(num_classes)))

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    out_root = Path(args.out).expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        print(f"\nrefusing to write into a non-empty directory: {out_root}")
        return 1
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\nwriting {out_root} ({args.link})")
    for split in SPLITS:
        split_dir = cfg["dataset"]["split_dirs"][split]
        for rec in selected[split]:
            link_clip(rec, src_root, out_root, split_dir, args.link)
        label_rel = str(cfg["dataset"]["label_files"][split])
        kept = write_labels(src_root / label_rel, out_root, label_rel,
                            {r.clip_id for r in selected[split]}
                            | {r.stem for r in selected[split]})
        print(f"  {split}: {len(selected[split])} clips, {kept} label rows")

    # Counts per class only -- never a clip-to-label mapping, which would
    # redistribute DAiSEE's per-clip annotations.
    save_json(summary, out_root / "subset_summary.json")
    print(f"\nwrote {out_root / 'subset_summary.json'}")
    print("\nNext, on this machine:")
    print(f"  export DAISEE_ROOT={out_root}")
    print("  python scripts/run_preprocess.py --config <config>      # Stage 1 cache")
    print("  python scripts/fit_mrs_stats.py  --config <config>      # MRS calibration")
    print("  python scripts/run_finetune.py   --config <config> --run-name <name>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
