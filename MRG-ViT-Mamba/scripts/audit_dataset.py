"""CHECKPOINT 1 (dataset audit) and CHECKPOINT 2 (split audit).

Writes artifacts/dataset_audit.json and artifacts/label_mapping.json.
Exits non-zero if a blocking condition fails, so it can gate the rest of the
pipeline rather than being advisory (spec sections 4-6, 38).

Usage:
    python scripts/audit_dataset.py [--probe N] [--probe-all]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_index import (  # noqa: E402
    SPLITS, build_index, label_distribution, subject_overlaps,
)
from src.utils import (  # noqa: E402
    environment_record, get_logger, load_config, save_json,
)
from src.video_sampling import probe_video  # noqa: E402

# Conceptual DAiSEE engagement scale. This is the *documented* meaning; the
# script verifies that the encoded values actually observed match this domain
# and refuses to guess if they do not (spec section 5, Rule 2).
CONCEPTUAL_ENGAGEMENT_CLASSES = {0: "Very Low", 1: "Low", 2: "High", 3: "Very High"}


def _dist(values):
    return {str(k): v for k, v in sorted(Counter(values).items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=12,
                    help="how many videos per split to decode-probe (0 = none)")
    ap.add_argument("--probe-all", action="store_true",
                    help="decode-probe every video (slow, but required before a full run)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    log = get_logger("audit", "logs/audit_dataset.log")
    cfg = load_config(args.config)
    target = cfg.dataset["target_label"]

    log.info("Scanning dataset root from config: %s", cfg.paths["dataset_root"])
    index = build_index(cfg)
    log.info("Resolved dataset root: %s", index.root)

    blocking: list[str] = []

    # ---------------------------------------------------------------- counts
    print("\n" + "=" * 72)
    print("CHECKPOINT 1 - DATASET AUDIT")
    print("=" * 72)
    print(f"dataset_root : {index.root}")
    print(f"split_source : DAiSEE official DataSet/ directory split + Labels/*.csv")

    for split in SPLITS:
        n = len(index.clips.get(split, []))
        print(f"  {split:<6} clips: {n:>6}   subjects: {len(index.subjects(split)):>4}")
        if n == 0:
            blocking.append(f"split '{split}' resolved zero clips")

    # ------------------------------------------------------------- problems
    if index.problems:
        print("\nProblems found while indexing:")
        for p in index.problems:
            shown = ", ".join(p.items[:5]) + (" ..." if len(p.items) > 5 else "")
            print(f"  [{p.kind}] {p.detail}")
            if shown.strip():
                print(f"      {shown}")
    else:
        print("\nNo indexing problems.")

    if not index.is_usable():
        blocking.append("fatal indexing problem (see above)")

    # ---------------------------------------------------------------- labels
    print("\n--- Engagement label distribution (target label: %s) ---" % target)
    observed_values: set[int] = set()
    label_dists = {}
    for split in SPLITS:
        d = label_distribution(index.clips.get(split, []), target)
        label_dists[split] = d
        observed_values |= set(d)
        total = sum(d.values()) or 1
        pretty = "  ".join(f"{k}:{v} ({100*v/total:.1f}%)" for k, v in d.items())
        print(f"  {split:<6} {pretty}")

    print(f"\nDistinct encoded {target} values observed: {sorted(observed_values)}")
    expected = set(CONCEPTUAL_ENGAGEMENT_CLASSES)
    if not observed_values:
        blocking.append("no engagement labels were read")
    elif not observed_values <= expected:
        blocking.append(
            f"observed {target} values {sorted(observed_values)} are outside the documented "
            f"domain {sorted(expected)} - label encoding must be re-checked before training"
        )
    elif observed_values != expected:
        # Not fatal for the pipeline, but the classifier still has 4 outputs and
        # the absent class can never be predicted correctly. Say so loudly.
        print(f"WARNING: only {len(observed_values)} of the 4 documented engagement classes "
              f"appear in this copy of the dataset: missing {sorted(expected - observed_values)}")

    per_split_present = {s: sorted(label_dists[s]) for s in SPLITS}
    for split, vals in per_split_present.items():
        if len(vals) < 4:
            print(f"NOTE: split '{split}' contains only classes {vals} "
                  f"(missing {sorted(expected - set(vals))}). Per-class metrics for the "
                  f"absent classes will be undefined for this split.")

    # --------------------------------------------------------- split audit
    print("\n" + "=" * 72)
    print("CHECKPOINT 2 - SPLIT AUDIT")
    print("=" * 72)
    for split in SPLITS:
        subs = sorted(index.subjects(split))
        print(f"{split.capitalize()} subjects ({len(subs)}): {', '.join(subs)}")
    overlaps = subject_overlaps(index)
    print()
    for key, items in overlaps.items():
        a, b = key.split("_")
        print(f"{a.capitalize()}/{b.capitalize()} overlap: {len(items)}"
              + (f"  -> {items}" if items else ""))
        if items:
            blocking.append(f"subject leakage between {a} and {b}: {items}")

    all_subjects = set().union(*(index.subjects(s) for s in SPLITS))

    # ------------------------------------------------------------ decoding
    probes = []
    if args.probe_all or args.probe > 0:
        print("\n--- Decode probe ---")
        for split in SPLITS:
            records = index.clips.get(split, [])
            selected = records if args.probe_all else records[: args.probe]
            for rec in selected:
                probes.append((rec, probe_video(rec.path)))
        ok = [p for _, p in probes if p.decodable]
        bad = [(r, p) for r, p in probes if not p.decodable]
        print(f"probed {len(probes)} videos: {len(ok)} decodable, {len(bad)} failed")
        for rec, p in bad:
            print(f"  FAILED {rec.split}/{rec.clip_id}: {p.error}")
        if ok:
            fps = [p.fps for p in ok if p.fps]
            fc = [p.frame_count for p in ok if p.frame_count]
            res = [f"{p.width}x{p.height}" for p in ok if p.width]
            dur = [p.duration_s for p in ok if p.duration_s]
            print(f"  fps        : {_dist(fps)}")
            print(f"  frame_count: {_dist(fc)}")
            print(f"  resolution : {_dist(res)}")
            if dur:
                print(f"  duration_s : min={min(dur):.2f} mean={sum(dur)/len(dur):.2f} max={max(dur):.2f}")
        if bad:
            blocking.append(f"{len(bad)} probed videos failed to decode")
        # Show the first few resolved paths so the layout assumption is visible.
        print("\n  first resolved video paths:")
        for rec, _ in probes[:10]:
            print(f"    {rec.split:<5} {rec.subject_id:<8} {rec.clip_id:<18} {rec.path}")

    # ------------------------------------------------------------- artifacts
    ok_probes = [p for _, p in probes if p.decodable]
    audit = {
        "dataset_root": index.root,
        "split_source": "DAiSEE official DataSet/{Train,Validation,Test} + Labels/*.csv",
        "target_label": target,
        "num_train": len(index.clips.get("train", [])),
        "num_val": len(index.clips.get("val", [])),
        "num_test": len(index.clips.get("test", [])),
        "num_unique_subjects": len(all_subjects),
        "subjects_per_split": {s: sorted(index.subjects(s)) for s in SPLITS},
        "subject_overlaps": overlaps,
        "fps_distribution": _dist([p.fps for p in ok_probes if p.fps]),
        "frame_count_distribution": _dist([p.frame_count for p in ok_probes if p.frame_count]),
        "resolution_distribution": _dist([f"{p.width}x{p.height}" for p in ok_probes if p.width]),
        "duration_s_distribution": _dist([round(p.duration_s, 2) for p in ok_probes if p.duration_s]),
        "label_distribution": {s: {str(k): v for k, v in label_dists[s].items()} for s in SPLITS},
        "missing_files": [
            {"kind": p.kind, "detail": p.detail, "items": p.items}
            for p in index.problems if p.kind == "label_without_video"
        ],
        "corrupt_files": [
            {"split": r.split, "clip_id": r.clip_id, "path": r.path, "error": p.error}
            for r, p in probes if not p.decodable
        ],
        "indexing_problems": [
            {"kind": p.kind, "detail": p.detail, "num_items": len(p.items)}
            for p in index.problems
        ],
        "probe_coverage": "all" if args.probe_all else f"first {args.probe} per split",
        "num_probed": len(probes),
        "environment": environment_record(),
    }
    save_json(audit, "artifacts/dataset_audit.json")
    print("\nwrote artifacts/dataset_audit.json")

    if observed_values and observed_values <= expected:
        mapping = {
            "target_label": target,
            "source": "DAiSEE Labels/*.csv integer codes, verified against the values "
                      "actually present in this copy of the dataset",
            "verified_encoded_values": sorted(observed_values),
            "index_to_name": {str(k): v for k, v in CONCEPTUAL_ENGAGEMENT_CLASSES.items()},
            "note": "The CSV integer is used directly as the class index; no remapping is "
                    "applied. Class names come from the DAiSEE documentation "
                    "(four intensity levels: very low, low, high, very high).",
            "counts_per_split": {s: {str(k): v for k, v in label_dists[s].items()} for s in SPLITS},
        }
        save_json(mapping, "artifacts/label_mapping.json")
        print("wrote artifacts/label_mapping.json")

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 72)
    if blocking:
        print("CHECKPOINT FAILED - do not proceed")
        for b in blocking:
            print(f"  - {b}")
        print("=" * 72)
        return 1
    print("CHECKPOINT 1 & 2 PASSED")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
