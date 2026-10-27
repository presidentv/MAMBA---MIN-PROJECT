"""Check that src/inference.py reproduces the evaluated model.

src/inference.py exists so a front end can score one arbitrary video without a
dataset index or a cache. That freedom is also the risk: it assembles the model
input itself, so a drift in the MRS computation, the landmark standardiser, the
backbone normalisation or the TTA set would change the prediction while still
producing a plausible-looking number. Nothing would raise.

This compares, clip by clip, against artifacts/per_clip_predictions_<run>_test.json
-- the file the reported metrics were computed from.

Two modes, depending on what is attached:

  --from-cache (default)  reads the Stage 1 arrays for test clips and runs
                          everything downstream of them. Needs only the cache
                          directory, which lives outside the repository.

  --from-video            decodes the DAiSEE clip itself, so face detection and
                          the landmark mesh are checked too. Needs the dataset
                          drive. This is the complete check.

Run it after any change to the preprocessing, the MRS, the fusion scaler or the
evaluation protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import EngagementPredictor  # noqa: E402
from src.utils import resolve_path  # noqa: E402

TOLERANCE = 1e-4


def load_expected(run: str) -> list[dict]:
    path = resolve_path(Path("artifacts") / f"per_clip_predictions_{run}_test.json")
    if not Path(path).is_file():
        raise SystemExit(
            f"{path} not found. It is written by scripts/run_finetune.py and is what "
            f"this check compares against.")
    return json.loads(Path(path).read_text(encoding="utf-8"))["predictions"]


def choose(rows: list[dict], n: int) -> list[dict]:
    """A spread across predicted classes, not just the head of the file.

    Checking only confidently-engaged clips would miss a drift that only shows
    up near the decision boundary.
    """
    by_class: dict[int, list[dict]] = {}
    for r in rows:
        by_class.setdefault(int(r["pred"]), []).append(r)
    # Within each class take the most and least confident, then fill in order.
    out: list[dict] = []
    per = max(1, n // max(1, len(by_class)))
    for _, group in sorted(by_class.items()):
        group = sorted(group, key=lambda r: max(r["probs"]))
        picks = {id(group[0]): group[0], id(group[-1]): group[-1]}
        for r in group:
            if len(picks) >= per:
                break
            picks.setdefault(id(r), r)
        out.extend(picks.values())
    return out[:n]


def probabilities_from_cache(predictor, npz_path: Path) -> np.ndarray:
    data = dict(np.load(npz_path, allow_pickle=True))
    probs = predictor._forward(predictor._build_item(data))
    extra = [v for v in predictor.tta_views if float(v.scale) < 1.0]
    if not extra:
        return probs
    views = [probs] + [predictor._forward(predictor._build_item(data, v)) for v in extra]
    return np.mean(np.stack(views), axis=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--run", default="fullbin",
                    help="names the per-clip predictions file to compare against")
    ap.add_argument("--cache-dir", default=None,
                    help="Stage 1 test directory, e.g. <cache>/faces_T16_<key>/test")
    ap.add_argument("--video-dir", default=None,
                    help="with --from-video: directory holding the test clips")
    ap.add_argument("--from-video", action="store_true",
                    help="decode the videos too, checking the whole path")
    ap.add_argument("-n", type=int, default=8, help="how many clips to check")
    args = ap.parse_args(argv)

    rows = load_expected(args.run)
    chosen = choose(rows, args.n)
    predictor = EngagementPredictor.from_checkpoint(args.checkpoint)
    print(f"checkpoint : {args.checkpoint}")
    print(f"classes    : {', '.join(predictor.class_names)}")
    print(f"tta        : {[v.describe()['scale'] for v in predictor.tta_views] or 'off'}")
    print(f"mode       : {'video' if args.from_video else 'cache'}\n")

    worst, failures, skipped = 0.0, 0, 0
    try:
        for row in chosen:
            clip_id = row["clip_id"]
            stem = Path(clip_id).stem
            try:
                if args.from_video:
                    if not args.video_dir:
                        raise SystemExit("--from-video needs --video-dir")
                    matches = list(Path(args.video_dir).rglob(clip_id))
                    if not matches:
                        print(f"  {stem}: not found under {args.video_dir}, skipped")
                        skipped += 1
                        continue
                    probs = np.asarray(
                        list(predictor.predict(matches[0]).probabilities.values()))
                else:
                    if not args.cache_dir:
                        raise SystemExit("--cache-dir is required in cache mode")
                    npz = Path(args.cache_dir) / f"{stem}.npz"
                    if not npz.is_file():
                        print(f"  {stem}: no cached npz, skipped")
                        skipped += 1
                        continue
                    probs = probabilities_from_cache(predictor, npz)
            except Exception as exc:
                print(f"  {stem}: FAILED to score ({exc!r})")
                failures += 1
                continue

            expected = np.asarray(row["probs"], dtype=np.float64)
            gap = float(np.max(np.abs(expected - probs)))
            worst = max(worst, gap)
            agree = int(np.argmax(probs)) == int(row["pred"])
            ok = gap < TOLERANCE and agree
            failures += (not ok)
            print(f"  {stem}: max|dp|={gap:.2e}  {'MATCH' if ok else 'MISMATCH'}")
            if not ok:
                print(f"      evaluated pred={row['pred']} "
                      f"probs={[round(float(v), 6) for v in expected]}")
                print(f"      this path pred={int(np.argmax(probs))} "
                      f"probs={[round(float(v), 6) for v in probs]}")
    finally:
        predictor.close()

    checked = len(chosen) - skipped
    print(f"\n{checked} clip(s) checked, {skipped} skipped, worst deviation {worst:.3e}")
    if failures:
        print(f"{failures} did not reproduce the evaluated probabilities "
              f"(tolerance {TOLERANCE:g}).")
        return 1
    if checked == 0:
        print("nothing was checked; the paths given hold none of the test clips.")
        return 1
    print("src/inference.py reproduces the evaluated model.")
    if not args.from_video:
        print("note: this was the cache path. Face detection and the landmark mesh "
              "were not exercised; rerun with --from-video when the dataset drive "
              "is attached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
