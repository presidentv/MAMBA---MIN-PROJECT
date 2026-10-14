# Prompt for building the DAiSEE subset on the training machine

**Do not upload any DAiSEE video, face crop or label file to ChatGPT.** DAiSEE is
licence-restricted per-user and is video of identifiable people. This prompt is
written so that everything runs locally on the machine that already holds the
data: ChatGPT produces a script, you run it, and only counts ever leave the
machine.

If the repository is already on that machine, you do not need this prompt at
all — `scripts/build_subset.py` does the same job and is tested:

```bash
python scripts/build_subset.py --config configs/config_full_3class.yaml --cap 1200 --dry-run
python scripts/build_subset.py --config configs/config_full_3class.yaml --out /data/DAiSEE_sub --cap 1200
```

Use the prompt below when you want the subset built independently of the repo.

---

## Copy from here

I need a single Python script that builds a smaller copy of the DAiSEE dataset
by undersampling the common classes. Please write the complete script, then
explain how to verify its output. Do not ask me to upload any data — the script
runs on the machine that holds the dataset.

### The data

DAiSEE ships in this layout:

```
<root>/DataSet/Train/<SubjectID>/<ClipID>/<ClipID>.avi
<root>/DataSet/Validation/<SubjectID>/<ClipID>/<ClipID>.avi
<root>/DataSet/Test/<SubjectID>/<ClipID>/<ClipID>.avi
<root>/Labels/TrainLabels.csv
<root>/Labels/ValidationLabels.csv
<root>/Labels/TestLabels.csv
```

Each label CSV has the header `ClipID,Boredom,Engagement,Confusion,Frustration`
and these quirks that must be handled, not worked around:

- the final header has a **trailing space** (`"Frustration "`)
- the `ClipID` column includes the file extension, e.g. `1100011002.avi`
- for at least one clip the CSV and the directory tree disagree on the
  extension, so match on the **stem** when an exact filename lookup fails
- the files may carry a UTF-8 BOM, so read them with `encoding="utf-8-sig"`

The **subject ID is the directory one level below the split directory**. This
matters: the splits are subject-disjoint by design, and that must be preserved.

I only care about the `Engagement` column. Its distribution is:

| Engagement | Train | Val | Test |
|---|---|---|---|
| 0 Very Low | 34 | 23 | 4 |
| 1 Low | 213 | 143 | 84 |
| 2 High | 2,617 | 813 | 882 |
| 3 Very High | 2,494 | 450 | 814 |

Have the script **measure** these from the CSVs rather than assuming them, and
print what it found. If what it finds disagrees with the table, print a warning
and continue with the measured values.

### What the script must do

1. **Merge engagement levels 0 and 1 into one class**, giving three classes:
   `0 = Low` (original 0 and 1), `1 = High` (original 2), `2 = Very High`
   (original 3). This merge is used **only to decide how many clips to keep per
   class**. The label CSVs written out must contain the **original, unmodified
   rows**, including the original 4-level `Engagement` value and all four affect
   labels — the merge is applied later by my training config, and baking it into
   the data would be irreversible.

2. **Subsample the training split only.** Do not touch Validation or Test. They
   must be copied across complete. Subsampling an evaluation split changes the
   baseline it is measured against (balancing DAiSEE's test split moves
   majority-class accuracy from 0.494 to 0.333), throws away most of the
   statistical power, and makes the result incomparable to published work. Make
   this a command-line flag that defaults to train-only, and print a loud
   warning if anyone overrides it.

3. **Keep the rare class whole, cap the common ones.** Default behaviour: keep
   all clips of merged class 0 (247 training clips), and cap merged classes 1
   and 2 at `--cap` (default 1200). Support `--per-class N N N` for explicit
   targets. A class with fewer clips than its target is kept whole, never
   padded or duplicated.

4. **Select clips round-robin across subjects, not at random.** This is the
   important part. The training split has 5,357 clips from only 69 subjects, and
   the clips per subject are very uneven. Random undersampling drops whole
   subjects as a side effect, and subject diversity rather than clip count is
   what limits generalisation on a 69-subject corpus — 2,000 clips from 69
   students teaches more than 2,000 clips from 25. So: group the clips of a
   class by subject, shuffle within each subject with a seed, then take one clip
   from each subject in turn, repeating until the target is met or every subject
   is exhausted. Print how many subjects each class retained; for the capped
   classes it should still be close to 69.

5. **Write a real DAiSEE root**, same layout as above, so the training pipeline
   runs against it unchanged. Reproduce each kept clip at
   `<out>/DataSet/<Split>/<SubjectID>/<ClipID>/<ClipID>.avi` and write filtered
   label CSVs at `<out>/Labels/<Split>Labels.csv` containing the original header
   and only the rows for the kept clips.

6. **Link rather than copy by default.** `--link hard` (default) uses hard links
   so the subset costs no extra disk on the same filesystem; `--link symlink`
   and `--link copy` are the alternatives. If a hard link fails because the
   destination is on a different filesystem, stop with a clear message saying to
   use `--link copy` — do not silently fall back to copying 40 GB.

7. **Refuse to write into a non-empty output directory.**

8. **Be deterministic.** A `--seed` (default 42) must make the selection
   reproducible, and the same seed must always pick the same clips.

9. **Support `--dry-run`** that prints the full plan — per class: clips
   available, clips selected, subjects available, subjects retained, and the
   resulting class prior — and writes nothing.

10. **Write `<out>/subset_summary.json`** recording the source root, seed, link
    mode, the targets requested, and per-split per-class counts before and
    after, plus subject counts. **Record counts per class only — never a
    clip-ID-to-label mapping**, which would redistribute DAiSEE's per-clip
    annotations.

### Verification I want you to specify afterwards

After the script, tell me the commands to check:

- the output's per-class counts match the plan
- Validation and Test have exactly their original clip counts
- the number of label CSV rows equals the number of video files, per split
- no subject appears in more than one split
- every kept video is readable at its new path (e.g. open the first and last
  with OpenCV and report the frame count)

### Style

One file, standard library plus `argparse` only — no pandas, no external
dependencies. Type hints. Comments that explain *why* a non-obvious choice was
made, not what the line does. No `try`/`except` that swallows an error silently.

## Copy to here

---

## After the subset exists

On the machine with the data, pointing at the new root:

```bash
export DAISEE_ROOT=/path/to/DAiSEE_sub
python scripts/run_preprocess.py --config configs/config_full_3class.yaml
python scripts/fit_mrs_stats.py  --config configs/config_full_3class.yaml
python scripts/probe_finetune_memory.py --config configs/config_full_3class.yaml
python scripts/run_finetune.py --config configs/config_full_3class.yaml \
    --run-name sub3class --logit-adjust --tta
```

Two things to set before the long run:

- `training.batch_size` from what the memory probe reports, with
  `grad_accum_steps = 32 / batch_size` so the effective batch stays 32.
- `evaluation.reference_prior: [0.049, 0.494, 0.456]` — **only when training on
  a subset.** The subset's own class prior is not the corpus prior, so the
  prior correction needs to be told what distribution to calibrate to. That
  published corpus-level distribution is a documented property of DAiSEE; a
  prior read off the test split's own labels would be using the test set and
  must never go there.
