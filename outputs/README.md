# DAiSEE engagement pipeline — outputs

Everything in this directory is generated. Nothing here is edited by hand, and
nothing under `Datasets/DAiSEE_Small/` is ever modified by the pipeline.

Generated on the **DAiSEE_Small sample**: 108 clips / 16 subjects / 3 splits.
All commands below are run from the `MINI PROJECT` directory.

---

## 0. Environment

Python 3.13 in a project venv. MediaPipe ≥ 1.0 **removed the legacy
`mediapipe.solutions` API** (`solutions.face_mesh`, `solutions.face_detection`)
from every wheel that supports Python 3.13, so this pipeline is written against
the **MediaPipe Tasks API** instead. Tasks is a functional superset — same
BlazeFace detector, same 478-point iris-refined mesh, plus a facial
transformation matrix that yields head pose directly — but it does not bundle
model weights inside the wheel, so they are fetched once (see step 1).

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

`torch` is the CPU build; for CUDA replace the torch/torchvision lines with the
appropriate index URL. **`timm` must stay below 1.0** — the HSEmotion
checkpoints were pickled against timm 0.x and fail at *inference* time (not load
time) under timm ≥ 1.0.

`ffmpeg` is **not required**: OpenCV's bundled decoder reads the DAiSEE MPEG-4
AVIs directly. The pipeline checks for `ffmpeg` on PATH and enables it as a
decode fallback only if present, logging a warning otherwise.

---

## 1. Fetch the MediaPipe model bundles (once)

```bash
.venv/Scripts/python.exe pipeline/fetch_models.py
```

Downloads to `models/` from Google's official MediaPipe model host:

| file | size | used by |
|---|---|---|
| `blaze_face_short_range.tflite` | 0.23 MB | Phase 1 face detection |
| `face_landmarker.task` | 3.76 MB | Phase 1 pose, Phase 2 landmarks |

---

## 2. Phase 1 — preprocessing + Motion Reliability Score

```bash
.venv/Scripts/python.exe pipeline/phase1_preprocessing.py --input-root Datasets/DAiSEE_Small --output-root outputs --config pipeline/configs/default.json
```

Per clip: sample at 5 fps → detect + track the subject's face → align to a
canonical eye line → normalise to 224×224 → score five MRS components → keep
frames with MRS ≥ threshold.

**Outputs**

```
phase1_reliable_frames/<split>/<SubjectID>/<ClipID>/
    frames/s0000_f000000.png ...   retained aligned crops
    mrs_scores.parquet             one row per SAMPLED frame (kept and discarded)
phase1_manifest.csv                one row per clip
logs/phase1.log, phase1_warnings.csv, phase1_config.json
```

`mrs_scores.parquet` carries the five sub-scores, the final `mrs`, `retained`,
the detection box, head pose, and the 2×3 alignment affine (`affine_a00` …
`affine_a12`) so any downstream point can be projected back to original-frame
pixels.

### MRS components

| component | measured on | notes |
|---|---|---|
| `blur` | **native-resolution** face ROI | variance of Laplacian. Measured at native resolution deliberately: DAiSEE faces are ~130 px and get upscaled to 224, so scoring the aligned crop would report every frame as soft. |
| `face_visibility` | detection | BlazeFace confidence; 0 when no face found |
| `head_rotation` | pose | `1 − ‖(yaw,pitch,roll)‖ / 60°` |
| `eye_visibility` | keypoints + mesh | both eyes in-frame and in-box, modulated by true eye-aspect-ratio |
| `motion_consistency` | consecutive **aligned** crops | median Farnebäck flow. Alignment already cancels genuine head motion, so residual flow indicates *detector jitter* — which is what should be penalised |

### MRS calibration

The two saturating constants were fitted to 240 sampled frames spread over 12
clips from all three splits, not guessed:

* Laplacian variance on the native ROI: p10 = 75, p25 = 110, **p50 = 150**,
  p90 = 424 → `blur_var_reference = 150` anchors saturation at the median, so
  the term penalises the blurred tail without rewarding extra sharpness.
* Median inter-frame flow on good frames: p50 = 1.5, p90 = 3.0, **p95 = 3.6 px**
  → `max_flow_magnitude_px = 8.0` sits at ≈2.2× p95, so ordinary micro-motion
  is barely penalised while genuine jitter is.

### Useful flags

```bash
.venv/Scripts/python.exe pipeline/phase1_preprocessing.py --input-root <root> --output-root outputs --sample-fps 10 --mrs-threshold 0.6 --weight blur=2.0 --weight head_rotation=1.5
```

`--no-mesh-pose` skips the face mesh in Phase 1 (~6× faster, ~1.7 ms/frame
instead of ~10.2 ms) at the cost of pose accuracy — head pose then comes from
solvePnP on only the six BlazeFace keypoints, which carries a noticeable yaw
bias because DAiSEE faces are small and often off-centre.

---

## 3. Phase 2 — landmarks, iris, head pose

```bash
.venv/Scripts/python.exe pipeline/phase2_landmarks.py --input-root Datasets/DAiSEE_Small --output-root outputs --config pipeline/configs/default.json
```

Reads the **aligned crops from Phase 1**, not the videos — so Phase 2 needs only
the Phase 1 output tree and video decoding is never paid for twice.

**Output:** `phase2_landmarks/<split>/<SubjectID>/<ClipID>/landmarks.parquet`,
one row per retained frame, **1488 columns**:

* `lm_000_x/y/z` … `lm_477_z` — 1434 cols, all 478 mesh points (468 face + 10
  iris), normalised to the **aligned crop**
* iris + eye derived: `left_iris_x/y/z`, `right_iris_*`, `left_ear`, `right_ear`,
  `mean_ear`, `left_iris_rel_x/y`, `right_iris_rel_x/y`, `*_eye_width`
* head pose: `yaw`, `pitch`, `roll_aligned` (in crop space), `roll` (referred
  back to the original frame by adding the alignment rotation), plus an
  independent `pnp_*` solvePnP cross-check
* tracking: `track_id`, `landmark_displacement`, `track_break`
* carried from Phase 1: `mrs` and its 5 sub-scores, the affine, alignment scale
* labels merged by ClipID: `Boredom`, `Engagement`, `Confusion`, `Frustration`

Because landmarks are in aligned-crop space, use
`alignment.invert_affine(...)` with the stored affine columns to recover
original-frame pixel coordinates.

---

## 4. Steps 1–2 — feature extraction

```bash
.venv/Scripts/python.exe src/features.py --output-root outputs --config src/configs/default.json
```

```bash
.venv/Scripts/python.exe src/affect.py --output-root outputs --config src/configs/default.json
```

* `features/gaze/<split>/<subj>/<clip>/gaze.parquet` — 21 per-frame gaze
  features; `features/gaze_clip_features.csv` — 104 clip-level aggregates
* `features/affect/<split>/<subj>/<clip>/affect.parquet` — 8 emotion
  probabilities + 1280-d embedding per frame

---

## 5. Steps 3–6 — training and evaluation

```bash
.venv/Scripts/python.exe src/train.py --output-root outputs --config src/configs/default.json
```

Runs all six models and writes `results/results.json`, `results/results.csv`,
`results/train.log`, `results/feature_scaler.npz`, and
`checkpoints/<model>.pt`.

Useful overrides: `--loss ce` (ablate CORAL), `--affect-feature-set embedding`,
`--epochs`, `--learning-rate`, `--models gaze_only cross_attention_fusion`.

---

## Scaling to the full DAiSEE dataset

The pipeline hardcodes **no** subject IDs, clip counts or split sizes —
everything is discovered by walking the tree — so the only change is the input
path. Given the official DAiSEE laid out as
`<DAiSEE>/DataSet/<Split>/<Subject>/<Clip>/<Clip>.avi` plus `<DAiSEE>/Labels/`:

```bash
.venv/Scripts/python.exe pipeline/fetch_models.py
```

```bash
.venv/Scripts/python.exe pipeline/phase1_preprocessing.py --input-root /path/to/DAiSEE --output-root outputs_full --config pipeline/configs/default.json --image-format jpg
```

```bash
.venv/Scripts/python.exe pipeline/phase2_landmarks.py --input-root /path/to/DAiSEE --output-root outputs_full --config pipeline/configs/default.json
```

```bash
.venv/Scripts/python.exe src/features.py --output-root outputs_full --config src/configs/default.json
```

```bash
.venv/Scripts/python.exe src/affect.py --output-root outputs_full --config src/configs/default.json
```

```bash
.venv/Scripts/python.exe src/train.py --output-root outputs_full --config src/configs/default.json
```

### What changes at full scale (9,068 clips ≈ 453k sampled frames)

* **Storage.** The sample used 533 MB for 108 clips with lossless PNG. Full
  DAiSEE at PNG would be roughly **45 GB**; `--image-format jpg` cuts that to
  ≈7 GB. Use JPEG unless you specifically need lossless crops.
* **Runtime.** Measured on this machine: Phase 1 ≈ 10.2 ms/frame with mesh pose
  (≈1.3 h) or ≈1.7 ms/frame without (≈0.2 h); Phase 2 ≈ 8.6 ms/frame (≈1.1 h).
  Add video decoding, ≈1.5 s/clip. Budget **3–5 h single-threaded**.
* **Label distribution.** This sample is unusually balanced (Engagement 0/1/2/3
  ≈ 4/6/16/10 per split). Full DAiSEE is severely imbalanced — class 0 is
  ~1% of clips — so `class_weighting` matters far more, and macro-F1 will drop
  sharply relative to what the sample shows.
* **Smoke test first.** `--limit-clips 20` on both phases before committing to
  a full run.

---

## Run summary — the sample

**Phase 1:** 108/108 clips OK · 5,403 frames sampled · **5,402 retained
(100.0%)** · mean MRS 0.855 · 0 clips flagged.
Retention is this high because the sample clips are uniformly good quality; the
`>70% discarded` warning path exists and is exercised by the config, it simply
never triggered here.

**Phase 2:** 108/108 clips OK · **5,384 frames landmarked** · 18 mesh failures
concentrated in 3 Test clips (`8264120120`, `8264120210`, `8264120240`) · 3
track breaks · 0 clips missing a label.

**Split integrity:** 16 subjects, **none shared across splits** (asserted, not
assumed).

See `../training_log.md` for the model results and their interpretation.
