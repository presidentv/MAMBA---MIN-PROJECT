# Explainable Multimodal Engagement Detection — DAiSEE pipeline

A reproducible pipeline for student-engagement detection on the
[DAiSEE](https://people.iith.ac.in/vineethnb/resources/daisee/) corpus: video
preprocessing with a **Motion Reliability Score**, MediaPipe facial landmark and
iris extraction, gaze + affect feature engineering, and a cross-attention
transformer fusion model with an ordinal head — evaluated against five required
baselines.

Built and verified end-to-end on a 108-clip DAiSEE sample. **The code is written
to scale to the full 9,068-clip dataset unchanged** — nothing hardcodes a subject
ID, clip count or split size; everything is discovered by walking the directory
tree.

> **Status: the training results in this repo are a smoke test, not a result.**
> With 36 training clips across 4 classes, the numbers below demonstrate that the
> pipeline runs correctly on real data. They do not demonstrate that the model
> works. See [Results](#results).

---

## Data is not in this repository

Nothing derived from DAiSEE's video is committed here, deliberately:

| Excluded | Why |
|---|---|
| `Datasets/` | DAiSEE is licence-restricted (per-user registration) and is video of identifiable people. Redistribution is not permitted. |
| `outputs/phase1_reliable_frames/` | 5,402 aligned face crops of research subjects — a distilled face dataset. |
| `outputs/phase2_landmarks/`, `outputs/features/` | Derived biometric data: 478-point landmark tables and 1280-d facial-expression embeddings. |
| `outputs/*_manifest.csv` | Carry the DAiSEE per-clip label values. The run statistics they summarise are reproduced in [`outputs/README.md`](outputs/README.md). |
| `models/` | MediaPipe model bundles — fetched on demand, see step 2 below. |

What **is** committed: all pipeline and model code, configs, the run logs, the
evaluation results (`outputs/results/`), and the documentation.

Obtain DAiSEE yourself from the authors, place it anywhere, and point
`--input-root` at it.

---

## Pipeline

```
DAiSEE video
    │
    ├─ Phase 1  pipeline/phase1_preprocessing.py
    │     sample @5fps → detect + track subject's face → align to canonical
    │     eye line → normalise 224×224 → Motion Reliability Score → gate
    │     ↳ aligned crops + per-frame MRS component table
    │
    ├─ Phase 2  pipeline/phase2_landmarks.py
    │     478-point iris-refined face mesh → head pose → frame-to-frame tracking
    │     ↳ landmarks.parquet (1488 cols), labels merged by ClipID
    │
    ├─ Step 1   src/features.py     21 per-frame gaze features + 104 clip-level
    ├─ Step 2   src/affect.py       frozen FER: 8 emotion probs + 1280-d embedding
    ├─ Step 3   src/datasets.py     padded masked sequences, split-integrity checks
    ├─ Step 4   src/models.py       2× temporal transformer + cross-attention + CORAL
    ├─ Step 5   src/baselines.py    majority, logreg, gaze-only, affect-only, late fusion
    └─ Step 6   src/train.py        AdamW + cosine, early stopping on val macro-F1
```

### The Motion Reliability Score

Five sub-scores in `[0,1]`, combined as a configurable weighted mean; frames
below `mrs_threshold` are discarded and clips losing >70% of frames are flagged
rather than silently dropped.

| component | measured on | note |
|---|---|---|
| `blur` | **native-resolution** face ROI | Variance of Laplacian. Deliberately not the aligned crop: DAiSEE faces are ~130px and get upscaled to 224, so scoring the crop reports every frame as soft. |
| `face_visibility` | detection | BlazeFace confidence; 0 when no face found. |
| `head_rotation` | pose | `1 − ‖(yaw,pitch,roll)‖ / 60°` |
| `eye_visibility` | keypoints + mesh | Both eyes in-frame and in-box, modulated by true eye-aspect-ratio. |
| `motion_consistency` | consecutive **aligned** crops | Median Farnebäck flow. Alignment already cancels real head motion, so residual flow indicates *detector jitter* — which is what should be penalised. |

Saturation constants were **calibrated on 240 frames across 12 clips spanning all
three splits**, not guessed: Laplacian variance p50 = 150 sets `blur_var_reference`;
observed flow p95 = 3.6px sets `max_flow_magnitude_px = 8.0` at ≈2.2× p95.

---

## Quickstart

Requires Python 3.13. All commands run from the repository root.

**1. Environment**

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

**2. Fetch the MediaPipe model bundles (once, ~4 MB)**

```bash
.venv/Scripts/python.exe pipeline/fetch_models.py
```

**3. Run the pipeline**

```bash
.venv/Scripts/python.exe pipeline/phase1_preprocessing.py --input-root /path/to/DAiSEE --output-root outputs --config pipeline/configs/default.json
```

```bash
.venv/Scripts/python.exe pipeline/phase2_landmarks.py --input-root /path/to/DAiSEE --output-root outputs --config pipeline/configs/default.json
```

```bash
.venv/Scripts/python.exe src/features.py --output-root outputs --config src/configs/default.json
```

```bash
.venv/Scripts/python.exe src/affect.py --output-root outputs --config src/configs/default.json
```

```bash
.venv/Scripts/python.exe src/train.py --output-root outputs --config src/configs/default.json
```

Smoke-test a subset first with `--limit-clips 20` on both phases.

Full details, tunable flags and full-dataset scaling notes (storage, runtime,
label-imbalance expectations): **[`outputs/README.md`](outputs/README.md)**.

---

## Results

Test split, 36 clips. `majority` is the trivial floor.

| model | macro-F1 | accuracy | QWK |
|---|---|---|---|
| logreg_meanpool | **0.349** | 0.306 | 0.201 |
| gaze_only | 0.320 | 0.417 | 0.152 |
| cross_attention_fusion | 0.181 | 0.167 | 0.135 |
| majority | 0.173 | **0.528** | 0.000 |
| late_fusion | 0.126 | 0.167 | 0.100 |
| affect_only | 0.087 | 0.111 | 0.011 |

**Read the two bolded cells together.** The majority-class predictor has the
*highest accuracy* of all six models and the *second-lowest macro-F1*. This is
why macro-F1 and the full confusion matrix are the headline everywhere in this
repo and bare accuracy is never reported alone.

What this run legitimately establishes:

1. The pipeline is correct end-to-end — shapes line up, no silent NaNs, split
   integrity asserted rather than assumed.
2. The transformers overfit hard, as expected at n=36: train macro-F1 reaches
   0.41–0.64 while validation sits at 0.15–0.21 and validation loss climbs from
   about epoch 3.
3. Logistic regression beating both transformers is the control working — it
   confirms that frozen encoders plus a light head, with logreg reported beside
   every result, is the right mitigation at this sample size.

`src/train.py` prints the per-split class distribution and an explicit warning
for every class holding fewer than five clips *before* it trains anything.

Full interpretation and known feature weaknesses: **[`training_log.md`](training_log.md)**.

---

## Design notes worth knowing

**MediaPipe Tasks, not `solutions`.** MediaPipe ≥ 1.0 removed the legacy
`mediapipe.solutions` API from every Python 3.13 wheel. This pipeline uses the
Tasks API, which is a functional superset and additionally provides a facial
transformation matrix — better head pose than solvePnP on six keypoints.

**Phase 2 reads Phase 1's crops, not the videos.** Video decoding is never paid
for twice, and Phase 2 depends only on the Phase 1 output tree. The 2×3
alignment affine is stored on every row, so any landmark can be projected back
to original-frame pixels via `alignment.invert_affine`.

**CORAL ordinal loss.** The head emits K−1 cumulative logits sharing one weight
vector, differing only in bias, which makes `P(y>0) ≥ P(y>1) ≥ P(y>2)` monotone
*by construction*. A cumulative-link model has the same intent but must learn
its thresholds freely, and at n=36 those thresholds routinely invert. Plain
cross-entropy is available via `--loss ce` as the honest ablation.

**Subject-disjoint splits are verified, not assumed.**
`datasets.verify_split_integrity` **raises** on violation — leakage would
invalidate every number, so it must not be recoverable by accident.

### Substitutions from the original plan

| Planned | Used | Why |
|---|---|---|
| OpenFace 2.0 action units | HSEmotion (frozen EfficientNet-B0) | OpenFace is not a pip package. Consequence: the affect channel is *categorical expression*, not *AU intensity* — AU-level interpretability claims no longer apply. |
| AU45 blink | Eye-aspect-ratio blink detection | Follows from the above. Uses a per-clip adaptive threshold, since EAR baseline varies with face shape and eyewear. |

Two environment pins this depends on: `timm` must be **< 1.0** (the HSEmotion
checkpoints were pickled against timm 0.x and fail at *inference* under ≥1.0),
and PyTorch ≥2.6 needs a scoped `weights_only=False` to load them.

### Known feature limitations

- `fixation_ratio` is ≈1.0 for nearly every clip. At 5 fps the sampling interval
  is 200 ms while a saccade lasts 30–80 ms, so the I-VT threshold rarely fires.
  **This feature measures the rate of gaze shifts between samples, not a saccade
  rate**, and must not be described as the latter.
- `off_screen_ratio` is 0.0 across the sample — subjects genuinely look at the
  screen throughout. It may gain variance on the full dataset.

---

## Repository layout

```
pipeline/
  fetch_models.py              download MediaPipe Tasks bundles
  phase1_preprocessing.py      Phase 1 driver
  phase2_landmarks.py          Phase 2 driver
  configs/default.json
  engagement_pipeline/         alignment, mrs, pose, faces, video, dataset, io
src/
  features.py  affect.py  datasets.py  models.py  baselines.py  metrics.py  train.py
  early_detection.py           STUB — label-shift early detection
  explain.py                   STUB — SHAP, attention rollout, deletion test
  configs/default.json
outputs/
  README.md                    run commands + full-dataset scaling notes
  results/                     results.json, results.csv, train.log
  logs/                        phase logs and resolved configs
training_log.md                what was built, substitutions, interpretation
requirements.txt
```

## Not yet implemented

Both are stubs with fixed signatures, so implementing them is a fill-in rather
than a redesign:

- **`src/early_detection.py`** — predicting clip *t+1*'s engagement from clip *t*.
  The clip ordering problem is solved and documented in the module: `ClipID =
  <6-digit SubjectID><suffix>`; left-pad the suffix to 4 digits, first digit is
  the session, last three a strictly-increasing within-session index. *Inferred
  from the ID pattern, not documented by the DAiSEE authors*, and indices are not
  dense, so only gap-of-1 pairs are safely adjacent.
- **`src/explain.py`** — SHAP over modality-level feature blocks, attention
  rollout over the cross-attention weights (already returned by the model), and a
  deletion test against a random-ablation control, which is what keeps the other
  two honest.
