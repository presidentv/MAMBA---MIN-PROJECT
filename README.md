# MRG-VM — Motion Reliability Guided Vision Mamba for Student Engagement Detection

Full implementation of the seven-phase framework described in the project
specification: video preprocessing with a novel **Motion Reliability Score**,
MediaPipe landmark extraction, a **reliability-guided Vision Mamba** backbone,
adaptive feature fusion, a lightweight MLP classifier, an ablation study, and
SHAP explainability.

Built and verified end-to-end on a 108-clip [DAiSEE](https://people.iith.ac.in/vineethnb/resources/daisee/)
sample. **The code scales to the full 9,068-clip dataset unchanged** — nothing
hardcodes a subject ID, clip count or split size; everything is discovered by
walking the directory tree.

> **The reported metrics are a smoke test, not a result.** With 36 training clips
> across 4 ordinal classes, they demonstrate that the pipeline runs correctly on
> real data. They do not demonstrate that the model works. See [Results](#results).

**Two places to start, depending on what you need:**

- [`docs/architecture.html`](docs/architecture.html) — the architecture diagram, with
  tensor shapes at every boundary and the Mamba block internals. For the report and
  the review slides.
- [`docs/how_it_works.html`](docs/how_it_works.html) — the whole pipeline in plain
  language, with diagrams. Open it in a browser. Best if you want to *understand*
  or explain the system out loud.
- [`MODEL_AND_LIMITATIONS.md`](MODEL_AND_LIMITATIONS.md) — the same ground covered
  technically, plus a full account of the shortcomings. Best for writing this up.

---

## The seven phases

| Phase | What it does | Code | Status |
|---|---|---|---|
| **1** | Video loading, frame extraction, face detection, alignment, normalisation, **Motion Reliability Score**, low-quality frame rejection | [`pipeline/phase1_preprocessing.py`](pipeline/phase1_preprocessing.py) | run: 108/108 clips, 5,402 frames retained |
| **2** | MediaPipe Face Mesh — face, eye and iris landmarks, head pose, landmark tracking | [`pipeline/phase2_landmarks.py`](pipeline/phase2_landmarks.py) | run: 5,384 frames landmarked |
| **3** | **MRG-VM** — Vision Mamba over reliable face regions, spatial-temporal behavioural representations | [`mrgvm/mamba.py`](mrgvm/mamba.py), [`mrgvm/vision_mamba.py`](mrgvm/vision_mamba.py) | run |
| **4** | Adaptive feature fusion of Mamba embeddings with landmark geometry, normalisation, final feature vector | [`mrgvm/fusion.py`](mrgvm/fusion.py), [`mrgvm/geometric.py`](mrgvm/geometric.py) | run |
| **5** | Lightweight MLP engagement classifier, trained and validated | [`mrgvm/train_mrgvm.py`](mrgvm/train_mrgvm.py) | run |
| **6** | Evaluations + ablation of MRS / Vision Mamba / landmarks / adaptive fusion | [`mrgvm/ablation.py`](mrgvm/ablation.py) | run: 7 variants |
| **7** | SHAP feature importance and behavioural-contribution visualisation | [`mrgvm/shap_explain.py`](mrgvm/shap_explain.py) | run |

`src/` additionally holds a **transformer baseline stack** (gaze + affect
branches, cross-attention fusion, CORAL ordinal head, five required baselines).
It is not part of the PDF's seven phases; it exists to give MRG-VM something
honest to be compared against. See [`training_log.md`](training_log.md).

---

## Data is not in this repository

Nothing derived from DAiSEE's video is committed, deliberately:

| Excluded | Why |
|---|---|
| `Datasets/` | DAiSEE is licence-restricted (per-user registration) and is video of identifiable people. Redistribution is not permitted. |
| `outputs/phase1_reliable_frames/` | 5,402 aligned face crops — a distilled face dataset of research subjects. |
| `outputs/phase2_landmarks/`, `outputs/features/`, `outputs/embeddings/` | Derived biometric data: landmark tables, expression embeddings, learned face representations. |
| `outputs/*_manifest.csv` | Carry the DAiSEE per-clip label values. Their statistics are reproduced in [`outputs/README.md`](outputs/README.md). |
| `models/`, `.venv/` | Fetched or built on demand. |

Committed: all code, configs, run logs, evaluation results and documentation.
Obtain DAiSEE from its authors and point `--input-root` at it.

---

## Quickstart

Python 3.13. All commands from the repository root.

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

```bash
.venv/Scripts/python.exe pipeline/fetch_models.py
```

**Phases 1–2**

```bash
.venv/Scripts/python.exe pipeline/phase1_preprocessing.py --input-root /path/to/DAiSEE --output-root outputs --config pipeline/configs/default.json
```

```bash
.venv/Scripts/python.exe pipeline/phase2_landmarks.py --input-root /path/to/DAiSEE --output-root outputs --config pipeline/configs/default.json
```

**Gaze/affect features** (feed the geometric stream and the baseline stack)

```bash
.venv/Scripts/python.exe src/features.py --output-root outputs --config src/configs/default.json
```

```bash
.venv/Scripts/python.exe src/affect.py --output-root outputs --config src/configs/default.json
```

**Phases 3–5 — train MRG-VM**

```bash
.venv/Scripts/python.exe -m mrgvm.train_mrgvm --output-root outputs --config mrgvm/configs/default.json
```

**Phases 3–4 deliverable — export embeddings**

```bash
.venv/Scripts/python.exe -m mrgvm.extract_embeddings --output-root outputs
```

**Phase 6 — ablation study**

```bash
.venv/Scripts/python.exe -m mrgvm.ablation --output-root outputs --config mrgvm/configs/default.json
```

**Phase 7 — SHAP**

```bash
.venv/Scripts/python.exe -m mrgvm.shap_explain --output-root outputs
```

Smoke-test with `--limit-clips 20` (phases 1–2) or `--epochs 2` (phases 5–6)
before committing to a full run. Full flag reference and full-dataset scaling
notes (storage, runtime, imbalance): [`outputs/README.md`](outputs/README.md).

---

## Phase 1 — the Motion Reliability Score

Five sub-scores in `[0,1]`, combined as a configurable weighted mean. Frames
below threshold are discarded; clips losing >70% of frames are flagged rather
than silently dropped.

| component | measured on | note |
|---|---|---|
| `blur` | **native-resolution** face ROI | Variance of Laplacian. Deliberately not the aligned crop: DAiSEE faces are ~130 px and get upscaled to 224, so scoring the crop reports every frame as soft. |
| `face_visibility` | detection | BlazeFace confidence; 0 when no face found. |
| `head_rotation` | pose | `1 − ‖(yaw,pitch,roll)‖ / 60°` |
| `eye_visibility` | keypoints + mesh | Both eyes in-frame and in-box, modulated by true eye-aspect-ratio. |
| `motion_consistency` | consecutive **aligned** crops | Median Farnebäck flow. Alignment already cancels real head motion, so residual flow indicates *detector jitter*. |

Saturation constants were **calibrated on 240 frames across 12 clips spanning all
three splits**, not guessed: Laplacian variance p50 = 150 sets
`blur_var_reference`; observed flow p95 = 3.6 px sets `max_flow_magnitude_px = 8.0`
at ≈2.2× p95.

## Phase 3 — where "reliability guided" actually lives

Phase 1 discards frames below threshold, but survivors are not equally
trustworthy — MRS 0.52 and 0.95 both pass. Two mechanisms carry that residual
reliability into the model:

1. **`dt` modulation.** In a state-space model `dt` controls how far the hidden
   state moves per step: `h_t = exp(dt·A)h_{t-1} + (dt·B)x_t`. Scaling `dt` by
   MRS makes a low-reliability frame *literally update the state less*, without
   masking it or breaking the sequence. This is the natural place to inject a
   per-timestep confidence into an SSM; a transformer has no clean equivalent.
2. **Reliability-weighted pooling.** The temporal pool is MRS-weighted.

Both are independently ablatable — and Phase 6 found that **only pooling
matters**: dt guidance flips zero predictions, because retained-frame MRS is
0.93 ± 0.049 and a near-constant dt rescale is absorbed by `dt_proj`'s learned
bias. `delta_map='clip_normalised'` is the implemented fix.

Note what Phase 6 *cannot* test: the frame **gate** itself. By then Phase 1 has
already discarded sub-threshold frames, so undoing it needs a Phase 1 re-run at
`--mrs-threshold 0` — and on this sample that is vacuous anyway, since the gate
rejected 1 frame out of 5,403.

The Mamba is hand-written in PyTorch because `mamba-ssm` requires CUDA/`nvcc`
and this is a CPU-only machine — same S6 algorithm, no fused kernel. Blocks are
bidirectional (Vim), since a patch grid has no causal direction. Details:
[`mrgvm/README.md`](mrgvm/README.md).

---

## Results

Test split, 36 clips. **Read macro-F1, not accuracy** — the majority-class
predictor scores the *highest accuracy* of any model here and the second-lowest
macro-F1, which is exactly the trap this dataset sets.

| model | macro-F1 | accuracy | QWK |
|---|---|---|---|
| logreg on mean-pooled features | **0.349** | 0.306 | 0.201 |
| gaze-only transformer | 0.320 | 0.417 | 0.152 |
| **MRG-VM (full)** | 0.286 | 0.472 | 0.170 |
| cross-attention transformer fusion | 0.181 | 0.167 | 0.135 |
| majority class | 0.173 | **0.528** | 0.000 |
| naive late fusion | 0.126 | 0.167 | 0.100 |
| affect-only transformer | 0.087 | 0.111 | 0.011 |

MRG-VM beats both transformer variants and the majority floor, and loses to
logistic regression on mean-pooled features. At n=36 that ordering is what you
would expect: a 1.17 M-parameter backbone has nothing to constrain it, while a
strongly regularised linear model on 29 hand-engineered features does not
overfit. Train macro-F1 reaches 0.570 against 0.392 validation.

### Phase 6 ablation (test split)

| variant | macro-F1 | Δ vs full | val macro-F1 | params |
|---|---|---|---|---|
| `no_vision_mamba` (geometry only) | **0.407** | **+0.121** | 0.236 | 42k |
| `concat_fusion` | 0.287 | +0.001 | 0.342 | 1.12M |
| `full` | 0.286 | — | 0.392 | 1.17M |
| `guide_pooling_only` | 0.286 | +0.000 | 0.392 | 1.17M |
| `no_mrs_guidance` | 0.261 | −0.025 | 0.389 | 1.17M |
| `guide_delta_only` | 0.261 | −0.025 | 0.389 | 1.17M |
| `no_landmarks` (Mamba only) | 0.052 | −0.234 | 0.194 | 1.09M |

**Removing the Vision Mamba branch improves the model**; removing the 32
hand-engineered landmark features collapses it. SHAP agrees independently
(appearance = 4.5% of attribution) as does the fusion gate (0.49, barely
committing). But `no_vision_mamba` has the *worst* validation macro-F1 and the
*best* test — an inversion that is direct evidence these gaps are noise-dominated
at n=36. The defensible claim is "the deep branch has no data to learn from yet".

**Only one of the two reliability-guidance mechanisms does anything.** `full` and
`guide_pooling_only` score identically, as do `no_mrs_guidance` and
`guide_delta_only` — toggling dt guidance flips zero predictions. Not a wiring
fault (verified with matched weights and a non-zero output delta): retained-frame
MRS is 0.93 ± 0.049, so the linear dt map is near-constant and `dt_proj`'s learned
bias absorbs it. `delta_map='clip_normalised'` is implemented as the fix and
measures 8.3× stronger, left non-default so this table stays reproducible.

Phase 7 SHAP: `outputs/results_mrgvm/shap_report.json` and the two PNGs.

`train_mrgvm.py` prints the per-split class distribution and warns for every
class holding fewer than five clips *before* it trains.

Full interpretation, substitutions and known feature weaknesses:
[`training_log.md`](training_log.md).

---

## Repository layout

```
pipeline/                 Phases 1-2
  phase1_preprocessing.py  phase2_landmarks.py  fetch_models.py
  engagement_pipeline/     alignment, mrs, pose, faces, video, dataset, io
mrgvm/                    Phases 3-7
  mamba.py                 pure-PyTorch S6 selective scan, bidirectional blocks
  vision_mamba.py          patch embed, spatial + reliability-guided temporal encoders
  geometric.py             landmark geometric descriptors
  fusion.py                adaptive gated fusion + lightweight MLP head
  model.py                 assembled Phase 3+4+5 model
  data.py  train_mrgvm.py  extract_embeddings.py  ablation.py  shap_explain.py
  README.md                architecture and design rationale
src/                      transformer baseline stack (not a PDF phase)
  features.py  affect.py  datasets.py  models.py  baselines.py  metrics.py  train.py
  early_detection.py       STUB — label-shift early detection
  explain.py               STUB — attention rollout, deletion test
outputs/
  README.md                run commands + full-dataset scaling notes
  results/  results_mrgvm/ metrics, ablation table, SHAP report
  logs/                    phase logs and resolved configs
training_log.md
requirements.txt
```

## Not yet implemented

Two stubs with fixed signatures, from the wider project plan rather than the
PDF's seven phases:

- **`src/early_detection.py`** — predicting clip *t+1*'s engagement from clip *t*.
  The ordering problem is solved and documented: `ClipID = <6-digit SubjectID><suffix>`;
  left-pad the suffix to 4 digits, first digit is the session, last three a
  strictly-increasing within-session index. *Inferred from the ID pattern, not
  documented by the DAiSEE authors*, and indices are not dense, so only gap-of-1
  pairs are safely adjacent.
- **`src/explain.py`** — attention rollout and a deletion test for the
  transformer baselines. Phase 7 SHAP for MRG-VM is fully implemented in
  `mrgvm/shap_explain.py`.
