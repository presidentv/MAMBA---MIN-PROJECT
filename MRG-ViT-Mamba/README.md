# MRG-ViT–Mamba: Reliability-Guided ViT + Mamba Temporal Modelling for Student Engagement on DAiSEE

A ViT spatial encoder per frame, a Motion Reliability Score (MRS) that says how much
each frame can be trusted, a Mamba selective state-space model across frames, and a
MediaPipe geometric/behavioural branch fused into a 4-class engagement classifier.

**Research question.** Can reliability-guided ViT–Mamba temporal modelling improve
robust student engagement recognition from webcam video while preserving temporal
information and avoiding unnecessary influence from unreliable observations?

This project is **not** framed as beating the highest published DAiSEE accuracy, and
the results below do not do so. Read [Results](#9-results) and
[Limitations](#11-limitations) before quoting any number from this repository.

---

## Read this first

Two facts determine how everything below should be interpreted.

**1. The dataset available here is a 108-clip subset, not full DAiSEE.**
Full DAiSEE is 9,068 clips from 112 users. The copy on this machine
(`Datasets/DAiSEE_Small`) contains **108 clips from 16 subjects** — 36 train / 36
validation / 36 test. The code is written for the full release and scales to it by a
config path change, but every number reported here comes from the 108-clip subset and
is **not comparable to published DAiSEE results**.

**2. On this subset, the trained model does not beat trivial baselines.**
Averaged over 5 seeds, the full model reaches test macro-F1 **0.214 ± 0.024**. A
stratified-random predictor reaches **0.242**, and always predicting the majority
class reaches accuracy **0.528** against the model's **0.278**. With 36 training clips
from 4 subjects, the model memorises the training split (train macro-F1 → 0.94) and
learns nothing that transfers. This is reported as it stands. What this repository
delivers is a **verified, checkpointed pipeline**, not an engagement result.

---

## Contents

1. [Architecture](#1-architecture)
2. [Verified environment](#2-verified-environment)
3. [Dataset setup and audit](#3-dataset-setup-and-audit)
4. [Preprocessing](#4-preprocessing)
5. [Motion Reliability Score](#5-motion-reliability-score-mrs)
6. [Model components](#6-model-components)
7. [Commands](#7-commands)
8. [Checkpoint outputs](#8-checkpoint-outputs)
9. [Results](#9-results)
10. [Three bugs the checkpoints caught](#10-three-bugs-the-checkpoints-caught)
11. [Limitations](#11-limitations)
12. [Exact configuration for the reported results](#12-exact-configuration-for-the-reported-results)
13. [Acceptance checklist](#13-acceptance-checklist)
14. [Training on the full DAiSEE release](#14-training-on-the-full-daisee-release)

---

## 1. Architecture

```
DAiSEE clip (.avi, 640x480, 30 fps, 10 s)
   |
   |  official Train / Validation / Test split, verified subject-disjoint
   v
uniform temporal sampling      i_k = round(k(N-1)/(T-1)),  k = 0..T-1,  T = 16
   |
   v
MediaPipe FaceDetector  -->  bbox + confidence + eye keypoints
   |                              |
   |                              +--> roll alignment --> crop --> 256 px (cached as JPEG)
   v
MediaPipe FaceLandmarker -->  478 landmarks + 52 blendshapes + 4x4 transform matrix
   |                              |
   |                              +--> 28 geometric features + 52 blendshapes = L = 80 / frame
   v
Motion Reliability Score  r_t in [0,1]      (blur, face visibility, head pose,
   |                                          eye visibility, motion consistency)
   v
ViT-B/16 frozen, per frame          [B,T,3,224,224] -> [B*T,3,224,224] -> [B,T,768]
   |
   v
reliability weighting               f'_t = r_t * f_t                       [B,T,768]
   |
   v
Linear projection 768 -> 256        (deliberately NOT followed by a norm; see section 10)
   |
   v
Mamba x 2  (d_model 256, d_state 16, d_conv 4, expand 2)                   [B,T,256]
   |
   v
reliability-weighted pooling        D = sum_t r_t h_t / sum_t r_t           [B,256]
   |
   |                       MediaPipe clip features: mean || std || mean|delta|   [B,240]
   |                                    |
   v                                    v
LayerNorm -> Linear -> 256        StandardScaler(train) -> Linear -> 256
   |                                    |
   +---------------- concat ------------+
                     |
                     v                                                     [B,512]
              MLP 512 -> 256 -> 128 -> 4  (GELU, dropout 0.3)
                     |
                     v
              4 logits -> cross-entropy (class-weighted)                   [B,4]
                     |
                     v
                   SHAP
```

**Terminology.** This is **ViT + Mamba temporal modelling**: a ViT spatial encoder per
frame, then Mamba across frames. It is deliberately *not* called Vim or Vision Mamba,
which is a visual backbone built from bidirectional Mamba blocks over *image tokens*
([hustvl/Vim](https://github.com/hustvl/Vim)). A Vim-style bidirectional scan is
available here as a labelled ablation arm, applied over the temporal axis only.

---

## 2. Verified environment

Recorded by `scripts/test_env.py` into `artifacts/environment.json`. These are the
versions the reported results were produced with, not a wish list.

| Component | Version |
|---|---|
| Python | 3.13.3 |
| OS | Windows 11 (10.0.26200) |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| CUDA (torch) | 12.8 |
| GPU | NVIDIA GeForce RTX 3060 Laptop, 6.0 GB, sm_86 |
| timm | 1.0.29 |
| transformers | 5.16.1 |
| OpenCV | 5.0.0 |
| MediaPipe | 1.0.1 (Tasks API) |
| scikit-learn | 1.9.0 |
| SHAP | 0.52.0 |
| einops | 0.8.2 |
| **Mamba backend** | **`pytorch_reference` (vendored)** |

### The Mamba backend, stated plainly

`pip install mamba-ssm` was attempted on this machine and **failed**. Verbatim:

```
UserWarning: mamba_ssm was requested, but nvcc was not found. Are you sure your
environment has nvcc available? ...
torch.__version__ = 2.14.0+cpu
...
packaging/version.py", line 200, in __init__
    match = self._regex.search(version)
TypeError: expected string or bytes-like object, got 'NoneType'
```

PyPI ships `mamba-ssm` as a source distribution only; its build requires the CUDA
Toolkit (`nvcc`), which is not installed here, and Windows is not an officially
supported build target for it.

`src/mamba_ref.py` therefore **vendors the official reference (non-fused)
implementation** from [state-spaces/mamba](https://github.com/state-spaces/mamba)
(Apache-2.0; Gu & Dao, [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)) —
`selective_scan_ref`, the `mamba_simple.Mamba` slow path, `RMSNorm`, and the pre-norm
residual block.

The fused CUDA kernel and this reference path **compute the same function**. The kernel
fuses the scan and recomputes to avoid materialising the `[B,D,L,N]` state; the
reference materialises it and loops over the sequence. Architecture, parameters and
outputs are identical; speed and memory are substantially worse. That is acceptable
here because the sequence is T frames (8–64), not thousands of tokens.

This is a Mamba, not an LSTM or Transformer standing in for one. It is verified as such
(section 8, CHECKPOINT 7), and `mamba_backend` is recorded in **every** artifact this
repository writes. If `mamba_ssm` ever becomes importable, `build_mamba_block` picks it
up automatically and the recorded backend changes to `mamba_ssm_cuda`.

---

## 3. Dataset setup and audit

See [`data/README.md`](data/README.md) for the expected directory layout. DAiSEE is
licence-restricted and is video of identifiable people; it is not in this repository and
must not be redistributed.

```bash
python scripts/audit_dataset.py --probe-all
```

This resolves every clip, joins it to its label row, decode-probes each video, verifies
the label domain, and checks subject disjointness. It **exits non-zero and blocks the
pipeline** if anything fails. Measured on this copy:

| | train | validation | test |
|---|---:|---:|---:|
| clips | 36 | 36 | 36 |
| subjects | 4 | 4 | 8 |
| Very Low (0) | 4 | 4 | 3 |
| Low (1) | 6 | 4 | 5 |
| High (2) | 16 | 17 | 19 |
| Very High (3) | 10 | 11 | 9 |

* **Decoding:** 108/108 decodable, 0 corrupt, 0 missing.
* **Video properties (measured, not assumed):** 30.0 fps for all 108; 640×480 for all
  108; 300 frames for 107 clips and **314 for one** (10.47 s) — which is why the code
  reads the true decoded frame count rather than trusting `CAP_PROP_FRAME_COUNT`.
* **Subject overlap:** train∩val = 0, train∩test = 0, val∩test = 0. The official DAiSEE
  directory split is subject-separated and is used as-is; no re-splitting is performed.
* **Labels:** the encoded `Engagement` values observed are exactly `{0,1,2,3}`, matching
  the documented four intensity levels (Very Low / Low / High / Very High). The CSV
  integer is used directly as the class index with no remapping. Verified into
  `artifacts/label_mapping.json`.

---

## 4. Preprocessing

Stage A is split into two cached stages, because they have different costs and
different invalidation conditions.

| | Stage 1 (`faces_*`) | Stage 2 (`vit_*`) |
|---|---|---|
| Work | decode, sample, detect, crop, landmarks, raw MRS | frozen ViT forward |
| Device | CPU (MediaPipe) | GPU |
| Depends on backbone | **no** | yes |
| Cost (108 clips, T=16) | ~40 s | ~27 s |

Keeping them apart is what makes an honest backbone comparison affordable: five
backbones cost five Stage 2 passes, not five repeats of the expensive MediaPipe sweep.
At full-DAiSEE scale that is the difference between ~30 minutes and several hours per
backbone.

**Stage 1 caches raw, uncalibrated MRS signals** — blur as raw variance-of-Laplacian,
face size as a raw area fraction, head pose in degrees. Consequently the MRS
calibration can be refitted and the five MRS weights changed **without decoding a
single video again**. That matters because the calibration must be fitted on the
training split only, which is not knowable until the training split has been swept.

Cache keys are SHA-1 digests over everything that would invalidate the contents
(preprocessing version, frame count, crop settings, detector thresholds, backbone,
input size, corruption). MRS *weights* are deliberately excluded from the key, and a
unit test asserts that.

**Face pipeline, as recorded in `artifacts/face_pipeline_report.json`:**

| Setting | Value |
|---|---|
| detector | `mediapipe.tasks.vision.FaceDetector` (blaze_face_short_range), MediaPipe 1.0.1 |
| confidence threshold | 0.3 |
| crop padding | 0.25 of box size per side |
| alignment | roll-aligned on the two detector eye keypoints |
| input resolution | 224 (backbone-resolved); crops stored at 256 |

Measured on 144 sampled frames across 9 clips: **100.0% face detection**, **100.0%
landmark detection**, zero fallbacks. Over the whole 1,728-frame corpus the landmark
rate is **99.48%** (9 frames without a face).

**Missing-face policy (spec §9).** A frame with no detection is *never dropped* — that
would corrupt the temporal spacing. It falls through an explicit, recorded chain:
detector box → landmark-derived box → centre crop, with `bbox_source` stored per frame,
`found=False`, and MRS scoring it as unreliable. The landmark feature row becomes zeros,
and clip-level mean/std are computed over detected frames only so a run of missing
frames cannot drag every statistic toward zero.

Visual check: `artifacts/face_crop_preview.png` (not committed — it is face imagery of
research subjects).

---

## 5. Motion Reliability Score (MRS)

MRS answers *"how much should I trust what I can see in this frame?"* — **not** *"how
engaged is this student?"* It is computed purely from observation quality and never
touches the engagement label.

```
MRS_t = w_b*B_t + w_f*F_t + w_h*H_t + w_e*E_t + w_m*M_t
```

with every component in [0,1] and `w_b = w_f = w_h = w_e = w_m = 0.2`. These weights
are a **starting point, not a tuned or optimal setting**.

| Term | What it measures | How |
|---|---|---|
| `B` blur | sharpness | variance of Laplacian on the face crop, mapped through **train-fitted** log percentiles |
| `F` face visibility | is there a clear, large-enough face | detector confidence × **train-fitted** area-fraction percentile ramp |
| `H` head pose | is frontal appearance evidence available | `max(|yaw|,|pitch|)`, 1.0 below 15°, 0.0 above 60°. Roll is excluded — the crop is roll-aligned, so roll costs no information |
| `E` eye visibility | were the eye/iris regions localisable | fraction of 6 key eye/iris points inside the frame + iris resolved distinctly from the eye centre |
| `M` motion consistency | is the observation stable | median landmark displacement per second, normalised by inter-ocular distance; falls back to **ZNCC** on the face crop |

Two deliberate design choices, both required by the spec:

* **A closed eye is not unreliable.** Blinking is normal behaviour, so `E` scores
  whether the eye region was *localisable*, not whether it was open. Penalising a low
  eye-aspect-ratio here would leak behaviour into the reliability signal.
* **A turned head is not disengagement.** `H` says frontal appearance evidence is
  weaker. Nothing more. Head pose is never converted into an engagement label.

**Motion consistency uses ZNCC rather than raw pixel difference** for the fallback,
because a raw difference responds to lighting and global camera shift as strongly as to
subject motion (spec §10.5).

### Calibration is fitted on training data only

```bash
python scripts/fit_mrs_stats.py     # reads TRAIN clips only; writes artifacts/mrs_calibration.json
```

Blur and face size have no dataset-independent scale. Hard-coded thresholds made both
near-constant on DAiSEE — the first pass had `face_visibility` averaging 0.39 with a
hard-coded "full face = 15% of frame", when the measured range on this corpus is
**3.5%–15.5%** of frame area. That wasted two fifths of the score's dynamic range, so
both are now percentile-mapped from training frames.

Fitted on 576 training frames: blur log bounds `[2.543, 4.542]`, area bounds
`[0.0416, 0.1395]` (5th/95th percentiles).

### Measured distribution — `artifacts/mrs_report.json`, 1,728 frames

| component | min | mean | max | std |
|---|---:|---:|---:|---:|
| blur | 0.0000 | 0.6022 | 1.0000 | 0.2628 |
| face_visibility | 0.0000 | 0.2582 | 0.9743 | 0.2959 |
| head_pose | 0.4356 | 0.9859 | 1.0000 | 0.0596 |
| eye_visibility | 0.0000 | 0.9948 | 1.0000 | 0.0720 |
| motion_consistency | 0.0000 | 0.4012 | 1.0000 | 0.3556 |
| **MRS** | **0.2976** | **0.6484** | **0.9880** | **0.1272** |

No NaN, no Inf, everything inside [0,1]. Motion was measured by landmark displacement
for 1,609 frames, by the ZNCC fallback for 11, and was undefined (first frame of a clip)
for 108.

**An honest observation about two of the five terms.** On this corpus `head_pose`
(std 0.060) and `eye_visibility` (std 0.072) are nearly constant: DAiSEE participants
sit facing a webcam, so there is very little head rotation and the eyes are almost
always localisable. Those two terms contribute little variation here. `blur`,
`face_visibility` and `motion_consistency` carry essentially all of the signal. This is
a property of the data, not a defect, but it means the equal-weight baseline is
probably not the right weighting for DAiSEE.

**Visual validation.** `artifacts/mrs_extremes.png` shows the 8 lowest- and 8
highest-MRS frames. The lowest are hand-occluded faces, heads tilted down out of frame,
and small/dark crops; the highest are large, sharp, frontal faces. Critically, the
high-MRS set includes neutral and frowning expressions and the low-MRS set includes an
attentive-looking subject behind a hand — MRS is tracking *observability*, not
engagement. (Not committed: face imagery.)

---

## 6. Model components

### ViT — spatial information within each frame

The embedding dimension is **discovered from the loaded model**, never assumed to be
768, and so are the input resolution and normalisation statistics. That last point
matters: the ImageNet ViT-B/16 used here wants mean/std `(0.5,0.5,0.5)`, not the
ImageNet constants, and a CLIP backbone wants something different again. Using the
wrong normalisation costs accuracy without raising anything.

Batching is `[B,T,C,H,W] → [B*T,C,H,W] → [B*T,D] → [B,T,D]`, done once, not in a Python
loop over frames. Measured: input `(2,16,3,224,224)` → **D = 768** → `(2,16,768)`.

The backbone is frozen (spec §13: freeze first, fine-tune only after the pipeline
works). Fine-tuning is not part of the reported results.

### MediaPipe — geometric and behavioural features

`mediapipe.tasks.vision.FaceLandmarker`, configured to return **more than landmarks**:

* 478 landmarks (including the 10 iris points),
* **52 ARKit blendshape activations** — `eyeBlinkLeft`, `browDownRight`, `jawOpen`,
  `mouthSmileLeft`, … ,
* a 4×4 facial transformation matrix, from which head pose is decomposed.

The blendshapes are the output of a pretrained head trained for exactly this kind of
expression read-out, so they are far more informative per dimension than concatenated
raw coordinates — which spec §12 explicitly warns against. They also restore
action-unit-style interpretability to the SHAP analysis.

Feature construction, all normalised by a face-scale reference (inter-ocular distance or
face width) so absolute pixel size cannot dominate:

* **28 geometric**: EAR per eye + mean + asymmetry, lid separation, iris offsets
  (a scale-free gaze proxy) and vergence, mouth aspect ratio / width / corner lift,
  brow–eye distances and asymmetry, head yaw/pitch/roll, face area and aspect,
  nose offset, inter-ocular ratio;
* **52 blendshapes**.

**Measured L = 80 per frame.** Clip aggregation is `mean ‖ std ‖ mean|Δ|` giving
**3L = 240**. Both dimensions are discovered and recorded in
`artifacts/landmark_feature_stats.json`, never assumed. The blendshape column order is
*verified* against the canonical 52-name list at runtime and raises if the model returns
a different set, rather than letting feature columns shift silently.

### Mamba — temporal dependencies

`[B,T,768] → Linear → [B,T,256] → 2 × Mamba → [B,T,256] → pooling → [B,256]`.

Config: `d_model 256`, `n_layers 2`, `d_state 16`, `d_conv 4`, `expand 2`
(→ `d_inner 512`, `dt_rank 16`). These are starting values, not proven optimal.

### Normalisation and projection are separate operations

Spec §18 is explicit that `y = Wx + b` changes dimensionality and representation but
does **not** control scale or distribution. Each branch is therefore normalised first,
then projected, as two separate modules:

* deep branch: `LayerNorm(256) → Linear(256→256) → GELU → Dropout`
* landmark branch: `StandardScaler(240) → Linear(240→256) → GELU → Dropout`

The landmark `StandardScaler` is fitted on the **training split only** and stored in the
module's **buffers**, so the statistics travel with the checkpoint and cannot drift
between training and evaluation. It refuses to run before being fitted (unit-tested),
and constant columns get scale 1 rather than exploding.

### Fusion and classifier

`concat` is the primary implementation: `F = [D_p ‖ L_p]` → `[B,512]`. A `gated`
alternative is implemented with its formula stated explicitly —
`a = sigmoid(W[D_p‖L_p]+b)`, `F = [a·D_p ‖ (1-a)·L_p]` — and is **not** claimed to be
better; it is an ablation arm.

Classifier: `512 → 256 → GELU → Dropout(0.3) → 128 → GELU → Dropout → 4` raw logits,
cross-entropy with class weights computed from the **training split only** (a class
absent from training gets weight 0 and a stated warning, not an infinity that would NaN
the first backward pass).

**Trainable parameters: 1,365,892** (temporal 1,073,152 · deep branch 66,304 · landmark
branch 61,696 · classifier 164,740). The 85.8 M ViT parameters are frozen.

---

## 7. Commands

```bash
# 0. environment + models
python scripts/test_env.py                       # TEST 1  -> artifacts/environment.json
python scripts/fetch_models.py                   # MediaPipe bundles -> models/

# 1. dataset gate (blocks everything downstream if it fails)
python scripts/audit_dataset.py --probe-all      # CHECKPOINT 1 + 2

# 2. preprocessing and calibration
python scripts/run_preprocessing.py --stage 1    # faces, landmarks, raw MRS  (CPU)
python scripts/fit_mrs_stats.py                  # TRAIN-only MRS calibration
python scripts/run_preprocessing.py --stage 2    # frozen ViT features        (GPU)

# 3. component checkpoints
python scripts/test_face_pipeline.py --clips 3   # CHECKPOINT 3  (TEST 2, 3)
python scripts/test_mrs.py                       # CHECKPOINT 4  (TEST 4)
python scripts/test_landmarks.py                 # CHECKPOINT 5  (TEST 5)
python scripts/test_vit.py                       # CHECKPOINT 6  (TEST 6, 7)
python scripts/test_mamba.py                     # CHECKPOINT 7  (TEST 8)
python scripts/test_full_pipeline.py             # TEST 9, 10, 11, 12
python -m pytest tests -q                        # 38 unit tests

# 4. training and evaluation
python scripts/run_training.py --run-name main
python scripts/run_training.py --run-name held_out --no-test   # never touches test

# 5. experiments
python scripts/compare_backbones.py              # 5 ViT backbones, train->val probe
python scripts/run_ablation.py                   # 7 arms x 5 seeds + trivial baselines
python scripts/run_robustness.py                 # 10 degradations of the test split
python scripts/run_sampling_experiment.py        # T = 8, 16, 32, 64
python scripts/run_shap.py --checkpoint checkpoints/main/best.pt
```

Every script exits non-zero when its checkpoint fails, so they can be chained in CI.

---

## 8. Checkpoint outputs

All twelve tests and seven checkpoints were run. Actual measured outputs:

| Gate | Result | Evidence |
|---|---|---|
| TEST 1 environment | **PASS** | all required packages present; CUDA available |
| **CHECKPOINT 1** dataset audit | **PASS** | 108/108 decodable, labels ⊆ {0,1,2,3}, 4 classes in every split |
| **CHECKPOINT 2** split audit | **PASS** | all three subject overlaps = 0 |
| TEST 2/3, **CHECKPOINT 3** face pipeline | **PASS** | 144/144 frames detected (100%), 0 boundary violations, 0 empty crops, montage inspected |
| TEST 4, **CHECKPOINT 4** MRS sanity | **PASS** | 1,728 frames; all components ∈ [0,1]; no NaN/Inf; MRS std 0.127; extremes visually inspected |
| TEST 5, **CHECKPOINT 5** landmarks | **PASS** | L = 80 fixed, 3L = 240, no NaN/Inf, blendshapes ∈ [0, 0.983], 99.48% detection |
| TEST 6, **CHECKPOINT 6** ViT shapes | **PASS** | `(2,16,3,224,224)` → **D = 768** → `(2,16,768)`, finite |
| TEST 7 MRS weighting | **PASS** | shape preserved; per-position scale equals `r_t` with max deviation **0.0** |
| TEST 8, **CHECKPOINT 7** Mamba | **PASS** | see below |
| TEST 9 fusion | **PASS** | deep 256, landmark 256, fused **512**, logits `(4,4)` |
| TEST 10 full forward | **PASS** | real batch → `logits.shape == [4,4]`, finite |
| TEST 11 one training step | **PASS** | loss 1.3767 → 1.2554, grad norm 1.049, 35/35 tensors updated, no NaN |
| TEST 12 tiny overfit | **PASS** | 16 clips, 250 steps: loss **1.379 → 1.7e-07**, train accuracy **1.000** |
| Unit tests | **38 passed** | `python -m pytest tests -q` |

### CHECKPOINT 7 in detail

Because the Mamba is vendored rather than pip-installed, shape checks alone are not
evidence of correctness. `scripts/test_mamba.py` verifies the properties that make it a
selective SSM at all:

| Property | Measured |
|---|---|
| Causality: perturb input at `t ≥ 8`, measure change at `t < 8` | **0.0** (exactly) |
| Same perturbation, change at `t ≥ 8` | 5.704 (the module does respond) |
| Bidirectional control, change at `t < 8` | 0.227 — confirms the causality probe is *sensitive*, not trivially passing |
| `selective_scan_ref` vs. an independently written naive S6 recurrence, float64 | **max abs error 0.0** |
| Parameters receiving no gradient | 0 |

---

## 9. Results

> Everything in this section comes from the **108-clip subset**. It is not comparable to
> published DAiSEE numbers, which use the full 9,068-clip release.

### 9.1 Baselines first

A 4-class macro-F1 is meaningless without knowing what trivial predictors score on the
same split. On the 36-clip test split:

| Predictor | Accuracy | Macro-F1 |
|---|---:|---:|
| Always majority class (High, taken from **train**) | **0.528** | 0.173 |
| Stratified random from the train prior (mean of 200 draws) | 0.337 | **0.242** |

### 9.2 The single `main` run

`artifacts/test_metrics_main_test.json`, seed 42, checkpoint chosen at epoch 8 by
validation macro-F1. Reported because it is the run the SHAP analysis and the committed
confusion matrix come from — the multi-seed table below is the number to quote.

| | Acc-4 | Weighted F1 | Macro-F1 |
|---|---:|---:|---:|
| main (36 test clips, 8 subjects) | 0.3056 | 0.3021 | 0.2397 |

Per class: Very Low P 0.167 / R 0.667 / F1 0.267 (n=3) · Low P 0.000 / R 0.000 / F1 0.000
(n=5) · High P 0.714 / R 0.263 / F1 0.385 (n=19) · Very High P 0.235 / R 0.444 / F1 0.308
(n=9). **Class 1 (Low) is never predicted.** A single run on 36 clips is not a result;
see the seed spread below.

### 9.3 Ablation — 7 arms × 5 seeds (mean ± std)

`artifacts/ablation_results.json`. Each arm's checkpoint is selected by its own
validation macro-F1; test is read once per run as the final evaluation. **No arm was
selected using test performance.**

| Arm | val macro-F1 | test accuracy | test macro-F1 |
|---|---:|---:|---:|
| `full` (multiply + reliability pooling) | 0.302 ± 0.075 | 0.278 ± 0.035 | 0.214 ± 0.024 |
| `multiply_only` (spec's literal `f'_t = r_t f_t`) | 0.298 ± 0.074 | 0.289 ± 0.045 | 0.217 ± 0.029 |
| `pool_only` (reliability only in pooling) | 0.284 ± 0.071 | 0.283 ± 0.032 | 0.216 ± 0.029 |
| `no_mrs` (r_t = 1 everywhere) | 0.289 ± 0.068 | 0.272 ± 0.044 | 0.205 ± 0.029 |
| `residual_mrs` | 0.313 ± 0.073 | 0.250 ± 0.030 | 0.189 ± 0.016 |
| `no_landmarks` (deep branch only) | 0.289 ± 0.065 | 0.278 ± 0.086 | 0.170 ± 0.046 |
| `bidirectional` (Vim-style temporal scan) | 0.281 ± 0.018 | 0.306 ± 0.105 | 0.219 ± 0.030 |
| *[majority baseline]* | – | *0.528* | *0.173* |
| *[stratified random]* | – | *0.337* | *0.242* |

**What this actually says.**

1. **No arm beats the trivial baselines.** The best test macro-F1 (0.219) is below
   stratified random (0.242); the best test accuracy (0.306) is far below always-majority
   (0.528). On this subset the model has not learned generalisable engagement.
2. **The arms are indistinguishable from each other.** Every difference is smaller than
   the seed-to-seed standard deviation. With 36 test clips, one clip is 2.8 percentage
   points of accuracy. **No conclusion should be drawn about whether MRS weighting helps.**
3. The only gap approaching the noise floor is `no_landmarks` being worst on macro-F1
   (0.170 vs 0.214), which is *consistent with* the MediaPipe branch contributing
   something — but at ±0.046 it is not established.
4. Training diverges from validation almost immediately: train macro-F1 reaches 0.94
   while validation loss rises monotonically from about epoch 4, and best checkpoints
   land at epoch 3–36. That is the signature of 36 training clips from 4 subjects, not
   of a broken pipeline — TEST 12 confirms the model can fit data when it is asked to.

### 9.4 Backbone comparison

`artifacts/backbone_comparison.json`. Protocol: multinomial logistic-regression probe on
MRS-weighted mean-pooled frozen features, scaler and probe fitted on **train**, scored on
**validation**. The test split is never read by this script.

| Backbone | D | val accuracy | val macro-F1 |
|---|---:|---:|---:|
| `timm:vit_base_patch16_224.augreg2_in21k_ft_in1k` | 768 | 0.361 | 0.279 |
| `hf:trpakov/vit-face-expression` | 768 | 0.333 | 0.232 |
| `timm:vit_base_patch16_clip_224.openai` | 768 | 0.333 | 0.228 |
| `hf:dima806/facial_emotions_image_detection` | 768 | 0.250 | 0.157 |
| `timm:vit_small_patch14_dinov2.lvd142m` | 384 | 0.222 | 0.136 |

**Every backbone reached train accuracy 1.000.** With 36 training clips and 768-d
features, a linear probe separates the training set perfectly regardless of backbone
quality, so this table measures overfitting as much as representation quality.
**The ranking is not statistically meaningful at this scale** and ImageNet ViT-B/16 is
*not* claimed to be better than the face-expression fine-tunes — it is kept as the
default because it ranked top here and is the conventional reference point. The
comparison machinery exists so it can be re-run on full DAiSEE, where it would mean
something.

One methodological caveat specific to DINOv2: its native input is 518 px, but the Stage 1
cache stores 256 px crops, so it received upsampled input. Its last-place finish is
partly an artifact of that choice and should not be read as a verdict on DINOv2.

### 9.5 A pretrained-model finding worth stating separately

While loading the HuggingFace face-expression checkpoints, transformers reported
`pooler.dense.weight | MISSING — newly initialized`. A HF image-classification checkpoint
stores no pooler weights, so `AutoModel` **randomly initialises one**. Reading
`pooler_output` would therefore have returned the output of an untrained layer and
silently discarded everything those fine-tunes learned. The encoder now builds with
`add_pooling_layer=False` and reads the **CLS token**, which is what
`ViTForImageClassification` actually classifies from. Anyone benchmarking HF vision
checkpoints as frozen feature extractors should check for this.

### 9.6 Robustness under controlled degradation

`artifacts/robustness_results.json`. Degradations are applied **in memory** to decoded
frames during Stage 1; the DAiSEE files on disk are never modified, and each degraded
sweep gets its own cache key. Both arms were trained on **clean** data only and are
evaluated unchanged. `baseline` = the `no_mrs` checkpoint (no reliability signal),
`proposed` = the `full` checkpoint.

| Degradation | MRS mean | face detection | baseline macro-F1 | proposed macro-F1 | Δ |
|---|---:|---:|---:|---:|---:|
| clean | 0.657 | 100.0% | 0.179 | 0.179 | +0.000 |
| blur σ=2 | 0.537 | 100.0% | 0.201 | 0.185 | −0.016 |
| blur σ=4 | 0.522 | 100.0% | 0.111 | 0.159 | +0.048 |
| downscale 4× | 0.536 | 100.0% | 0.228 | 0.297 | +0.068 |
| downscale 8× | 0.515 | 100.0% | 0.121 | 0.138 | +0.017 |
| occlusion 10% | 0.634 | 98.1% | 0.251 | 0.246 | −0.006 |
| occlusion 25% | 0.484 | **32.3%** | 0.231 | 0.230 | −0.002 |
| dark (0.5× contrast, −30) | 0.492 | 100.0% | 0.176 | 0.208 | +0.032 |
| bright (1.4× contrast, +45) | **0.691** | 100.0% | 0.195 | 0.195 | +0.000 |
| JPEG q15 | 0.634 | 100.0% | 0.180 | 0.180 | +0.000 |

**Finding 1 — MRS does detect degradation.** This is the clearest positive result in the
project. Mean MRS falls from 0.657 on clean data to 0.522 (blur σ=4), 0.515
(downscale 8×), 0.492 (dark) and 0.484 (occlusion 25%), and face detection collapses
from 100% to 32.3% under heavy occlusion. The reliability score is measuring what it
claims to measure, on inputs it was never tuned against.

**Finding 2 — MRS is fooled by contrast enhancement.** The `bright` condition
(1.4× contrast, +45 brightness) *raised* mean MRS to **0.691, above clean**. Increasing
contrast increases the variance of the Laplacian, so the blur term reads an
artificially brightened frame as *sharper* than the original. That is a real weakness of
a variance-of-Laplacian sharpness proxy and it is not fixed by percentile calibration,
which rescales the measure without changing what it responds to. Anyone reusing this MRS
formulation should know this.

**Finding 3 — no robustness claim is supported.** The proposed arm is ahead on 5 of 9
degradations and behind or level on 4, with Δ ranging −0.016 to +0.068. Both arms sit
near or below the trivial baselines throughout. Decisively, `downscale_4x` produces the
*best* scores of any condition for both arms — better than clean — which can only be
noise. **These numbers do not show that MRS weighting improves robustness.** As spec §33
puts it, the existence of an MRS term is not evidence of robustness, and this experiment
does not supply that evidence; at 36 test clips it is not capable of doing so.

### 9.7 Temporal sampling — T = 8 / 16 / 32 / 64

`artifacts/sampling_experiment.json`, 3 seeds per setting. Each T gets its own Stage 1
and Stage 2 cache, so no result can be contaminated by a stale one.

| T | test accuracy | weighted F1 | macro F1 | head ms/video | peak GPU MB | preprocessing (108 clips, all splits) |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.222 ± 0.000 | 0.200 ± 0.008 | 0.184 ± 0.003 | 8.8 | 377 | 94 s |
| 16 | 0.259 ± 0.035 | 0.244 ± 0.028 | 0.203 ± 0.023 | 15.2 | 378 | *(cached)* |
| 32 | 0.222 ± 0.039 | 0.205 ± 0.041 | 0.164 ± 0.006 | 21.7 | 380 | 255 s |
| 64 | 0.287 ± 0.069 | 0.276 ± 0.060 | 0.213 ± 0.065 | 78.6 | 384 | 465 s |

**The cost curve is reliable; the accuracy curve is not.** Compute scales as expected —
head inference goes 8.8 → 78.6 ms/video from T=8 to T=64, and preprocessing goes 94 s →
465 s. GPU memory is essentially flat (377 → 384 MB) because the sequence is short and
the frozen ViT is chunked.

Accuracy does not order cleanly: T=16 beats T=32, and T=64's macro-F1 advantage
(0.213 vs 0.203 at T=16) is a third of its own standard deviation. **This experiment does
not establish that more frames help**, and it certainly does not establish that 16 frames
is universally sufficient — it shows that at this data scale the frame count is not the
binding constraint. The 36-clip test split is the binding constraint.

One measurement worth noting for scaling: Stage 1 decodes every frame of a clip and then
samples, because per-frame seeking in these MPEG-4 AVIs lands on the nearest keyframe and
silently returns the wrong frame. Decoding cost is therefore roughly constant in T, and
the growth from 94 s to 465 s is MediaPipe running on more frames.

### 9.8 SHAP

`artifacts/shap_analysis_main_test.json` and two plots. Attribution is taken over
`[Mamba temporal embedding (256) ‖ landmark clip features (240)]` — one step *before*
fusion, so the landmark half keeps named features. Attributing over the fused 512-d
vector could only ever have said "deep vs. landmark", since both halves there are learned
projections with unnamed dimensions.

`shap.GradientExplainer` (expected gradients), 36 background and 36 evaluated clips.

| Group | dims | share of total | per-dimension |
|---|---:|---:|---:|
| deep temporal (Mamba) | 256 | 44.5% | 0.00259 |
| mouth / jaw | 90 | 29.1% | 0.00481 |
| iris / gaze | 45 | 7.7% | 0.00254 |
| brow | 27 | 6.7% | 0.00370 |
| eye openness | 36 | 4.5% | 0.00187 |
| head pose | 15 | 3.9% | 0.00389 |
| face scale | 9 | 2.8% | 0.00463 |
| cheek / nose | 15 | 0.9% | 0.00091 |
| neutral | 3 | 0.0% | 0.00000 |

Both columns are given because they answer different questions. The deep branch takes the
largest *total* share (44.5%) but it also has the most dimensions; **per dimension the
mouth/jaw blendshapes are the strongest contributors** (0.00481 vs 0.00259). Top
individual features: `mean_blendshape_mouthRollLower` (0.052),
`mean_blendshape_mouthDimpleLeft` (0.029), `mean_blendshape_browInnerUp` (0.021),
`delta_blendshape_mouthClose` (0.020).

**How much to read into this: very little.** These are the attributions of a model that
does not beat a random baseline. They describe what a poorly-generalising model latched
onto — plausibly subject-specific mouth idiosyncrasies across four training identities —
not what indicates engagement. SHAP is attribution, not causation, and attribution from a
model with no demonstrated skill is not evidence about the phenomenon.

The grouping is enforced as an exact partition of all 80 per-frame features at import
time. The first run pooled 66 dimensions (20% of total attribution, including the single
highest-attribution feature) into an uninterpretable `other_landmark` bucket; the
import-time check now makes that impossible.

### 9.9 Efficiency

`artifacts/test_metrics_*.json`. Conditions stated as spec §31 requires.

| | |
|---|---|
| Hardware | NVIDIA GeForce RTX 3060 Laptop (6 GB), CUDA 12.8 |
| Scope | cached-feature head only: MRS weighting + Mamba + fusion + MLP |
| Batch size | 1 |
| Warm-up / repetitions | 5 / 20 |
| Sampled frames | 16 |
| Input resolution | 224 |
| Trainable parameters (the checkpointed model) | **1,365,892** |
| Frozen ViT-B/16 parameters (preprocessing stage) | 85,798,656 |
| Whole pipeline | 87,164,548 |
| Inference, head only | **18.9 ms/video** (53 videos/s), peak 50 MB GPU |
| Mamba forward, `[4,16,768]`, reference path | 180 ms |

The two timings measure different things and neither is an end-to-end figure. The
**18.9 ms/video** is the trained head on one cached clip. The **180 ms** is a
`[4,16,768]` batch through the temporal module in isolation, which is what
CHECKPOINT 7 reports. The frozen ViT and MediaPipe are not in either — they are
preprocessing, and they dominate.

Preprocessing dominates end-to-end cost and is reported separately: Stage 1 ≈ 40 s and
Stage 2 ≈ 27 s for all 108 clips across all three splits. Extrapolating to full DAiSEE
(9,068 clips) gives roughly 55 minutes of Stage 1 and 38 minutes of Stage 2 on this
hardware. Timings from different hardware or batch conditions are not comparable to
these without stating those conditions.

### 9.10 Literature context — explicitly *not* a comparison

The MC-MIFA paper reports on DAiSEE: Vision Mamba (Vim-S) 76.10 accuracy / F1 0.724;
VMamba (Tiny) 82.60 / 0.801; MC-MIFA 97.45 / 0.968
([source](https://pmc.ncbi.nlm.nih.gov/articles/PMC13328615/)).

These are **literature reference values, not targets and not results of this project**.
They are not comparable to anything above: they use the full 9,068-clip dataset and this
work uses 108 clips, and the preprocessing and protocol differ. No claim is made in
either direction. Mamba has been applied to DAiSEE before; this project does not claim
otherwise.

---

## 10. Three bugs the checkpoints caught

All three were found by tests, not by inspection, and all three would have silently
produced plausible-looking numbers.

### 10.1 Scale-invariant normalisation was erasing the entire MRS mechanism

The ablation returned **bit-identical** metrics for the "MRS multiply" and "no MRS" arms
on 4 of 5 seeds. That is not noise.

`TemporalMamba` applied `nn.LayerNorm` immediately after the input projection. LayerNorm
and RMSNorm are **scale-invariant**, so for a per-frame scalar `r_t > 0`:

```
Norm(W (r_t f_t) + b)  ==  Norm(W f_t + b/r_t)  ->  Norm(W f_t)   as b -> 0
```

Multiplying features by the reliability score and then normalising puts the signal
straight back where it started. Only the projection bias survived — which is exactly why
one seed differed slightly instead of not at all. The same argument applies at the
output: `final_norm` renormalises every timestep to unit RMS, so a subsequent mean-pool
weights an unreliable frame exactly as much as a reliable one — the opposite of the
intent.

**Fix.** The input LayerNorm was removed, and reliability-weighted pooling
(`sum_t r_t h_t / sum_t r_t`) was added as the place where the signal can express itself
without being normalised away. Temporal positions are all still present and in order; no
frame is dropped. Two regression tests now guard this
(`test_reliability_weighted_pooling_actually_depends_on_mrs`,
`test_temporal_mamba_has_no_scale_removing_input_norm`).

This is worth stating in its own right: **the spec's literal formulation, `f'_t = r_t f_t`
followed by normalisation, is close to a no-op.** Where reliability is applied matters as
much as the formula.

### 10.2 Checkpoints were being rebuilt from the wrong config

`load_checkpoint` built the model from whatever config the *caller* held rather than the
config the checkpoint was *trained* with. An ablation checkpoint trained with mean pooling
and no MRS weighting has a `state_dict` that loads cleanly into a model built with
reliability-weighted pooling — the parameter shapes are identical — and would then be
evaluated with a mechanism it was never trained with. Nothing raises; the numbers are just
wrong. The architecture now always comes from `ckpt["config"]`, and the dataset's
reliability switch follows the checkpoint too.

### 10.3 A dtype bug in the vendored scan

`selective_scan_ref` followed the official code in upcasting
`u`/`delta`/`B`/`C` to float32 while leaving `A` at its own dtype. This is invisible in the
fp32 happy path and raises a hard `TypeError` the moment anything runs in float64 — which
is exactly what the equivalence test in CHECKPOINT 7 does. Fixed by promoting all operands
to a common working dtype, which preserves the official fp16→fp32 upcast intent while
letting a float64 caller keep its precision.

A fourth, caught while writing this README: `checkpoints/main/best.pt` from before fix
10.1 failed to load with `Unexpected key(s): temporal.input_norm.*`. That is the strict
`load_state_dict` doing its job — the stale checkpoint was retrained rather than loaded
with `strict=False`, which would have silently evaluated a model with an uninitialised
layer.

---

## 11. Limitations

**About this dataset copy**

* **108 clips, not 9,068.** Every number here comes from a subset roughly 1.2% the size
  of DAiSEE. Nothing here is comparable to published DAiSEE results.
* **4 training subjects.** Deep models can exploit subject-specific appearance, and with
  four identities the model has almost no way to learn anything else. This is the single
  biggest constraint on the results.
* **Single-digit minority classes.** The test split has 3 Very Low and 5 Low clips. One
  clip changes per-class recall by 20–33 percentage points, so per-class metrics are
  extremely unstable.
* **One test split, one protocol.** No cross-validation, and with a fixed official split
  there is no way to get a confidence interval over splits.

**About the method**

* **MRS weights are arbitrary.** Equal 0.2 weights are a baseline, not an optimum, and on
  this corpus two of the five terms (`head_pose`, `eye_visibility`) are nearly constant
  and contribute little.
* **MRS thresholds affect results.** The blur and area percentile bounds are fitted, but
  the head-pose 15°/60° ramp and the motion reference rate are hand-chosen.
* **Head-pose decomposition is uncalibrated.** The Euler convention is consistent across
  frames, which is all the reliability term needs, but signs are not validated against a
  head-pose benchmark.
* **The Mamba is the reference path, not the fused kernel.** Same function, ~180 ms per
  forward instead of a fused kernel's few ms.
* **Temporal sampling can miss short behaviours.** 16 frames over 10 s is one frame per
  0.33 s; a blink lasts 0.1–0.4 s. This is why frame count is configurable and measured.
* **MediaPipe fails under occlusion and extreme pose.** 9 of 1,728 frames here, but that
  rate will differ on harder data.

**About the labels and the framing**

* **DAiSEE engagement labels are affective intensity annotations** and should not be
  read as objective measurements of an internal mental state.
* **Adjacent engagement levels are visually similar.** High vs. Very High is a difficult
  distinction even for annotators.
* **SHAP is attribution, not causation.** It reports what this model leaned on, not what
  causes engagement.
* **Published results use different preprocessing, splits and metrics.** Protocol
  differences are stated wherever literature values appear.

---

## 12. Exact configuration for the reported results

`configs/config.yaml`, unmodified, at the commit recorded in each artifact's
`environment.git_commit`. The values that determine the reported numbers:

```yaml
seed: 42
dataset:  { num_classes: 4, target_label: Engagement }
video:    { num_frames: 16, sampling: uniform }
face:     { detector: mediapipe_face_detector, image_size: 224,
            min_detection_confidence: 0.3, crop_padding: 0.25, align: true }
mrs:      { weights: {blur: 0.2, face_visibility: 0.2, head_pose: 0.2,
                      eye_visibility: 0.2, motion_consistency: 0.2},
            weighting_mode: multiply, calibration_percentiles: [5.0, 95.0],
            head_pose_full_reliability_deg: 15.0,
            head_pose_zero_reliability_deg: 60.0,
            motion_ref_displacement: 0.08 }
vit:      { model_name: vit_base_patch16_224.augreg2_in21k_ft_in1k,
            pretrained: true, freeze: true, batch_size: 32 }
mamba:    { d_model: 256, n_layers: 2, d_state: 16, d_conv: 4, expand: 2,
            bidirectional: false, dropout: 0.1, pooling: mrs_weighted }
fusion:   { projected_dim: 256, mode: concat, use_landmark_branch: true,
            landmark_norm: standard, deep_norm: layernorm, dropout: 0.1 }
classifier: { hidden_dims: [256, 128], activation: gelu, dropout: 0.3 }
training: { batch_size: 8, epochs: 60, learning_rate: 0.0003, weight_decay: 0.01,
            optimizer: adamw, scheduler: cosine, warmup_epochs: 3, grad_clip: 1.0,
            mixed_precision: true, class_weighting: inverse_frequency,
            label_smoothing: 0.05, early_stopping_patience: 20,
            model_selection_metric: val_macro_f1 }
```

Fitted MRS calibration (`artifacts/mrs_calibration.json`, from 576 **training** frames):
`blur_log_low 2.5434`, `blur_log_high 4.5418`, `area_low 0.04157`, `area_high 0.13948`.

Cache keys for the reported run: Stage 1 `faces_T16_d54f87dc2c`, Stage 2
`vit_timm_vit_base_patch16_224.augreg2_in21k_ft_in1k_*`.

---

## 13. Acceptance checklist

Spec §37, answered against what was actually run. Each row names the artifact that
proves it.

**Dataset**

- [x] DAiSEE path verified — `artifacts/dataset_audit.json`
- [x] labels verified from actual metadata — `artifacts/label_mapping.json`
- [x] four engagement classes verified, present in all three splits
- [x] train/validation/test splits verified (official DAiSEE directory split, used as-is)
- [x] subject overlap checked — 0 / 0 / 0
- [x] corrupt/missing files reported — 0 corrupt, 0 missing, 108/108 decodable

**Preprocessing**

- [x] uniform temporal sampling works — formula unit-tested, edge cases covered
- [x] face detection works — 100% on the inspected sample, 99.5% landmark rate corpus-wide
- [x] face crops inspected — `artifacts/face_crop_preview.png`, visually verified
- [x] MRS values in [0,1] — 1,728 frames checked
- [x] MRS components inspected — `artifacts/mrs_report.json`, `mrs_histograms.png`, `mrs_extremes.png`
- [x] MediaPipe features have fixed dimensions — L = 80, 3L = 240, asserted per clip
- [x] no NaN/Inf anywhere in the cached features

**Model**

- [x] ViT forward pass works
- [x] actual ViT embedding dimension recorded — **768**, measured not assumed
- [x] MRS weighting preserves `[B,T,D]` — max scale deviation 0.0
- [x] Mamba forward pass works — plus causality and S6-recurrence equivalence
- [x] temporal pooling works — mean / last / attention / mrs_weighted
- [x] landmark branch works
- [x] normalisation works — train-fitted scaler stored in checkpoint buffers
- [x] projections work — both branches → 256
- [x] fusion dimension verified — **512**
- [x] MLP output is `[B,4]`

**Training**

- [x] one training step works — loss ↓, 37/37 tensors updated
- [x] tiny overfit test passes — 1.379 → 1.7e-07, accuracy 1.000
- [x] no NaN/Inf — training raises rather than logging a meaningless loss
- [x] checkpoint saves — `checkpoints/<run>/best.pt`
- [x] validation metrics log correctly — `logs/training_history_<run>.csv`
- [x] test set not used for tuning — selection is `val_macro_f1`; `--no-test` available

**Evaluation**

- [x] Acc-4, Weighted F1, Macro-F1, per-class precision/recall/F1
- [x] confusion matrix — raw and row-normalised
- [x] inference time — with hardware, batch size, warm-up and repetitions stated
- [x] parameter count — total, trainable, and per module
- [x] GPU memory — peak allocated, reported per run

**Explainability**

- [x] SHAP runs on the final classifier
- [x] feature contributions saved — `artifacts/shap_analysis_*.json` + two plots
- [x] interpretations stated as attribution, not causation

**Experiments**

- [x] ablation — 7 arms × 5 seeds, with trivial baselines
- [x] robustness — 10 degradations, originals unmodified
- [x] temporal sampling — T = 8/16/32/64
- [x] backbone comparison — 5 pretrained ViTs, train→val probe

---

## Anti-fabrication statement

No metric in this document was estimated, extrapolated or invented. Every number is
reproducible from a committed artifact in `artifacts/` or a log in `logs/`. Where an
experiment was not run, it is marked **NOT RUN** rather than filled in. Where a result is
negative — and the headline result here is negative — it is reported as negative.

## 14. Training on the full DAiSEE release

Everything above was measured on 108- and 216-clip subsets. `configs/config_full.yaml`
runs the same fine-tuned model — ViT-B/16 trained end to end, T = 32 frames, Mamba
temporal head, learnable MRS weights — on the complete release. It is written to run
unattended on a machine other than the one it was developed on.

### Hardware

Measured on the development machine (RTX 3060 Laptop, 6 GB) unless marked *estimate*.

| | Minimum (verified) | Recommended |
|---|---|---|
| GPU | NVIDIA, 6 GB VRAM, CUDA (with `vit.grad_checkpointing: true`); ~12 GB as configured | 16–24 GB+ (e.g. RTX A4000, 4090, A5000, A100) |
| CPU | 4 cores | 8–16+ cores — Stage 1 (MediaPipe) is CPU-bound and shards across cores |
| RAM | 16 GB | 32 GB (8 DataLoader workers) |
| Disk | ~45 GB free | 60 GB+ on SSD |
| OS | Windows 11 or Linux | Linux — `run_full_daisee.sh` is bash, and the fused `mamba-ssm` kernel needs `nvcc` |

GPU memory, from `scripts/probe_finetune_memory.py`: a training step over 128 crops
(4 clips × 32 frames) peaks at **2.7 GB** with gradient checkpointing; 256 crops peaks
at 4.2 GB, and **17.5 GB without checkpointing**. The config is set up for an
**RTX A4000 (16 GB)**, so **checkpointing is off** (`vit.grad_checkpointing: false`):
128 crops then need about 9–10 GB (*estimate*, from the 256-crop measurement), and
skipping the recomputation makes each epoch about 20–25% faster, with the same trained
model. **On a 6–8 GB card, set it to `true`**. That is what makes 6 GB enough. The
setting is not checked on `--resume`, so after an out-of-memory error you can switch it
on and resume. The config trains at an **effective batch of 32**
(`batch_size: 4` × `grad_accum_steps: 8`): a real batch of 32 clips is 1,024 crops, about
13 GB even with checkpointing. On a 16 GB+ card, raise `training.batch_size` and lower
`grad_accum_steps` so their product stays 32. Learning rates are 2× the development
values (√(32/8) scaling for the 4× larger effective batch).

Disk, per component:

| Item | Size |
|---|---|
| DAiSEE release (video) | ~13.5 GB (*estimate*: 1.49 MB/clip × 9,068, from the 216-clip subset) |
| Stage 1 cache at T = 32 | ~4.3 GB (0.49 MB/clip measured × 9,068) |
| One fine-tuned checkpoint (`best.pt`) | 333 MB (`save_every_epoch: true` stores every epoch: ~13 GB at 40) |
| `last.pt` (resume state incl. optimiser) | ~1 GB, overwritten each epoch |
| `after_epoch_010.pt`, `_020`, … (every `save_every_n_epochs: 10`) | ~1 GB each (weights + full resume state, like `last.pt`): ~2 GB if the run stops at 20, ~4 GB at 40 |

Time, *estimates* scaled from the development machine:

| Step | 6 GB laptop GPU, measured rate | Scaled to ~9,000 clips |
|---|---|---|
| Stage 1, one process | 0.62 s/clip | ~95 min; roughly ÷ number of shards |
| Fine-tuning epoch | 62 s for 120 train + 80 val clips | ~35–40 min per epoch on the same GPU; several times faster on a data-centre GPU |

*Estimate* for an **RTX 4060 (8 GB) desktop with 16 GB RAM**, assuming it is ~1.2–1.4×
the development GPU: ~30–33 min training + ~2 min validation per epoch, so **~11–12 h
if the run stops at the epoch-20 decision point, ~22–24 h if it continues to 40**, plus
~20–30 min of sharded Stage 1 and a few minutes of final evaluation. That estimate is
with gradient checkpointing on, which an 8 GB card needs.

*Estimate* for the **RTX A4000 (16 GB)** the config targets, with checkpointing off:
~20–23 min per epoch, so **~7–8 h to the epoch-20 decision point, ~14–16 h to 40**.
The real time per epoch and peak GPU memory are at the end of every `ep N | …` log
line.

**Epoch budget: 20, extended to 40 only if still improving, and only if you agree.** On
the development subsets the best epoch was 2 (fine-tuned) and validation loss rose from
about epoch 4 (frozen). A full-release epoch is ~11× more optimiser steps, so the peak
is expected around epochs 5–15. The first 20 epochs are one complete cosine cycle, so a
run that stops there has a fully annealed model. After epoch 20 the rule says continue
only if the best validation macro-F1 in epochs 16–20 beats the best of epochs 1–15 by at
least 0.005; the continuation is a second cosine cycle restarting at half the peak
learning rate.

When the rule says continue, the run **asks on the terminal first**, rings the terminal
bell and shows both scores and the current best epoch:

```
Validation macro-F1 is still rising after 20 epochs: best of the last 5 = 0.4520, best before = 0.4410.
Continuing trains 20 more epochs, to 40. best.pt so far is epoch 18 (val macro-F1 0.4520) and is kept either way. If there is no answer in 60 min: yes.
Continue? [y/n]:
```

Type `y` or `n` in the tmux window (`tmux attach -t train`). With no answer within
`extend_confirm_timeout_minutes` (60), or when no terminal is attached (e.g. `nohup`),
the rule's "continue" stands, so an unattended run still finishes; set the timeout to
`null` to wait indefinitely. The question is asked only after the epoch is saved, so
Ctrl+C at the prompt loses nothing, and `--resume` asks it again. A "stop" is never
asked about. The decision, and how it was made (`yes`, `no`, `no answer (timeout)`,
`not asked (no terminal)`), is logged, stored in `last.pt` and written to
`finetune_report_<run>.json` as `extend_decision`. `best.pt` is always the epoch with
the highest validation macro-F1. Settings: `training.extend_decision_epoch`,
`extend_window`, `extend_min_delta`, `restart_lr_factor`, `extend_confirm`,
`extend_confirm_timeout_minutes`; set `extend_decision_epoch: null` for a single cosine
over all epochs.

### What changed for full scale, and why

| Change | Failure it prevents |
|---|---|
| `DAISEE_ROOT`, `MRG_CACHE_DIR`, `MRG_CHECKPOINT_DIR` environment overrides | editing a tracked config on every machine |
| Stage 1 writes are atomic (temp file + rename) | a process killed mid-write leaves a truncated `.npz` that later runs treat as done, crashing training hours later |
| `run_preprocessing.py --shard I/N` | one CPU core doing ~95 minutes of MediaPipe while the rest sit idle |
| `dataset.missing_clip_policy: skip`, capped by `max_missing_fraction: 0.02` | one undecodable video aborting the whole run — while still stopping if preprocessing is merely incomplete |
| `run_finetune.py --resume` from `last.pt` (optimiser, scheduler, AMP scaler, RNG, early-stopping state) | an interruption at epoch 40 costing 40 epochs |
| Significance tested against the **majority-class** accuracy, via `scipy.stats.binom` | the old exact sum raised `OverflowError` at n ≈ 1,784 — after training had finished; and "beats 25 %" is meaningless when one class is ~50 % |
| `class_weighting: effective_number` | pure inverse frequency weighting the ~30 Very Low clips ~77× above High |
| Calibration must be fitted, to its own file | silently training with generic blur/face-size bounds, or with the 120-clip development calibration |
| Figure titles and reference lines taken from the run | full-release plots labelled "DAiSEE_mini, 120 train / 80 val" with a 25 % chance line |

Each is covered by `tests/test_full_scale.py`, and the preprocessing, calibration guard,
incomplete-cache guard, training, resume and evaluation paths were run end to end on the
216-clip subset with `config_full.yaml` before release.

### Running it

```bash
git clone https://github.com/presidentv/MAMBA---MIN-PROJECT.git
cd MAMBA---MIN-PROJECT/MRG-ViT-Mamba
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install -r requirements.txt
# optional, Linux + nvcc: the fused Mamba kernel, picked up automatically
# pip install mamba-ssm --no-build-isolation

export DAISEE_ROOT=/path/to/DAiSEE           # contains DataSet/ and Labels/
bash scripts/run_full_daisee.sh
```

The script runs, in order: environment check → fetch MediaPipe models → dataset audit
(stops on subject leakage) → Stage 1 across all cores → MRS calibration on the train
split → GPU memory probe → fine-tuning with validation-based selection → evaluation →
figures. It is safe to re-run after an interruption; delete `checkpoints/full_ft32/` for a
fresh start. Outputs: `artifacts/finetune_report_full_ft32.json`,
`artifacts/report_full_ft32/`, `logs/training_history_full_ft32.csv`.

On Windows, `scripts/run_full_daisee.ps1` runs the same steps, including Stage 1
sharded across cores, and is equally safe to re-run after an interruption:

```powershell
$env:DAISEE_ROOT = "D:\DAiSEE"
powershell -ExecutionPolicy Bypass -File scripts\run_full_daisee.ps1
```

Or run the steps individually:

```powershell
$env:DAISEE_ROOT = "D:\DAiSEE"
python scripts/fetch_models.py
python scripts/audit_dataset.py --config configs/config_full.yaml
python scripts/run_preprocessing.py --config configs/config_full.yaml --stage 1
python scripts/fit_mrs_stats.py --config configs/config_full.yaml
python scripts/run_finetune.py --config configs/config_full.yaml --run-name full_ft32 --workers 4 --resume
python scripts/make_finetune_figures.py --run full_ft32 --baseline=
```

(`--baseline=` rather than `--baseline ""`: Windows PowerShell 5.1 drops an empty-string
argument, leaving `--baseline` with no value.)

**If training stops** (crash, power cut, reboot, Ctrl+C), run the same command again.
`--resume` continues from `checkpoints/full_ft32/last.pt`, which is rewritten after
**every** epoch, so at most the epoch in progress is lost. The optimiser, learning-rate
schedule, AMP scaler, random-number state, early-stopping counters and the epoch-20
decision all carry over, and the learning-rate curve is identical to an uninterrupted
run.

Every 10 epochs a snapshot is also kept: `after_epoch_010.pt`, `after_epoch_020.pt`, …
Each holds the same full resume state as `last.pt`, plus everything needed to load it
for evaluation. To go back to one, for example if `last.pt` is lost or you want to
retrain from an earlier point:

```powershell
python scripts/run_finetune.py --config configs/config_full.yaml --run-name full_ft32 --workers 4 --resume-from checkpoints/full_ft32/after_epoch_020.pt
```

Training continues from epoch 21 and the history CSV is rolled back to match. A
`best.pt` written after the snapshot belongs to the abandoned continuation, so it is
renamed `best_abandoned_epoch_NNN.pt` and best-model selection restarts from the snapshot.

### What the run writes after training

`run_full_daisee.sh` / `.ps1` finish with two steps that read the saved results. If
either fails, the trained model and the metrics are unaffected, and the step can be
re-run on its own.

**Graphs** (`scripts/make_finetune_figures.py` → `artifacts/report_full_ft32/`):
training curves (`ft1`), learned MRS weights per epoch (`ft2`), per-class
precision/recall/F1 (`ft3`), the row-normalised test confusion matrix (`ft6`),
one-vs-rest ROC and precision-recall curves (`ft7`), confidence when right vs wrong
with a calibration curve (`ft8`), and the learning-rate schedule with the selected epoch
and the epoch-20 decision (`ft9`). The unnormalised confusion matrices and
classification reports are in `artifacts/` as before.

**Worked examples** (`scripts/explain_finetuned.py` → `artifacts/examples_full_ft32/`):
three test clips from three different people, chosen automatically: the most confident
correct prediction, the most confident mistake, and the rarest remaining class. For each:

* `example<k>_<clip>_landmarks.png`: a real frame with the MediaPipe face markers,
  detector box and head pose, the five MRS components on that frame, and the actual and
  predicted engagement level;
* `example<k>_<clip>_decision.png`: how the prediction was reached. (1) Grad-CAM, showing
  where the ViT looked on the face crop; (2) the per-frame reliability and the pooling
  weight it gave each frame; (3) the learned MRS weights; (4) how much of the
  predicted-class score came from the ViT → Mamba branch and how much from the landmark
  branch (integrated gradients); (5) the named landmark features that pushed the score
  up or down relative to the average training clip; (6) the class probabilities;
* `examples_overview.png`: the three frames side by side, with actual and predicted labels;
* `examples.json`: every number in the figures, plus a one-paragraph explanation of each
  prediction.

To choose the clips yourself:
`python scripts/explain_finetuned.py --config configs/config_full.yaml --run full_ft32 --clips <id> <id> <id>`.
Attribution shows what this model relied on. It does not show what causes engagement.

**`artifacts/examples_*/` is git-ignored and must not be shared.** The images show
identifiable DAiSEE participants, whom the dataset licence forbids redistributing,
and `examples.json` pairs clip ids with their labels.

### Reading the results

The full release is severely imbalanced — roughly 1 % Very Low, 4 % Low, and the rest
split between High and Very High. Always predicting the most common class already scores
around 50 % accuracy. So:

* compare accuracy to `baselines.majority_class.accuracy`, not to 25 %;
* lead with **macro-F1**, which that trivial predictor cannot inflate;
* treat Very Low per-class numbers with caution — the test split holds only a handful.

## References

* DAiSEE — https://people.iith.ac.in/vineethnb/resources/daisee/ · [arXiv:1609.01885](https://arxiv.org/abs/1609.01885)
* Mamba — [state-spaces/mamba](https://github.com/state-spaces/mamba) · [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
* Vision Mamba (Vim) — [hustvl/Vim](https://github.com/hustvl/Vim) · [arXiv:2401.09417](https://arxiv.org/abs/2401.09417)
* MC-MIFA (literature reference values only) — https://pmc.ncbi.nlm.nih.gov/articles/PMC13328615/
* Class-balanced loss — Cui et al., CVPR 2019
