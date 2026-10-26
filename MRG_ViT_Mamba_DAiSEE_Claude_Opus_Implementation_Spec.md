# MRG-ViT--Mamba on DAiSEE --- Complete Implementation Specification

## 0. Purpose

This document is the implementation contract for building the project
end-to-end with Claude Opus 4.5.

The goal is to prevent accidental architecture changes, unsupported
assumptions, data leakage, silent preprocessing failures, and fabricated
results.

**Non-negotiable rule:** do not invent dataset paths, label mappings,
tensor dimensions, model APIs, package versions, or experimental
results. Inspect the local environment and dataset first, record what is
actually found, and stop with a clear diagnostic if a required
assumption is false.

## 1. Final target architecture

``` text
DAiSEE video
    ↓
Official train / validation / test split
    ↓
Temporal sampling
    ↓
Face detection / crop / alignment
    ↓
Motion Reliability Score (MRS)
    ↓
ViT frame-level spatial features
    ↓
MRS reliability weighting
    ↓
Mamba temporal modeling
    ↓
Deep temporal embedding
                +
MediaPipe geometric / behavioral features
    ↓
Normalize each branch separately
    ↓
Projection to common dimension
    ↓
Feature fusion
    ↓
MLP
    ↓
4-class engagement prediction
    ↓
SHAP analysis
```

The central research question is:

> Can reliability-guided ViT--Mamba temporal modeling improve robust
> student engagement recognition from webcam video while preserving
> temporal information and avoiding unnecessary influence from
> unreliable observations?

Do not frame the project as "beating the highest published DAiSEE
accuracy."

------------------------------------------------------------------------

# 2. Source-of-truth information

## 2.1 Original project proposal

The original proposal defines:

-   Phase 1: video preprocessing and Motion Reliability Score using
    blur, face visibility, head rotation, eye visibility, and motion
    consistency.
-   Phase 2: MediaPipe Face Mesh and face/eye/iris/head-pose/landmark
    tracking.
-   Phase 3: Vision Mamba behavioral feature learning.
-   Phase 4: feature fusion with landmark-based geometric features.
-   Phase 5: MLP engagement classifier.
-   Phase 6: SHAP explainability.

For the updated implementation, Phase 1 must **not** simply discard
every low-quality frame. MRS should primarily be used as a reliability
signal/weight while preserving temporal positions.

## 2.2 DAiSEE

The official DAiSEE page states that DAiSEE contains 9,068 video
snippets from 112 users and covers boredom, confusion, engagement, and
frustration. Each affective state has four intensity levels: very low,
low, high, and very high.

For this project, use **only the engagement label**.

The official download is about 15 GB.

Official resources:

-   DAiSEE: https://people.iith.ac.in/vineethnb/resources/daisee/
-   Original paper: https://arxiv.org/abs/1609.01885

A commonly reported version describes 10-second clips at 30 FPS. **Do
not hard-code this.** Inspect the actual local files because
copies/preprocessing versions can differ.

## 2.3 Related MC-MIFA paper

A recent MC-MIFA paper reports on DAiSEE:

-   Vision Mamba (Vim-S): 76.10 accuracy, F1 0.724
-   VMamba (Tiny): 82.60 accuracy, F1 0.801
-   MC-MIFA: 97.45 accuracy, F1 0.968

These are literature reference values, **not target values and not
expected results**.

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC13328615/

Do not claim that Mamba has never been used on DAiSEE.

------------------------------------------------------------------------

# 3. Critical terminology: ViT + Mamba vs Vim

## 3.1 Recommended architecture

The recommended architecture is:

``` text
Frame
  ↓
ViT
  ↓
frame embedding
  ↓
sequence of frame embeddings
  ↓
Mamba temporal module
  ↓
temporal embedding
```

Call this:

-   ViT--Mamba
-   ViT + Mamba temporal modeling
-   ViT spatial encoder + Mamba temporal encoder

## 3.2 Vision Mamba / Vim

Vision Mamba (Vim) is itself a visual backbone based on bidirectional
Mamba blocks for image token sequences. It is **not automatically
equivalent** to applying Mamba to a sequence of video-frame embeddings.

Official Vim implementation: https://github.com/hustvl/Vim

Therefore:

**Do not call the temporal module "Vim" unless the actual Vision Mamba
implementation is used.**

If the research goal is temporal video modeling, the cleanest
implementation is:

``` text
ViT per frame → Mamba across frames
```

If a true Vim backbone is used, document exactly where it is used and
what its input is.

------------------------------------------------------------------------

# 4. Dataset inspection --- mandatory first step

Before implementing the model:

1.  Locate the DAiSEE root directory.
2.  List its directories.
3.  Find the official train/validation/test split information.
4.  Locate label metadata.
5.  Inspect at least 10 video paths.
6.  Decode at least 3 videos.
7.  Record:
    -   FPS
    -   frame count
    -   width
    -   height
    -   duration
    -   decoding success.

Create:

``` text
artifacts/dataset_audit.json
```

with:

``` text
dataset_root
split_source
num_train
num_val
num_test
num_unique_subjects
fps_distribution
frame_count_distribution
resolution_distribution
label_distribution
missing_files
corrupt_files
```

## CHECKPOINT 1 --- DATASET AUDIT

Do not continue until:

-   all split definitions can be parsed;
-   referenced videos can be resolved;
-   label mapping is known from actual metadata;
-   missing/corrupt files are understood;
-   four engagement classes are verified.

If any condition fails, stop and report the exact problem.

------------------------------------------------------------------------

# 5. Labels

The conceptual classes are:

``` text
0 = Very Low
1 = Low
2 = High
3 = Very High
```

However, **verify the actual encoded values in the local metadata before
training**.

Do not assume that a numeric label means a specific class without
inspecting the dataset documentation/metadata.

Create:

``` text
artifacts/label_mapping.json
```

after verification.

------------------------------------------------------------------------

# 6. Train/validation/test leakage

Do not randomly split clips if the same student can occur in multiple
splits.

The preferred condition is:

``` text
train_subjects ∩ val_subjects  = empty
train_subjects ∩ test_subjects = empty
val_subjects   ∩ test_subjects  = empty
```

Preserve the official DAiSEE split if it is subject-separated.

## CHECKPOINT 2 --- SPLIT AUDIT

Print:

``` text
Train subjects: ...
Validation subjects: ...
Test subjects: ...

Train/Val overlap: 0
Train/Test overlap: 0
Val/Test overlap: 0
```

If overlap is non-zero, stop.

------------------------------------------------------------------------

# 7. Class imbalance and metrics

DAiSEE engagement is strongly imbalanced.

Do not report only accuracy.

Required:

-   Accuracy / Acc-4
-   Weighted F1
-   Macro-F1
-   per-class precision
-   per-class recall
-   per-class F1
-   confusion matrix

## Accuracy / Acc-4

For four-class classification:

\[ Acc ext{-}4 = rac{ ext{correct predictions}}{ ext{all predictions}}
\]

## Weighted F1

\[ F1\_{weighted} = `\sum`{=tex}\_c rac{n_c}{N}F1_c \]

Each class is weighted by its number of samples.

## Macro-F1

\[ F1\_{macro} = rac{1}{4}`\sum`{=tex}\_{c=1}\^{4}F1_c \]

Every class receives equal importance.

Interpretation:

-   Accuracy: overall correctness.
-   Weighted F1: overall F1 under the observed class distribution.
-   Macro-F1: whether performance is balanced across the four classes.

**Macro-F1 should be emphasized because majority-class performance can
hide poor minority-class recognition.**

------------------------------------------------------------------------

# 8. Temporal sampling

Do not assume that 16 frames is optimal.

The implementation must make frame count configurable:

``` text
8
16
32
64
```

Start with 16 for pipeline validation. If GPU memory permits, use 32 as
the first serious temporal setting.

Uniform temporal sampling:

\[ i_k = round`\left`{=tex}(rac{k(N-1)}{T-1}ight) \]

where:

-   N = number of frames in the actual video
-   T = requested sampled frames
-   k = 0,...,T-1

Do not use random sampling for the core reproducible pipeline.

Sampling is computational subsampling, not an assertion that the
unsampled frames are useless.

Short events such as blinks can be missed, which is why frame-count
evaluation is important.

------------------------------------------------------------------------

# 9. Face processing

For each sampled frame:

``` text
frame
 ↓
face detector
 ↓
face bounding box
 ↓
face crop
 ↓
alignment if used
 ↓
resize
 ↓
normalization
 ↓
ViT
```

The actual detector must be selected only after checking package
availability and compatibility.

Record:

``` text
detector
version
confidence threshold
crop padding
input resolution
```

Do not silently replace a failed detector.

## CHECKPOINT 3 --- FACE PIPELINE

Run face extraction on a small sample.

Verify:

-   faces are detected where expected;
-   crops are non-empty;
-   coordinates are within image boundaries;
-   crop dimensions are valid;
-   missing-face frames have an explicit handling policy.

Save:

``` text
artifacts/face_crop_preview.png
```

Do not continue if crops are visibly wrong.

------------------------------------------------------------------------

# 10. Motion Reliability Score (MRS)

MRS is an **observation reliability score**, not an engagement score.

For frame t:

\[ MRS_t = w_bB_t+ w_fF_t+ w_hH_t+ w_eE_t+ w_mM_t \]

where each component is normalized to \[0,1\]:

-   B = blur reliability
-   F = face visibility reliability
-   H = head-pose reliability
-   E = eye visibility reliability
-   M = motion consistency reliability

Initial baseline:

\[ w_b=w_f=w_h=w_e=w_m=0.2 \]

Do not claim these weights are optimal.

## 10.1 Blur

Use variance of Laplacian on the face crop:

``` python
gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
lap = cv2.Laplacian(gray, cv2.CV_64F)
blur_raw = lap.var()
```

Convert the raw value to a \[0,1\] reliability score using thresholds or
percentiles derived from the **training data only**.

## 10.2 Face visibility

Use detector confidence if available and face-area information.

A tiny detected face should not receive the same reliability as a clear
sufficiently large face.

## 10.3 Head-pose reliability

Estimate yaw, pitch and roll using landmarks.

Important:

**Large head rotation does not mean low engagement.**

It means facial appearance information may be less reliable.

Never convert head pose directly into an engagement label.

## 10.4 Eye visibility

Use eye/iris landmark visibility or successful localization.

Do not treat closed eyes automatically as unreliable. A blink is a valid
behavior.

## 10.5 Motion consistency

Use lightweight consecutive-frame information.

Preferred initial method:

-   landmark displacement when MediaPipe landmarks are available;
-   otherwise a lightweight image similarity method.

Do not use raw pixel difference as the only measure without checking
sensitivity to lighting/camera movement.

------------------------------------------------------------------------

# 11. Critical MRS decision: do not delete frames

The original proposal says to discard low-quality frames. For the
ViT--Mamba version, **do not automatically delete all low-MRS frames**.

Deleting frames can damage temporal structure.

Instead:

\[ f'\_t = r_t f_t \]

where:

-   (f_t) = ViT feature for frame t
-   (r_t) = MRS reliability in \[0,1\]
-   (f'\_t) = reliability-weighted feature

Then:

``` text
[f'_1, f'_2, ..., f'_T]
        ↓
      Mamba
```

Temporal positions remain intact.

This is the main reliability-guided mechanism.

## MRS implementation

Create:

``` text
src/mrs.py
```

with a function conceptually equivalent to:

``` python
compute_mrs(frame, previous_frame, face_data, landmark_data)
```

Return:

``` python
{
    "mrs": ...,
    "blur": ...,
    "face_visibility": ...,
    "head_pose": ...,
    "eye_visibility": ...,
    "motion_consistency": ...
}
```

## CHECKPOINT 4 --- MRS SANITY

For at least 100 sampled frames:

-   every component is in \[0,1\];
-   MRS is in \[0,1\];
-   no NaN;
-   no Inf;
-   inspect min/mean/max;
-   generate histograms;
-   manually inspect high-MRS and low-MRS examples.

If nearly all frames have the same MRS, investigate.

------------------------------------------------------------------------

# 12. MediaPipe feature branch

Use MediaPipe Face Mesh/Face Landmarker.

Extract:

-   face landmarks
-   eye landmarks
-   iris landmarks
-   head pose
-   temporal landmark tracking

Do not blindly concatenate all raw coordinates.

Construct normalized behavioral/geometric features such as:

## Eye

-   eye aspect ratio / eye openness
-   relative iris position
-   left/right eye geometry

## Head

-   yaw
-   pitch
-   roll

## Face

-   normalized landmark distances
-   mouth geometry
-   eyebrow geometry if reliable

## Temporal geometry

For consecutive frames:

\[ `\Delta `{=tex}p_t=p_t-p\_{t-1} \]

This captures movement.

Normalize geometry relative to face scale so absolute pixel size does
not dominate.

## CHECKPOINT 5 --- LANDMARK SANITY

Verify:

-   landmarks exist where expected;
-   coordinate ranges are valid;
-   no NaN/Inf;
-   fixed feature dimension;
-   explicit handling for missing landmarks;
-   feature statistics saved.

Save:

``` text
artifacts/landmark_feature_stats.json
```

------------------------------------------------------------------------

# 13. ViT frame encoder

ViT is responsible for **spatial information within each frame**.

``` text
face crop
 ↓
ViT
 ↓
frame embedding f_t
```

The actual output dimension must be discovered from the selected model.

Do not hard-code 768 unless the selected model actually outputs 768.

Recommended order:

1.  Start with pretrained ViT.
2.  Freeze ViT.
3.  Extract embeddings.
4.  Verify tensor shapes.
5.  Cache embeddings.
6.  Train temporal/fusion components.
7.  Fine-tune ViT only after the pipeline works.

## Efficient batching

Input:

``` text
[B,T,C,H,W]
```

reshape:

``` text
[B*T,C,H,W]
```

run ViT:

``` text
[B*T,D]
```

reshape:

``` text
[B,T,D]
```

This is preferable to a Python loop over every frame.

## CHECKPOINT 6 --- VIT SHAPES

Print and verify:

``` text
input shape
flattened shape
actual ViT output dimension D
reshaped shape
device
dtype
```

Expected final feature shape:

``` text
[B,T,D]
```

------------------------------------------------------------------------

# 14. MRS-weighted ViT sequence

Given:

``` text
features = [B,T,D]
mrs      = [B,T]
```

do:

``` python
mrs = mrs.unsqueeze(-1)
weighted_features = features * mrs
```

Expected:

``` text
weighted_features = [B,T,D]
```

The number and order of temporal positions must not change.

------------------------------------------------------------------------

# 15. Mamba temporal module

Mamba receives:

``` text
[B,T,D]
```

and models temporal dependencies.

Conceptually:

``` text
f1 → f2 → f3 → ... → fT
                 ↓
               Mamba
                 ↓
          temporal embedding
```

Use the official Mamba implementation where possible:

https://github.com/state-spaces/mamba

The installed package/API must be inspected before writing constructors.
**Do not invent constructor arguments.**

If the environment cannot install or import Mamba:

``` text
STOP
```

Report the exact error.

Do not silently replace Mamba with LSTM/Transformer and still call the
model Mamba.

## Starting configuration

Use a small temporal model initially, for example:

``` text
Mamba input dimension = actual ViT feature dimension
Mamba model dimension = 256
Mamba blocks = 1–2
```

These are starting values, not proven optimal values.

If ViT output is 768:

``` text
ViT 768
 ↓
Linear projection
 ↓
Mamba 256
```

## CHECKPOINT 7 --- MAMBA

Pass a synthetic tensor:

``` text
[B,T,D]
```

through the actual Mamba implementation.

Verify:

-   import works;
-   forward pass works;
-   output shape is correct;
-   output contains no NaN/Inf.

------------------------------------------------------------------------

# 16. Deep temporal embedding

If Mamba outputs:

``` text
[B,T,256]
```

create one representation per video.

Simple initial method:

\[ D=rac{1}{T}`\sum`{=tex}\_{t=1}\^{T}H_t \]

giving:

``` text
[B,256]
```

Do not silently use the last timestep without justification.

------------------------------------------------------------------------

# 17. Landmark temporal representation

MediaPipe produces:

``` text
[B,T,L]
```

where L is the actual landmark-feature dimension.

Do not assume L=128.

A simple initial temporal aggregation is:

``` text
mean
std
mean absolute temporal difference
```

which can produce:

``` text
[L] + [L] + [L] = [3L]
```

per video.

The exact feature dimension must be printed and recorded.

------------------------------------------------------------------------

# 18. Normalization and projection

Normalization and projection are separate.

## Deep branch

``` text
Mamba embedding
 ↓
normalization
 ↓
Linear projection
 ↓
256-D
```

## Landmark branch

``` text
landmark representation
 ↓
normalization
 ↓
Linear projection
 ↓
256-D
```

A linear layer:

\[ y=Wx+b \]

does not guarantee normalized values.

Projection changes dimensionality/representation. Normalization controls
feature scale/distribution.

Fit any dataset-dependent scaler using **training data only**.

------------------------------------------------------------------------

# 19. Feature fusion

Start with simple concatenation.

Let:

\[ D_p`\in `{=tex}R\^{256} \]

and:

\[ L_p`\in `{=tex}R\^{256} \]

Then:

\[ F=\[D_p\|\|L_p\] \]

giving:

``` text
[B,512]
```

Then:

``` text
512
 ↓
MLP
 ↓
4 logits
```

This is the primary fusion implementation.

## Adaptive fusion

If an adaptive gate is used, document the exact formula.

Conceptually:

\[ lpha=sigmoid(g(D_p,L_p)) \]

then combine the branches.

Do not claim adaptive fusion is better until actual experiments show it.

------------------------------------------------------------------------

# 20. MLP classifier

Starting architecture:

``` text
512
 ↓
256
 ↓
GELU/ReLU
 ↓
Dropout
 ↓
128
 ↓
GELU/ReLU
 ↓
4
```

The final layer produces raw logits.

Use cross-entropy loss.

For imbalance, support class-weighted cross entropy. Calculate class
weights from the **training split only**.

------------------------------------------------------------------------

# 21. Complete model forward pass

The model should conceptually execute:

``` text
Video
 ↓
T sampled frames
 ↓
Face crops
 ↓
MediaPipe features
 ↓
MRS
 ↓
ViT
 ↓
Frame embeddings
 ↓
MRS weighting
 ↓
Mamba
 ↓
Deep temporal embedding
      +
Landmark temporal embedding
 ↓
Separate normalization
 ↓
Separate projection
 ↓
Fusion
 ↓
MLP
 ↓
4 logits
```

------------------------------------------------------------------------

# 22. Recommended repository

``` text
project/
│
├── configs/
│   └── config.yaml
│
├── data/
│   └── README.md
│
├── src/
│   ├── dataset.py
│   ├── video_sampling.py
│   ├── face_processing.py
│   ├── mrs.py
│   ├── landmarks.py
│   ├── vit_encoder.py
│   ├── temporal_mamba.py
│   ├── fusion.py
│   ├── classifier.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── scripts/
│   ├── audit_dataset.py
│   ├── test_video_reader.py
│   ├── test_face_pipeline.py
│   ├── test_mrs.py
│   ├── test_landmarks.py
│   ├── test_vit.py
│   ├── test_mamba.py
│   └── test_full_pipeline.py
│
├── artifacts/
├── checkpoints/
├── logs/
└── README.md
```

------------------------------------------------------------------------

# 23. Configuration

Do not hard-code settings throughout the code.

Example:

``` yaml
seed: 42

dataset:
  root: "/PATH/TO/DAISEE"
  num_classes: 4

video:
  num_frames: 16
  sampling: "uniform"

face:
  image_size: 224

mrs:
  blur_weight: 0.2
  face_weight: 0.2
  head_weight: 0.2
  eye_weight: 0.2
  motion_weight: 0.2
  weighting_mode: "multiply"

vit:
  pretrained: true
  freeze: true

mamba:
  d_model: 256
  layers: 2

fusion:
  projected_dim: 256
  mode: "concat"

training:
  batch_size: 4
  epochs: 20
  learning_rate: 0.0001
  mixed_precision: true

evaluation:
  metrics:
    - accuracy
    - weighted_f1
    - macro_f1
```

These values are starting points, not guaranteed optimal settings.

------------------------------------------------------------------------

# 24. Caching

Cache expensive preprocessing:

``` text
cache/
├── sampled_frames/
├── mrs/
├── landmarks/
└── vit_features/
```

Recommended workflow:

## Stage A

``` text
Video
 ↓
sampling
 ↓
face crop
 ↓
MRS
 ↓
MediaPipe
 ↓
ViT
 ↓
cache
```

## Stage B

``` text
cached ViT features
+
cached MRS
+
cached MediaPipe features
 ↓
Mamba
 ↓
fusion
 ↓
MLP
```

Cache identifiers must include enough information to prevent stale-cache
reuse:

-   video ID
-   split
-   preprocessing version
-   frame count
-   image size
-   model version

If preprocessing changes, invalidate/rebuild the affected cache.

------------------------------------------------------------------------

# 25. GPU and environment checks

First query the actual environment:

``` python
torch.cuda.is_available()
torch.cuda.get_device_name(0)
torch.cuda.get_device_properties(0).total_memory
```

Do not assume CUDA exists.

If an NVIDIA RTX A4000 is present, its memory can be used for planning,
but the code must detect the actual hardware.

Recommended order:

1.  CPU/single-batch validation.
2.  GPU single-batch validation.
3.  Frozen ViT.
4.  Mixed precision if supported.
5.  Cache ViT features.
6.  Fine-tune ViT only later.

The official Mamba repository currently documents a CUDA/Linux-oriented
installation path. Inspect the actual environment before selecting a
package/version: https://github.com/state-spaces/mamba

------------------------------------------------------------------------

# 26. Mandatory progressive testing

Do not launch full training immediately.

## TEST 1 --- Environment

Verify:

``` text
Python
PyTorch
CUDA
GPU
OpenCV
MediaPipe
selected ViT library
Mamba
```

Record versions.

## TEST 2 --- One video

Print:

``` text
FPS
frame count
resolution
duration
```

Save sampled frames.

## TEST 3 --- Face extraction

Save face-crop montage.

## TEST 4 --- MRS

Print all five components and final MRS.

## TEST 5 --- MediaPipe

Print landmark feature shape.

## TEST 6 --- ViT

Print input/output shapes.

## TEST 7 --- MRS weighting

Verify:

``` text
features.shape == weighted_features.shape
```

## TEST 8 --- Mamba

Run a synthetic `[B,T,D]` tensor.

## TEST 9 --- Fusion

Verify:

``` text
deep projected shape
landmark projected shape
fused shape
```

## TEST 10 --- Full forward pass

Run one real batch:

``` text
videos → logits
```

Verify:

``` text
logits.shape == [batch_size,4]
```

## TEST 11 --- One training step

Run:

``` text
forward
loss
backward
optimizer.step
```

Verify no NaN/Inf.

## TEST 12 --- Tiny overfit test

Train on 8--32 samples.

The model should substantially reduce training loss.

If it cannot overfit a tiny subset, do not start full training.

------------------------------------------------------------------------

# 27. Anti-hallucination rules

These rules are mandatory for the coding agent.

## Rule 1 --- Never invent paths

If DAiSEE is not found:

``` text
STOP
```

Ask for the dataset path.

## Rule 2 --- Never invent labels

Inspect the actual metadata.

## Rule 3 --- Never invent model APIs

Inspect installed packages and actual classes/functions.

## Rule 4 --- Never fabricate metrics

If training has not run:

``` text
Accuracy = NOT RUN
Macro-F1 = NOT RUN
```

Never generate plausible-looking numbers.

## Rule 5 --- Never hide errors

For failures, log:

``` text
video_id
stage
error
exception
```

If files are skipped, report:

``` text
total
successful
failed
failure percentage
```

## Rule 6 --- Never silently change architecture

If Mamba fails, stop.

Do not replace it with LSTM/Transformer without explicit approval.

## Rule 7 --- Never use test data for tuning

Test data is for final evaluation only.

## Rule 8 --- Never fit preprocessing statistics on test data

Training data only.

## Rule 9 --- Never use engagement labels to calculate MRS

MRS is based on observation quality/reliability.

## Rule 10 --- Never interpret MRS as engagement

MRS = reliability.

Engagement = target.

------------------------------------------------------------------------

# 28. Training sequence

## Stage 0

Dataset audit.

## Stage 1

Tiny preprocessing.

## Stage 2

Run all unit tests.

## Stage 3

Tiny overfit test.

## Stage 4

Freeze ViT and train:

``` text
ViT → MRS weighting → Mamba → MLP
```

## Stage 5

Add MediaPipe:

``` text
ViT → Mamba
       +
MediaPipe
       ↓
Fusion
       ↓
MLP
```

## Stage 6

Enable full MRS weighting.

## Stage 7

Full training.

## Stage 8

Final test evaluation.

## Stage 9

SHAP.

------------------------------------------------------------------------

# 29. Training logging

Every epoch:

``` text
epoch
train_loss
train_accuracy
train_macro_f1
train_weighted_f1
val_loss
val_accuracy
val_macro_f1
val_weighted_f1
learning_rate
GPU memory if available
```

Save:

``` text
logs/training_history.csv
```

Select the best checkpoint using validation Macro-F1 unless another
criterion is explicitly justified.

Never select a checkpoint using test performance.

------------------------------------------------------------------------

# 30. Final evaluation

Create:

``` text
artifacts/test_metrics.json
artifacts/confusion_matrix.png
artifacts/classification_report.txt
```

Include:

``` text
Accuracy / Acc-4
Weighted F1
Macro-F1
per-class precision
per-class recall
per-class F1
```

Also report:

``` text
number of test videos
number of unique test subjects
```

------------------------------------------------------------------------

# 31. Efficiency evaluation

Measure:

-   total parameters
-   trainable parameters
-   inference time per video
-   videos/second
-   GPU memory
-   number of sampled frames
-   input resolution

Timing must state:

-   hardware
-   batch size
-   warm-up policy
-   number of repetitions

Do not compare timings from different hardware/batch conditions without
stating them.

------------------------------------------------------------------------

# 32. SHAP

Apply SHAP to the final classifier/fusion representation.

Conceptually:

``` text
fused feature
 ↓
MLP
 ↓
SHAP
 ↓
feature contribution
```

Analyze groups such as:

``` text
deep temporal representation
eye features
iris/gaze features
head pose
facial geometry
```

SHAP is model attribution. It does not prove causation.

------------------------------------------------------------------------

# 33. Robustness experiment

If time permits, create controlled degraded validation/test copies
using:

-   blur
-   reduced resolution
-   synthetic occlusion
-   brightness/contrast changes

Do not modify the original test set.

Compare:

``` text
Baseline:
ViT → Mamba

vs

Proposed:
MRS → ViT → MRS weighting → Mamba
```

Report performance under each degradation.

Do not claim robustness merely because MRS exists.

------------------------------------------------------------------------

# 34. Temporal sampling experiment

Evaluate:

``` text
8
16
32
64
```

frames/video.

For each setting record:

``` text
frames
accuracy
weighted F1
macro F1
inference time
GPU memory
```

The goal is to identify the tradeoff between temporal information and
computation.

Do not claim that 16 frames is universally sufficient.

------------------------------------------------------------------------

# 35. Literature comparison rules

When comparing published results:

-   use the exact metric name;
-   use the exact dataset split if known;
-   use the exact preprocessing if reported;
-   state when protocols are not directly comparable.

Never write:

> "Our model is better than MC-MIFA"

unless the same evaluation protocol has actually been reproduced.

Never use literature values as if they were generated by this project.

------------------------------------------------------------------------

# 36. Expected limitations

Document:

-   DAiSEE is imbalanced.
-   Engagement labels are affective/intensity annotations and should not
    automatically be interpreted as objective internal mental states.
-   Some engagement levels are visually similar.
-   Temporal sampling can miss short behaviors.
-   MRS thresholds/weights can affect performance.
-   MediaPipe can fail under occlusion/extreme pose.
-   Deep models may exploit subject-specific appearance.
-   Published results can use different preprocessing/splits/metrics.

------------------------------------------------------------------------

# 37. Final acceptance checklist

## Dataset

-   [ ] DAiSEE path verified
-   [ ] labels verified
-   [ ] four engagement classes verified
-   [ ] train/validation/test splits verified
-   [ ] subject overlap checked
-   [ ] corrupt/missing files reported

## Preprocessing

-   [ ] uniform temporal sampling works
-   [ ] face detection works
-   [ ] face crops inspected
-   [ ] MRS values are in \[0,1\]
-   [ ] MRS components inspected
-   [ ] MediaPipe features have fixed dimensions
-   [ ] no NaN/Inf

## Model

-   [ ] ViT forward pass works
-   [ ] actual ViT embedding dimension recorded
-   [ ] MRS weighting preserves `[B,T,D]`
-   [ ] Mamba forward pass works
-   [ ] temporal pooling works
-   [ ] landmark branch works
-   [ ] normalization works
-   [ ] projections work
-   [ ] fusion dimension verified
-   [ ] MLP output is `[B,4]`

## Training

-   [ ] one training step works
-   [ ] tiny overfit test passes
-   [ ] no NaN/Inf
-   [ ] checkpoint saves
-   [ ] validation metrics log correctly
-   [ ] test set not used for tuning

## Evaluation

-   [ ] Acc-4
-   [ ] Weighted F1
-   [ ] Macro-F1
-   [ ] per-class metrics
-   [ ] confusion matrix
-   [ ] inference time
-   [ ] parameter count
-   [ ] GPU memory

## Explainability

-   [ ] SHAP runs on final classifier
-   [ ] feature contributions saved
-   [ ] interpretations are attribution, not causation

------------------------------------------------------------------------

# 38. Final implementation instruction

Build the project **incrementally**.

At every checkpoint:

1.  Run the test.
2.  Show the output.
3.  Compare output against the expected shape/range/condition.
4.  If the test fails, stop.
5.  Diagnose the failure.
6.  Fix it.
7.  Re-run the failed test.
8.  Only then proceed.

Never proceed through a failed checkpoint just to produce a
complete-looking repository.

Never fabricate a successful result.

Never fabricate a metric.

Never silently replace a requested component.

Never claim an experiment was performed unless an actual output file/log
exists.

The final README must contain:

-   architecture diagram
-   dataset setup
-   verified environment
-   preprocessing
-   MRS formula
-   ViT role
-   Mamba role
-   MediaPipe role
-   normalization/projection
-   fusion
-   training commands
-   evaluation commands
-   checkpoint outputs
-   actual results
-   limitations
-   exact configuration used for final results

------------------------------------------------------------------------

# 39. Reference resources

DAiSEE official: https://people.iith.ac.in/vineethnb/resources/daisee/

DAiSEE original paper: https://arxiv.org/abs/1609.01885

Vision Mamba official implementation: https://github.com/hustvl/Vim

Mamba official implementation: https://github.com/state-spaces/mamba

MC-MIFA reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC13328615/

Use these resources for verification. The coding agent must inspect the
actual local environment and installed APIs before writing
package-specific code.
