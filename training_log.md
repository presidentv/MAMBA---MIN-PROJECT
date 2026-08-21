# Training log — MRG-VM, all seven phases

Data: `Datasets/DAiSEE_Small`, 108 clips / 16 subjects / 3 subject-disjoint splits.
Code: `pipeline/` (Phases 1–2), `mrgvm/` (Phases 3–7), `src/` (transformer baselines).
Re-run instructions: [`outputs/README.md`](outputs/README.md) and [`mrgvm/README.md`](mrgvm/README.md).

---

## Status of every phase

| Phase | Deliverable | Status |
|---|---|---|
| 1 — Preprocessing + MRS | reliable frames + per-frame MRS table | **run** — 108/108 clips, 5,403 sampled, 5,402 retained, mean MRS 0.855 |
| 2 — Landmark extraction | landmark coordinates | **run** — 5,384 frames landmarked, 18 mesh failures in 3 Test clips |
| 3 — MRG-VM | deep behavioural embeddings | **run** — 128-d per clip, `outputs/embeddings/` |
| 4 — Adaptive feature fusion | optimised feature representation | **run** — 256-d fused vector + 128-d gate |
| 5 — MLP classifier | trained model | **run** — `outputs/checkpoints/mrgvm.pt`, 1.17 M params |
| 6 — Evaluations + ablation | ablation table | **run** — 7 variants, `outputs/results_mrgvm/ablation_results.csv` |
| 7 — Explainable AI (SHAP) | explainable predictions | **run** — `shap_report.json` + 2 PNGs |

---

## Headline results (test split, 36 clips)

| model | macro-F1 | accuracy | QWK |
|---|---|---|---|
| logreg on mean-pooled features | **0.349** | 0.306 | 0.201 |
| gaze-only transformer | 0.320 | 0.417 | 0.152 |
| **MRG-VM (full)** | 0.286 | 0.472 | 0.170 |
| cross-attention transformer fusion | 0.181 | 0.167 | 0.135 |
| majority class | 0.173 | **0.528** | 0.000 |
| naive late fusion | 0.126 | 0.167 | 0.100 |
| affect-only transformer | 0.087 | 0.111 | 0.011 |

**These are not results.** 36 training clips, 4 ordinal classes, 3–5 clips in the
minority test classes. Every training script prints a small-sample warning for
each class under five clips *before* it trains.

What the runs legitimately establish:

1. **All seven phases execute end-to-end on real data.** Shapes line up, no
   silent NaNs, gradients finite, split integrity asserted rather than assumed.
2. **The accuracy trap is real.** The majority-class predictor posts the highest
   accuracy of any model (0.528) and the second-lowest macro-F1 (0.173).
   Reporting bare accuracy here would be actively misleading.
3. **MRG-VM beats both transformer variants** (0.286 vs 0.181 / 0.087) and the
   majority floor, and loses to logistic regression on 29 hand-engineered
   features (0.349). At n=36 that ordering is exactly what you would expect: a
   1.17 M-parameter backbone has nothing to constrain it. MRG-VM train macro-F1
   reaches 0.570 against 0.392 validation.
4. **The Mamba implementation is correct.** Gradients are finite, and the
   bidirectional scan was confirmed by perturbing timestep 15 and observing the
   change propagate back to timestep 2. The reliability guidance is wired
   correctly too — but Phase 6 showed that only one of its two mechanisms
   actually affects predictions. See the Phase 6 section, finding 2.

---

## Phase 6 — ablation study

Seven variants, each removing exactly one component, sharing one seed, one data
cache and one training schedule. Full table:
`outputs/results_mrgvm/ablation_results.csv`.

| variant | test macro-F1 | Δ vs full | val macro-F1 | params |
|---|---|---|---|---|
| `no_vision_mamba` (geometry only) | **0.407** | **+0.121** | 0.236 | 42k |
| `concat_fusion` | 0.287 | +0.001 | 0.342 | 1.12M |
| `full` | 0.286 | — | 0.392 | 1.17M |
| `guide_pooling_only` | 0.286 | +0.000 | 0.392 | 1.17M |
| `no_mrs_guidance` | 0.261 | −0.025 | 0.389 | 1.17M |
| `guide_delta_only` | 0.261 | −0.025 | 0.389 | 1.17M |
| `no_landmarks` (Mamba only) | 0.052 | −0.234 | 0.194 | 1.09M |

### Three findings, in order of how much they should change the writeup

**1. The Vision Mamba branch earns nothing here — removing it *helps*.**
Dropping 1.1 M parameters of learned appearance raises test macro-F1 from 0.286
to 0.407, while dropping the 32 hand-engineered landmark features collapses the
model to 0.052. Three independent lines of evidence agree: the ablation, SHAP
putting appearance at 4.5%, and the fusion gate sitting at 0.49 (range
0.35–0.63, i.e. barely committing either way).

**Do not over-read it.** `no_vision_mamba` has the *worst* validation macro-F1
(0.236) and the *best* test (0.407). Early stopping selects on validation, so
this configuration would never have been chosen by the training loop. That
val/test inversion is direct evidence the gaps are noise-dominated at n=36. The
defensible claim is "the deep branch has no data to learn from yet", not
"geometry beats Vision Mamba by 0.12".

**2. Of the two reliability-guidance mechanisms, only pooling does anything.**
`full` == `guide_pooling_only` and `no_mrs_guidance` == `guide_delta_only`, to
four decimals. Toggling `guide_delta` moves the clip embedding by ~7e-5 relative
and flips zero predictions.

This was checked rather than assumed: with matched weights the outputs *do*
differ, so the mechanism is wired correctly. The cause is the data. Once Phase 1
has gated, retained-frame MRS is 0.93 ± 0.049, so `scale = 0.25 + 0.75·MRS`
gives 0.95 ± 0.037 — a rescale with under 4% relative spread, which is exactly
what the learned `dt_proj` bias absorbs.

So the novel contribution, as parameterised, reduces to **MRS-weighted pooling**,
worth +0.025 test macro-F1. `delta_map='clip_normalised'` (standardise MRS within
each clip, so dt responds to *relative* reliability) measures 8.3× stronger and
is implemented but not default, so this table stays reproducible. It is the first
thing to try at full scale.

**3. Adaptive fusion is indistinguishable from plain concatenation**
(0.287 vs 0.286). At this sample size the gate has nothing to learn.

### Two limitations of the ablation itself

- **The frame gate is not ablated by any row here.** By Phase 6, Phase 1 has
  already discarded sub-threshold frames; undoing that needs a Phase 1 re-run at
  `--mrs-threshold 0`. On this sample that experiment is vacuous regardless —
  **the gate rejected 1 frame out of 5,403**, so there is nothing for it to have
  changed. The gate only becomes measurable on data containing genuinely bad
  frames.
- **Single seed.** At n=36, a macro-F1 gap under roughly ±0.10 is
  indistinguishable from seed noise, which covers every row except
  `no_landmarks`. Multiple seeds per variant is a one-line change (`--seed`) and
  is the prerequisite for reading this table as evidence.

An earlier version of this study also contained a redundant row: `no_mrs`
(setting every frame to MRS = 1.0) is *provably* the same experiment as
`no_mrs_guidance`, since mrs = 1 makes the dt map the identity and the weighted
pool a uniform mean. It produced identical metrics and has been removed; the
reasoning is recorded in `mrgvm/ablation.py` so it is not reintroduced.

## Phase 7 — SHAP

Behavioural group contributions (test-time, surrogate fidelity **1.000**):

| group | share of total \|SHAP\| |
|---|---|
| facial dynamics | 41.1% |
| head movement | 23.2% |
| eye gaze | 19.3% |
| blink patterns | 11.9% |
| appearance (Vision Mamba) | 4.5% |

Top individual features: `yaw_mean` (4.8%), `gaze_yaw_mean` (4.5%),
`geo_nose_chin_dist_mean` (3.2%), `geo_brow_eye_right_mean` (3.0%).

Two things to note honestly:

- The Vision Mamba appearance stream contributes only **4.5%**, consistent with
  the fusion gate sitting near 0.49 (range 0.35–0.63) — the model is leaning on
  hand-engineered geometry, not on learned appearance. That is the expected
  outcome at this sample size and is itself an argument for the geometric
  branch, not against the architecture.
- SHAP is computed over a RandomForest surrogate fitted to MRG-VM's own
  predictions, because KernelSHAP against the torch model would need thousands
  of forward passes through the selective scan. Fidelity is measured and
  reported (`surrogate_fidelity` in `shap_report.json`); at 1.000 the surrogate
  reproduces the model's predictions exactly on this sample, so the attributions
  are trustworthy *here*. On a larger set fidelity will drop and must be
  re-checked — the script warns below 0.8.

---

## Substitutions and environment constraints

| Planned | Used | Why |
|---|---|---|
| `mamba-ssm` (official Vision Mamba) | hand-written S6 in `mrgvm/mamba.py` | `mamba-ssm` ships a fused CUDA kernel and will not build without `nvcc`; this is a CPU-only machine. Same algorithm, no fused kernel — the scan is a Python loop over the sequence axis with the batch dimension supplying parallelism. |
| OpenFace 2.0 action units | HSEmotion frozen EfficientNet-B0 | OpenFace is not a pip package. **Consequence:** the affect channel is *categorical expression*, not *AU intensity* — AU-level interpretability claims no longer apply. |
| AU45 blink | eye-aspect-ratio blink detection | Follows from the above. Per-clip adaptive threshold, since EAR baseline varies with face shape and eyewear. |
| MediaPipe `solutions` | MediaPipe Tasks API | `solutions` was removed from every Python 3.13 wheel at mediapipe ≥ 1.0. Tasks is a superset and supplies a facial transformation matrix, giving better head pose than solvePnP on six keypoints. |

Two pins that will bite if changed: **`timm` must be < 1.0** (HSEmotion
checkpoints were pickled against timm 0.x and fail at *inference*, not load,
under ≥ 1.0), and PyTorch ≥ 2.6 needs a scoped `weights_only=False` to load them.

---

## Known weak spots

- **`fixation_ratio` is ≈1.0 for nearly every clip.** At 5 fps the sampling
  interval is 200 ms while a saccade lasts 30–80 ms, so the I-VT threshold
  rarely fires. This feature measures the *rate of gaze shifts between samples*,
  not a saccade rate, and must not be described as the latter in the writeup.
- **`off_screen_ratio` is 0.0 across the sample** — subjects genuinely look at
  the screen throughout, so the feature has no variance here.
- **Single seed everywhere.** Needed for the ablation table to mean anything.
- Both feature issues are documented at their definition sites in
  `src/features.py`.

---

## Findings worth keeping

**DAiSEE splits are subject-disjoint — verified, not assumed.** 16 subjects
(Train 4, Validation 4, Test 8), none in more than one split. Both dataset
loaders **raise** on violation rather than warning.

**This sample is not representative of full DAiSEE's imbalance.** Engagement
counts run roughly 4/6/16/10 across classes 0–3 per split. Full DAiSEE has ~1%
in class 0, so expect macro-F1 to fall and class weighting to matter far more.

**ClipID encodes a within-subject ordering.**
`ClipID = <6-digit SubjectID><suffix>`; left-pad the suffix to 4 digits, first
digit is the session (observed 0/1/2), last three a strictly-increasing
within-session index. *Inferred from the ID pattern, not documented by the DAiSEE
authors*, and indices are not dense (subject 110001 jumps 1012 → 1040 → 1048), so
only gap-of-1 pairs are safely adjacent. Written up in `src/early_detection.py`.

---

## Next steps

1. Obtain the full DAiSEE and re-run — commands in `outputs/README.md`.
   Everything scales unchanged; only `--input-root` differs.
2. Re-run the ablation with 3–5 seeds per variant so the deltas exceed noise.
3. If a CUDA machine becomes available, swap `mrgvm/mamba.py` for `mamba-ssm`
   and raise `image_size` to 224 — the interfaces already match.
4. Implement `src/early_detection.py` (signatures fixed, ordering solved).
5. Ablate the affect branch into MRG-VM as a third fusion stream.
