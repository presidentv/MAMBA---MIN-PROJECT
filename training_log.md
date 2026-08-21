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
| 6 — Evaluations + ablation | ablation table | **run** — 6 variants, `outputs/results_mrgvm/ablation_results.csv` |
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
4. **The reliability guidance is live, not decorative.** Verified directly:
   changing `delta_scale` measurably changes the Mamba output, and the
   bidirectional scan was confirmed by perturbing timestep 15 and observing the
   change propagate back to timestep 2.

---

## Phase 6 — ablation study

See `outputs/results_mrgvm/ablation_results.csv` for the full table and
`ablation_results.json` for the per-component deltas against the full model.

Six variants, each removing exactly one component, sharing one seed, one data
cache and one training schedule:

| variant | removes |
|---|---|
| `full` | nothing — reference |
| `no_mrs` | the score entirely (all retained frames set to MRS = 1.0) |
| `no_mrs_guidance` | guidance only — Phase 1 still gates frames, but MRS no longer steers `dt` or pooling |
| `no_vision_mamba` | the appearance stream (landmark geometry only) |
| `no_landmarks` | the geometric stream (Vision Mamba only) |
| `concat_fusion` | the gate (plain concatenation, matched capacity) |

The `no_mrs` vs `no_mrs_guidance` pair is the one that matters: it separates
"filtering bad frames helped" from "letting reliability steer the SSM helped".
Those are different claims and are routinely conflated in papers that report a
quality-filtering step.

**Caveat that applies to the whole table:** at n=36 with a single seed, a
macro-F1 difference below roughly ±0.10 is indistinguishable from seed noise.
The ablation harness is correct and will produce a meaningful table on full
DAiSEE; the current numbers should be read as a demonstration that it runs, not
as evidence about any component's value. Multiple seeds per variant are the
obvious next step and are a one-line change (`--seed`).

---

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
