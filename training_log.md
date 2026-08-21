# Training log — engagement classifier smoke test

Run date: 20 August 2026. Data: `Datasets/DAiSEE_Small`, 108 clips / 16 subjects.
Code: `pipeline/` (Phases 1–2), `src/` (Steps 1–7).
Re-run instructions: `outputs/README.md`.

---

## What was built

| Step | Module | Status |
|---|---|---|
| Phase 1 | `pipeline/phase1_preprocessing.py` | run, 108/108 clips |
| Phase 2 | `pipeline/phase2_landmarks.py` | run, 5,384 frames |
| 1. Gaze features | `src/features.py` | run, 21 per-frame + 104 clip-level |
| 2. Affect branch | `src/affect.py` | run, 8 probs + 1280-d embedding/frame |
| 3. Dataset | `src/datasets.py` | run |
| 4. Fusion model | `src/models.py` | run |
| 5. Baselines | `src/baselines.py` | run, all 5 |
| 6. Train/eval | `src/train.py` | run, results written |
| 7a. Early detection | `src/early_detection.py` | **stub only**, by request |
| 7b. XAI | `src/explain.py` | **stub only**, by request |

---

## Substitutions made, and why

**OpenFace 2.0 action units → HSEmotion (frozen EfficientNet-B0).**
OpenFace is not a pip package — it needs a compiled binary plus separate model
downloads — so it was unavailable. HSEmotion was the pre-agreed fallback. It
gives an 8-class AffectNet emotion distribution *and* the 1280-d penultimate
embedding; both are written per frame and `data.affect_feature_set` selects
which the model consumes (default `probs`, because 1280 dimensions over 36
training clips would be pure memorisation). Consequence for the writeup: the
affect channel is now **categorical expression**, not **action-unit intensity**,
so AU-level interpretability claims no longer apply.

**AU45 blink → EAR-based blink detection.**
No OpenFace means no AU45. Blinks are detected from the eye aspect ratio
computed off the MediaPipe eye contours (Soukupova & Cech 2016), with a
**per-clip adaptive threshold** (75% of that clip's own median EAR) rather than
a global constant, because EAR baseline varies strongly with face shape,
eyewear and camera angle. Implemented in `src/features.py`, flagged in a comment
at the call site.

**MediaPipe `solutions` → MediaPipe Tasks API.**
Not a plan change, an environment constraint: MediaPipe ≥ 1.0 removed the legacy
`solutions` module from every Python 3.13 wheel. Tasks is a superset and also
supplies a facial transformation matrix, which gives better head pose than
solvePnP on six keypoints.

**Two environment fixes worth remembering:**
* PyTorch ≥ 2.6 defaults `torch.load(weights_only=True)` and refuses the pickled
  HSEmotion checkpoint. Scoped context manager in `src/affect.py`.
* `timm` must be **< 1.0** (`timm==0.9.16` pinned). Under timm ≥ 1.0 the
  checkpoint loads fine and then fails at *inference* with
  `AttributeError: DepthwiseSeparableConv.conv_s2d`.

---

## Smoke test results (test split)

| model | macro-F1 | accuracy | QWK | F1 c0 | F1 c1 | F1 c2 | F1 c3 |
|---|---|---|---|---|---|---|---|
| logreg_meanpool | **0.349** | 0.306 | 0.201 | 0.571 | 0.250 | 0.240 | 0.333 |
| gaze_only | 0.320 | 0.417 | 0.152 | 0.333 | 0.000 | 0.533 | 0.414 |
| cross_attention_fusion | 0.181 | 0.167 | 0.135 | 0.333 | 0.222 | 0.000 | 0.167 |
| majority | 0.173 | **0.528** | 0.000 | 0.000 | 0.000 | 0.691 | 0.000 |
| late_fusion | 0.126 | 0.167 | 0.100 | 0.273 | 0.000 | 0.000 | 0.231 |
| affect_only | 0.087 | 0.111 | 0.011 | 0.182 | 0.000 | 0.000 | 0.167 |

**These numbers are not results.** 36 training clips, 4 classes, 3–5 clips in
the minority test classes. `train.py` prints an explicit small-sample warning
for every class under 5 clips before it trains anything.

### What the run does legitimately establish

1. **The pipeline is correct end-to-end.** Video → aligned frames → landmarks →
   two feature streams → padded masked sequences → fusion → metrics → JSON/CSV.
   All shapes line up; no silent NaNs; split integrity asserted, not assumed.
2. **The accuracy trap is real and immediate.** `majority` posts the *highest*
   accuracy of all six models (0.528) with the *second-lowest* macro-F1 (0.173).
   Reporting bare accuracy here would be actively misleading. This is exactly
   why macro-F1 + confusion matrix is the headline everywhere.
3. **The transformers overfit hard, as predicted.** Train macro-F1 reaches
   0.41–0.64 while validation sits at 0.15–0.21 and validation loss climbs
   monotonically from epoch ~3. Early stopping fires at epochs 20–23.
4. **Logistic regression beats both transformers.** This is the control working
   as intended — at n=36 the 267k–667k-parameter models have nothing to learn
   from. It confirms the mitigation already agreed in `memory.md`: frozen
   encoders + a light fusion head, with logistic regression reported beside
   every result.
5. **Late fusion underperforms both its own branches** (0.126 vs 0.320 / 0.087).
   Averaging a decent branch with a near-random one drags the good one down.
   Not a bug — expected when one modality is uninformative at this scale.

### Known weak spots in the features

* `fixation_ratio` is ~1.0 for nearly every clip. At 5 fps the inter-sample
  interval is 200 ms while a real saccade lasts 30–80 ms, so the I-VT threshold
  almost never fires. **What this feature actually measures is the rate of gaze
  shifts between samples, not a saccade rate** — it must not be described as the
  latter in the writeup. Raising `--sample-fps` in Phase 1 narrows the gap.
* `off_screen_ratio` is 0.0 for every clip: subjects genuinely look at the
  screen throughout, so the feature has no variance here. It may become
  informative on full DAiSEE.
* Both are documented at their definition site in `src/features.py`.

---

## Findings worth keeping

**DAiSEE splits are subject-disjoint — verified, not assumed.**
16 subjects, none in more than one split (Train 4, Validation 4, Test 8).
`datasets.verify_split_integrity` **raises** on violation rather than warning,
since leakage would invalidate every number.

**This sample is not representative of full DAiSEE's imbalance.**
Engagement counts per split are roughly 4/6/16/10 across classes 0–3. Full
DAiSEE has ~1% in class 0. Expect macro-F1 to fall at full scale, and expect
class weighting to matter much more.

**ClipID encodes a within-subject ordering.** `ClipID = <6-digit SubjectID><suffix>`,
suffix 4 digits (2 of 108 have 3, reading as a stripped leading zero). Left-pad
to 4: first digit = session (observed 0/1/2), last three = clip index within
session, strictly increasing. This gives the ordering the early-detection
experiment needs. **Caveat: inferred from the ID pattern, not documented by the
DAiSEE authors**, and indices are not dense (subject 110001 jumps 1012 → 1040 →
1048), so only pairs with an index gap of exactly 1 can be treated as adjacent.
Written up in full in `src/early_detection.py`.

---

## Next steps

1. Get the full DAiSEE and re-run — commands in `outputs/README.md`. Everything
   scales unchanged; only `--input-root` differs.
2. Re-tune `off_screen_*` thresholds and consider `--sample-fps 10` once real
   variance exists.
3. Implement `src/early_detection.py` (signatures fixed, ordering solved).
4. Implement `src/explain.py` — the deletion test is the one that keeps SHAP and
   attention rollout honest; run it against a random-ablation control.
5. Ablate `--loss ce` against CORAL once n is large enough for the comparison to
   mean anything.
