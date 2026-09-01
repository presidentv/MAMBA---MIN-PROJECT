# MRG-VM v2 — results, negative results, and the one number worth reporting

Everything below comes from the 108-clip DAiSEE sample. It supersedes the
single-split figures in `training_log.md`, which this work showed are not
trustworthy.

---

## The headline

**Leave-one-subject-out cross-validation, all 108 clips, 16 folds, 3 seeds:**

| model | features | macro-F1 | std | accuracy | QWK |
|---|---|---|---|---|---|
| **random forest** | 96 | **0.296** | **0.024** | 0.441 | 0.036 |
| extra trees | 96 | 0.292 | 0.003 | 0.355 | 0.079 |
| logistic regression | 20 | 0.272 | 0.027 | 0.312 | 0.063 |
| random forest | 20 | 0.257 | 0.007 | 0.358 | 0.044 |
| ordinal ridge | 96 | 0.242 | 0.000 | 0.315 | 0.072 |

Majority-class baseline is ≈0.17. So the result is real but modest, and the
tight standard deviations mean it is a *measurement* rather than an observation.

**Report 0.296 ± 0.024.** Not 0.494, not 0.376. Why those two numbers are not
reportable is the substance of this document.

---

## Why the fixed split cannot be used

Over 5 seeds on the official 36/36/36 split, validation and test rankings came
out essentially **anti-correlated**:

| model / features | validation | test |
|---|---|---|
| ordinal ridge / 10 | **0.385 (1st)** | 0.238 |
| soft vote / 10 | 0.362 (2nd) | 0.282 |
| random forest / 10 | 0.311 (4th) | 0.324 |
| random forest / 96 | 0.278 (5th) | **0.494 (1st)** |

Selecting on validation — the only honest criterion — picks the configuration
that lands near the bottom on test. The best test score belongs to a
configuration validation ranks fifth. Quoting 0.494 would be selecting on the
test set.

With 36 clips per partition this is not fixable by choosing a better model. It
is a property of the split size, and it invalidates every single-split
comparison in this project, including the ones reported earlier. Leave-one-
subject-out replaces one arbitrary partition with sixteen, pooling held-out
predictions across all 108 clips. Folds are subject-wise, never clip-wise:
clips from the same person are not independent, and a clip-wise split would let
the model memorise a face.

That departs from the official DAiSEE split, so it answers "how good is this
model class on this data" rather than "what is the benchmark score". Both are
reported; neither replaces the other.

---

## Negative results

Five things were tried on the expectation that they would help. Four did not.
They are recorded because a negative result that cost real compute is worth more
than a silent deletion, and because each was a specific, testable prediction
that turned out wrong.

### 1. Corruption augmentation — made it much worse

Reliability during training is 0.93 ± 0.049, so the conditioner has no varying
signal to learn from. Augmenting the training split with corrupted variants
(reliability recomputed per variant) was meant to fix that. It worked
mechanically — reliability std rose 0.05 → 0.205 — and performance collapsed:

| run | test macro-F1 | train | best epoch |
|---|---|---|---|
| v2, no augmentation | 0.363 | 0.639 | 14 |
| v2 + 2 variants, severity 1–5 | 0.176 | 0.270 | 3 |
| v3 + 1 variant, severity 1–3 | 0.170 | 0.224 | 4 |

The cause is a trade-off with no escape at this sample size. At severity 4–5 the
face mesh fails on 15–27% of frames, and a clip that is a quarter unobservable
no longer carries its label in any recoverable way — those variants are label
noise. At severity 1–3 the label survives but reliability std falls back to
0.079, barely above the 0.05 the augmentation existed to escape. Either way it
adds harder inputs and **zero new label information**, since a corrupted clip
shares its original's label.

### 2. Weight decay was not the reason the reliability weights stayed uniform

The five learned reliability weights never moved off 0.2 (spread 0.0055). The
hypothesis was that AdamW weight decay was pulling the logits toward zero, which
*is* the uniform softmax. Gradient magnitude reaching them measured 6.3% of a
typical weight's, so signal starvation was ruled out and decay looked like the
culprit.

Excluding them from weight decay made the spread **smaller** (0.0055 → 0.0023).
The hypothesis was wrong. The gradient evidently points in inconsistent
directions across batches and averages to nothing: the loss cannot discriminate
between reliability weightings at this sample size.

The parameter-group fix was kept anyway — not decaying parameters whose zero
point is a meaningful prior is correct practice regardless.

### 3. Affect features — hurt

Adding the HSEmotion clip-level features (+17 columns) to the ensemble:

| model | without affect | with affect |
|---|---|---|
| random forest | 0.505 | 0.366 |
| logistic regression | 0.400 | 0.228 |
| soft vote | 0.417 | 0.323 |

Frozen pretrained features were predicted to be the cheapest available win,
because they bring knowledge learned from hundreds of thousands of faces rather
than from 36 clips. At 36 training clips, 17 extra columns cost more in variance
than they contribute in signal.

### 4. CORAL ordinal loss — slightly worse than cross-entropy

`v2_ce_loss` beat the ordinal head on both splits (test 0.376 vs 0.363,
validation 0.351 vs 0.293). Engagement *is* ordered, so CORAL should be the right
choice in principle; at this sample size it is not. Small enough to be noise, but
the direction is against the prediction.

### 5. Feature selection — no help for the tree models

Reducing to the top 10/20/40 features by mutual information hurt random forest on
test (0.494 → 0.324 at k=10). Tree ensembles already perform implicit feature
selection, so external selection only removes information.

---

## What did work

**Cross-attention fusion.** The clearest architectural result in the ablation:
replacing it with v1 clip-level gating costs **−0.134**. This is about
sequence-level alignment between the visual and landmark streams, and has nothing
to do with the reliability score.

**Shallow models over deep ones.** Random forest beats the 1.26 M-parameter
network under LOSO. At n=36 the deep model is the wrong capacity, and no amount
of architecture fixes that.

**The conditioner as a whole, with a caveat.** Removing it entirely costs
−0.210 — the largest single effect in the ablation. But removing FiLM, the
learned weights, or dt scaling *individually* changes nothing at all (all three
score exactly 0.363/0.293/0.195, identical to the full model). The module
matters; no part of it does. The most likely explanation is that its ~14k
parameters act as a reparameterisation or regulariser rather than as reliability
conditioning, and the three sub-ablations directly contradict any claim that the
reliability signal is doing the work.

---

## Status of the novelty

The Motion Reliability Score has now been tested four ways and has not been shown
to help:

| test | outcome |
|---|---|
| frame gating (v1) | never evaluable — gate rejected 1 frame in 5,403 |
| fixed dt guidance (v1) | zero prediction flips |
| learned multi-factor conditioning (v2) | weights stayed uniform, sub-ablations null |
| corruption robustness | guided ≈ blind, delta 0.000 |

The corruption protocol in `mrgvm/robustness.py` is sound and took two iterations
to become so — the first kept the original MRS (meaningless), the second left the
geometric stream uncorrupted (null by construction, because SHAP attributes only
4.5% to appearance). The corrected version degrades both streams and shows real
damage: macro-F1 0.363 → 0.091 under severe blur, with the mesh failing on 15% of
frames.

It still shows no benefit from guidance, and the reason is now well established
rather than guessed: **a model cannot learn to use a signal that was constant
during training, and it cannot be taught to by augmentation without destroying
the labels.**

This is a defensible position to present. The mechanism is correctly implemented,
verified by tests, and honestly evaluated; the corpus cannot demonstrate it. The
experiment that could is the same protocol run on a model trained where
reliability genuinely varies — which needs the full 9,068-clip corpus, or a
corpus with naturally degraded video.

---

## Reproducing

```bash
.venv/Scripts/python.exe src/ensemble.py --output-root outputs_ungated --no-affect --loso --top-k 0 20 --seeds 1 2 3
```

```bash
.venv/Scripts/python.exe -m mrgvm.train_mrgvm --output-root outputs_ungated --config mrgvm/configs/v2.json
```

```bash
.venv/Scripts/python.exe -m mrgvm.robustness --output-root outputs_ungated --checkpoint outputs_ungated/checkpoints/mrgvm_v2.pt
```

```bash
.venv/Scripts/python.exe -m pytest tests -q
```
