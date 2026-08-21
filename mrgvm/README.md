# MRG-VM — Phases 3–7

Implementation of the Motion Reliability Guided Vision Mamba framework described
in the project PDF. Phases 1–2 live in [`../pipeline/`](../pipeline) and must be
run first — this package consumes their outputs.

| PDF phase | Module | Deliverable |
|---|---|---|
| 3 — MRG-VM behavioural feature learning | `mamba.py`, `vision_mamba.py` | `outputs/embeddings/mrgvm_embeddings.parquet` → `mamba_*` columns |
| 4 — Adaptive feature fusion | `geometric.py`, `fusion.py`, `model.py` | same file → `fused_*` columns |
| 5 — MLP engagement classifier | `fusion.MLPClassifier`, `train_mrgvm.py` | `outputs/checkpoints/mrgvm.pt` |
| 6 — Evaluations + ablation studies | `ablation.py` | `outputs/results_mrgvm/ablation_results.{csv,json}` |
| 7 — Explainable AI (SHAP) | `shap_explain.py` | `outputs/results_mrgvm/shap_report.json` + PNGs |

---

## Why the Mamba is hand-written

The reference `mamba-ssm` package ships a fused CUDA kernel and will not build
without `nvcc`. This machine is CPU-only, so `mamba.py` implements the S6
selective-scan recurrence directly in PyTorch:

```
h_t = exp(dt_t · A) ⊙ h_{t-1} + (dt_t · B_t) · x_t
y_t = ⟨C_t, h_t⟩ + D · x_t
```

with `dt`, `B` and `C` produced from the input at every step — the "selective"
part. It is the same algorithm as Gu & Dao (2023), just without the fused
kernel: the scan is a Python loop over the sequence axis, and the batch
dimension supplies the parallelism. On this project's sequence lengths (49
spatial patches, 50 temporal frames) that costs ~0.5 s per batch forward.

Blocks are **bidirectional** (Vision Mamba / Vim). Plain Mamba is causal, which
is the wrong inductive bias for a patch grid or a short clip where t+1 is as
informative as t−1.

`exp(dt·A)` is `(batch, length, d_inner, d_state)` — hundreds of MB if
materialised for a whole sequence — so it is recomputed per timestep inside the
scan and never held for the full sequence.

---

## Where "motion reliability guided" actually lives

Phase 1 already discards frames below the MRS threshold, but survivors are not
equally trustworthy: MRS 0.52 and MRS 0.95 both pass the gate. Two mechanisms
carry that residual reliability into the model, and both are independently
ablatable:

**1. `dt` modulation** (`vision_mamba.guide_delta`). In a state-space model `dt`
controls how far the hidden state moves at each step. Scaling `dt` by a function
of MRS makes a low-reliability frame *literally update the state less*, without
masking it out or breaking the sequence. This is the natural place to inject a
per-timestep confidence into an SSM — a transformer has no clean equivalent, only
attention-bias hacks.

**2. Reliability-weighted pooling** (`vision_mamba.guide_pooling`). The temporal
pool is an MRS-weighted mean rather than a flat one.

The default mapping is `scale = min_delta_scale + (1 − min_delta_scale) · mrs`,
so a perfect frame is unmodified and the worst survivor is damped to
`min_delta_scale` (0.25) rather than silenced — zeroing it would make the block
ignore real motion, which is not what "guided" should mean.

**Phase 6 found that mechanism 1 does nothing on this data**, and it is worth
being blunt about why. Once Phase 1 has gated, retained-frame MRS is
0.93 ± 0.049, so the linear map yields a multiplier of 0.95 ± 0.037 — under 4%
relative spread. A near-constant rescale of `dt` is precisely what the learned
`dt_proj` bias absorbs, so toggling `guide_delta` moved the clip embedding by
~7e-5 relative and flipped zero predictions. Confirmed with matched weights and
a genuinely non-zero output delta, so this is a parameterisation problem, not a
wiring fault.

`delta_map='clip_normalised'` standardises MRS *within each clip* first, so `dt`
responds to **relative** reliability rather than a near-constant absolute score.
Measured 8.3× stronger on the same batch. Not the default, so the published
ablation stays reproducible — but it is the first thing to try at full scale.

---

## Architecture

```
frames (T,3,112,112) ─ PatchEmbed 16×16 ─→ (T,49,128)
                     + spatial position
                     ─ bidirectional MambaEncoder ×2 (spatial)
                     ─ spatial mean pool ─────────→ (T,128)
                     + temporal position
                     ─ bidirectional MambaEncoder ×2 (temporal, dt ← MRS)
                     ─ MRS-weighted pool ─────────→ (128,)   Phase 3 embedding
                                                       │
landmarks ─ geometric.py ─→ (T,32) ─ mean/std/min/max ─→ (128,)
                                                       │
                     AdaptiveFeatureFusion (gated) ────→ (256,)  Phase 4
                                                       │
                     MLPClassifier 128→64 ─────────────→ 4 logits  Phase 5
```

**Adaptive fusion** projects both streams to a common width, computes a
per-channel gate from their concatenation, and interpolates:
`mixed = g·mamba + (1−g)·geometric`. The interaction term `m⊙g` is concatenated
alongside so information is not lost when the gate saturates. The gate is
returned with every prediction, so Phase 7 can report which stream drove it.
`ConcatFusion` is the matched-capacity ablation control.

**Geometric stream** = 12 landmark descriptors computed here (face width/height,
aspect ratio, mouth openness, brow–eye distances, asymmetries, whole-mesh
deformation energy and velocity — all normalised by inter-ocular distance so they
are scale-invariant) + 20 gaze/blink/head descriptors reused from
`src/features.py`. Total 32 per frame, ×4 pooling statistics = 128.

---

## Running it

Phases 1–2 must have been run into `outputs/` first.

**Phase 5 — train**

```bash
.venv/Scripts/python.exe -m mrgvm.train_mrgvm --output-root outputs --config mrgvm/configs/default.json
```

**Phases 3–4 — export the deliverable embeddings** (after training; an untrained
backbone emits noise)

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

Useful overrides: `--epochs`, `--batch-size`, `--image-size`, `--seed`,
`--variants full no_landmarks`, `--device cuda`.

---

## Ablation design (Phase 6)

Each row removes exactly one component from the full model, sharing one seed,
one data cache and one schedule, so deltas are attributable:

| variant | what it removes | test macro-F1 |
|---|---|---|
| `full` | nothing — the reference | 0.286 |
| `no_mrs_guidance` | both guidance mechanisms | 0.261 |
| `guide_delta_only` | pooling guidance | 0.261 |
| `guide_pooling_only` | dt guidance | 0.286 |
| `no_vision_mamba` | the appearance stream (geometry only) | **0.407** |
| `no_landmarks` | the geometric stream (Vision Mamba only) | 0.052 |
| `concat_fusion` | the gate (plain concat, matched capacity) | 0.287 |

**What this measured.** Removing the Vision Mamba branch *improves* the model
(+0.121) while removing the landmark stream collapses it (−0.234) — though
`no_vision_mamba` also has the worst *validation* score, so the gap is
noise-dominated at n=36 and the honest claim is "the deep branch has no data to
learn from yet".

The `guide_*` rows pair up exactly: dt guidance changes nothing, pooling
guidance is worth +0.025. Verified as a real effect and not a wiring fault — see
`vision_mamba.reliability_to_delta_scale`.

**What it cannot measure.** The frame **gate**. By Phase 6 the sub-threshold
frames are already gone; undoing that needs a Phase 1 re-run at
`--mrs-threshold 0`, and on this sample the gate rejected 1 frame out of 5,403,
so there is nothing to recover.

---

## Phase 7 — what is explained

SHAP runs over the **named** landmark/gaze features, not the raw Mamba
embedding: a Shapley value for "mamba dimension 87" explains nothing actionable,
whereas "blink rate" does. The appearance stream still appears, as a handful of
PCA components, so its total contribution is comparable on the same axis.
Attributions are folded into the four behavioural categories the PDF names for
Phase 3 — eye gaze, blink patterns, head movement, facial dynamics — plus
appearance.

**Caveat, reported not buried:** attributions are computed over a RandomForest
surrogate fitted to MRG-VM's own predictions, because KernelSHAP against the
torch model would need thousands of forward passes through the scan. The
surrogate's fidelity is measured and written to `shap_report.json` as
`surrogate_fidelity`; below ~0.8 the attributions describe the surrogate rather
than MRG-VM, and the script warns.
