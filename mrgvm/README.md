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

The mapping is `scale = min_delta_scale + (1 − min_delta_scale) · mrs`, so a
perfect frame is unmodified and the worst survivor is damped to `min_delta_scale`
(0.25 by default) rather than silenced — zeroing it would make the block ignore
real motion, which is not what "guided" should mean.

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
`--variants full no_mrs`, `--device cuda`.

---

## Ablation design (Phase 6)

Each row removes exactly one component from the full model, sharing one seed,
one data cache and one schedule, so deltas are attributable:

| variant | what it removes |
|---|---|
| `full` | nothing — the reference |
| `no_mrs` | the score entirely (every retained frame set to MRS = 1.0) |
| `no_mrs_guidance` | guidance only — frames are still gated in Phase 1, but MRS no longer steers `dt` or pooling |
| `no_vision_mamba` | the appearance stream (landmark geometry only) |
| `no_landmarks` | the geometric stream (Vision Mamba only) |
| `concat_fusion` | the gate (plain concatenation at matched capacity) |

`no_mrs` versus `no_mrs_guidance` is the pair that matters: it separates
"filtering bad frames helped" from "letting reliability steer the SSM helped",
which are different claims and are usually conflated.

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
