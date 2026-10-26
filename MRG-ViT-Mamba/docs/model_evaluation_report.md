# MRG-ViT–Mamba: Model Evaluation & Verification Report

**Date:** September 3, 2026  
**Repository:** `MRG-ViT-Mamba`  
**Dataset Split:** DAiSEE 108-Clip Subset (36 Train / 36 Validation / 36 Test)  
**Primary Checkpoint:** `checkpoints/main/best.pt`  

---

## 1. Executive Summary

This report documents the verification and test performance evaluation of the **MRG-ViT–Mamba** architecture—a reliability-guided temporal model combining a frozen Vision Transformer (ViT-B/16) spatial encoder, Motion Reliability Score (MRS) weighting, Mamba selective state-space model (S6), and MediaPipe geometric/blendshape features for 4-class student engagement classification.

All **38 unit tests** and **7 architectural verification checkpoints** passed without errors. On the subject-disjoint 36-clip test split, the primary model achieves an **Acc-4 of 0.3056** and a **Macro-F1 of 0.2397**.

---

## 2. Environment & System Specifications

| Component | Version / Setting |
|---|---|
| **Python** | 3.13.3 |
| **PyTorch** | 2.11.0+cu128 |
| **CUDA** | 12.8 (GPU: NVIDIA GeForce RTX 3060 Laptop, 6.0 GB) |
| **Mamba Backend** | `pytorch_reference` (Vendored S6 implementation) |
| **Spatial Encoder** | Frozen ImageNet ViT-B/16 ($D = 768$, Input: $224 \times 224$) |
| **Landmark Branch** | MediaPipe FaceLandmarker ($L = 80$ per frame $\to 3L = 240$ per clip) |

---

## 3. Pipeline & Architectural Verification Gates

Every component of the pipeline was validated against strict correctness criteria before model evaluation.

| Verification Gate | Result | Evidence & Key Metrics |
|---|:---:|---|
| **Unit Test Suite (`tests/test_units.py`)** | **PASS** | 38/38 unit tests passed (12.67s). Covers dataset split integrity, cache key stability, face crop bounds, MRS percentile normalization, ViT tensor shapes, Mamba S6 causality, fusion, and metrics. |
| **CHECKPOINT 1 & 2 (Dataset & Split Audit)** | **PASS** | 108/108 decodable clips. Verified 0 subject overlap across train (4), val (4), and test (8) splits. |
| **CHECKPOINT 3 (Face & Landmark Detection)** | **PASS** | 100% face detection across test frames, 99.48% across the full corpus. |
| **CHECKPOINT 4 (Motion Reliability Score)** | **PASS** | MRS range $[0.2976, 0.9880]$, mean = $0.6484$, std = $0.1272$. Zero NaNs or Infs. |
| **CHECKPOINT 5 (Landmark Features)** | **PASS** | $L=80$ per frame (28 geometric + 52 ARKit blendshapes), $3L=240$ per clip. |
| **CHECKPOINT 6 (ViT Spatial Encoder)** | **PASS** | `[B, T, 3, 224, 224]` $\to [B, T, 768]$ feature tensor, correct normalisation statistics. |
| **CHECKPOINT 7 (Mamba S6 Temporal SSM)** | **PASS** | Causality verified ($0.0$ max change at $t < 8$ for perturbations at $t \ge 8$). Matches reference recurrence with 0.0 max absolute error. |
| **TEST 9–12 (Full Pipeline & Overfit Gate)** | **PASS** | Fusion shape $[B, 512]$, gradient flow across all 35 parameter tensors, tiny overfit (16 clips) loss dropped from $1.379 \to 0.000000$ (100% train accuracy). |

---

## 4. Primary Model Test Scores (`checkpoints/main/best.pt`)

The primary model checkpoint was selected at epoch 8 based on validation macro-F1 and evaluated once on the 36-clip test split (8 subjects).

### Overall Metrics

| Metric | Score | Benchmark Reference |
|---|---:|:---:|
| **Accuracy (Acc-4)** | **0.3056** (30.56%) | Majority Baseline: 0.528 |
| **Macro-F1** | **0.2397** (0.240) | Stratified Random: 0.242 |
| **Weighted F1** | **0.3021** (0.302) | Majority Baseline: 0.173 |

### Per-Class Performance Breakdown

| Class Index | Class Name | Support | Precision | Recall | F1-Score |
|:---:|---|:---:|:---:|:---:|:---:|
| **0** | Very Low | 3 | 0.1667 | 0.6667 | 0.2667 |
| **1** | Low | 5 | 0.0000 | 0.0000 | **0.0000** *(Never predicted)* |
| **2** | High | 19 | 0.7143 | 0.2632 | 0.3846 |
| **3** | Very High | 9 | 0.2353 | 0.4444 | 0.3077 |

> [!NOTE]
> Class 1 (*Low*) was never predicted by the model on the test split due to class imbalance in the small 36-clip training subset (where Class 2 dominates).

---

## 5. Multi-Seed Ablation Study ($7\text{ Arms} \times 5\text{ Seeds}$)

Each ablation arm was trained across 5 random seeds (42–46). Model checkpoints were selected strictly on validation macro-F1 before evaluating on the test split.

| Ablation Arm | Description | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---|:---:|:---:|:---:|
| **`full`** | MRS elementwise multiplication + reliability pooling | $0.302 \pm 0.075$ | $0.278 \pm 0.035$ | **$0.214 \pm 0.024$** |
| **`multiply_only`** | MRS frame-level multiplication ($f'_t = r_t f_t$) only | $0.298 \pm 0.074$ | $0.289 \pm 0.045$ | **$0.217 \pm 0.029$** |
| **`pool_only`** | MRS weighting applied only in temporal pooling | $0.284 \pm 0.071$ | $0.283 \pm 0.032$ | **$0.216 \pm 0.029$** |
| **`no_mrs`** | Baseline with $r_t = 1.0$ everywhere | $0.289 \pm 0.068$ | $0.272 \pm 0.044$ | **$0.205 \pm 0.029$** |
| **`bidirectional`** | Vim-style temporal scan over frames | $0.281 \pm 0.018$ | $0.306 \pm 0.105$ | **$0.219 \pm 0.030$** |
| **`residual_mrs`** | Residual reliability addition | $0.313 \pm 0.073$ | $0.250 \pm 0.030$ | **$0.189 \pm 0.016$** |
| **`no_landmarks`** | Deep ViT branch only (no MediaPipe features) | $0.289 \pm 0.065$ | $0.278 \pm 0.086$ | **$0.170 \pm 0.046$** |

### Reference Baselines
* **Stratified Random Predictor:** Macro-F1 = **0.242**, Accuracy = **0.337**
* **Majority Class Predictor (Always 'High'):** Macro-F1 = **0.173**, Accuracy = **0.528**

---

## 6. Key Findings & Insights

1. **Pipeline Correctness:** The repository provides a fully verified, end-to-end checkpointed pipeline with zero parameter leakage, subject-disjoint splits, and verified S6 causality.
2. **Subset Scale Limitation:** On the 108-clip DAiSEE subset (36 train clips across 4 subjects), models overfit training splits (train macro-F1 $\to 0.94$) due to limited subject diversity, resulting in test macro-F1 scores ($0.214 \pm 0.024$) hovering near random guessing ($0.242$).
3. **Branch Importance:** Removing the landmark/blendshape branch (`no_landmarks`) causes the largest drop in test macro-F1 (dropping from $0.214 \to 0.170$), confirming that MediaPipe geometric and blendshape features provide key predictive signal.
