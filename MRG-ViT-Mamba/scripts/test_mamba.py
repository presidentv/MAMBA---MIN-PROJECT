"""TEST 8 + CHECKPOINT 7 - Mamba temporal module.

Because the Mamba here is a *vendored* reference implementation rather than the
pip package, shape checks alone are not enough evidence that it is correct. This
script also verifies the two properties that make it a selective SSM at all:

  1. **Causality.** Perturbing the input at time t must leave outputs at every
     time < t bit-identical. A bug in the scan, the causal convolution padding,
     or the flip logic would show up here and nowhere else.
  2. **Recurrence equivalence.** ``selective_scan_ref`` is checked against an
     independently written naive loop over the S6 recurrence
     x_t = exp(dt_t A) x_{t-1} + dt_t B_t u_t,  y_t = C_t x_t + D u_t.

Also runs gradient flow, dtype/device checks, and reports the backend actually
in use so no reader has to guess whether the CUDA kernel was involved.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mamba_ref import Mamba, mamba_backend_info, selective_scan_ref  # noqa: E402
from src.temporal_mamba import TemporalMamba  # noqa: E402
from src.utils import Timer, get_device, load_config, save_json, set_seed  # noqa: E402


def naive_selective_scan(u, delta, A, B, C, D):
    """Independently written reference for the S6 recurrence, used only to check
    selective_scan_ref. Deliberately written as an explicit loop with no einsum."""
    batch, dim, seqlen = u.shape
    dstate = A.shape[1]
    y = torch.zeros_like(u)
    x = torch.zeros(batch, dim, dstate, dtype=u.dtype, device=u.device)
    for t in range(seqlen):
        dt = delta[:, :, t].unsqueeze(-1)              # (b, d, 1)
        dA = torch.exp(dt * A.unsqueeze(0))            # (b, d, n)
        dBu = dt * B[:, :, t].unsqueeze(1) * u[:, :, t].unsqueeze(-1)
        x = dA * x + dBu
        y[:, :, t] = (x * C[:, :, t].unsqueeze(1)).sum(-1) + D.unsqueeze(0) * u[:, :, t]
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = get_device()
    dev = device.device

    B, T, D = 4, int(cfg["video"]["num_frames"]), 768
    d_model = int(cfg["mamba"]["d_model"])

    print("=" * 72)
    print("TEST 8 / CHECKPOINT 7 - MAMBA")
    print("=" * 72)
    backend = mamba_backend_info()
    print("backend in use:")
    for k, v in backend.items():
        print(f"  {k}: {v}")

    problems = []

    # --------------------------------------------------- 1. import + forward
    print(f"\n--- forward pass ---")
    model = TemporalMamba(input_dim=D, d_model=d_model,
                          n_layers=int(cfg["mamba"]["n_layers"]),
                          d_state=int(cfg["mamba"]["d_state"]),
                          d_conv=int(cfg["mamba"]["d_conv"]),
                          expand=int(cfg["mamba"]["expand"]),
                          dropout=0.0,
                          bidirectional=bool(cfg["mamba"]["bidirectional"]),
                          pooling="mean").to(dev).eval()

    x = torch.randn(B, T, D, device=dev)
    with Timer() as timer:
        seq, pooled = model(x)
    print(f"input          : {tuple(x.shape)}")
    print(f"sequence out   : {tuple(seq.shape)}   expected {(B, T, d_model)}")
    print(f"pooled out     : {tuple(pooled.shape)}   expected {(B, d_model)}")
    print(f"forward time   : {timer.elapsed*1000:.1f} ms")
    print(f"finite         : seq={bool(torch.isfinite(seq).all())} "
          f"pooled={bool(torch.isfinite(pooled).all())}")
    print(f"params         : {sum(p.numel() for p in model.parameters()):,}")

    if seq.shape != (B, T, d_model):
        problems.append(f"sequence shape {tuple(seq.shape)} != {(B, T, d_model)}")
    if pooled.shape != (B, d_model):
        problems.append(f"pooled shape {tuple(pooled.shape)} != {(B, d_model)}")
    if not torch.isfinite(seq).all() or not torch.isfinite(pooled).all():
        problems.append("Mamba output contains NaN/Inf")

    # --------------------------------------------------------- 2. causality
    print(f"\n--- causality (unidirectional scan) ---")
    uni = TemporalMamba(input_dim=D, d_model=d_model, n_layers=1, dropout=0.0,
                        bidirectional=False, pooling="mean").to(dev).eval()
    with torch.no_grad():
        base, _ = uni(x)
        perturbed_input = x.clone()
        cut = T // 2
        perturbed_input[:, cut:] += 10.0     # change everything from `cut` onward
        perturbed, _ = uni(perturbed_input)
        before = (base[:, :cut] - perturbed[:, :cut]).abs().max().item()
        after = (base[:, cut:] - perturbed[:, cut:]).abs().max().item()
    print(f"max change at t <  {cut}: {before:.3e}   (must be ~0)")
    print(f"max change at t >= {cut}: {after:.3e}   (must be large)")
    if before > 1e-4:
        problems.append(f"causality violated: perturbing t>={cut} changed earlier outputs "
                        f"by {before:.3e}")
    if after < 1e-3:
        problems.append("perturbing the input did not change the output at all - the "
                        "module may be ignoring its input")

    # A bidirectional block must *fail* this test; checking that confirms the
    # causality probe is actually sensitive rather than trivially passing.
    bi = TemporalMamba(input_dim=D, d_model=d_model, n_layers=1, dropout=0.0,
                       bidirectional=True, pooling="mean").to(dev).eval()
    with torch.no_grad():
        b_base, _ = bi(x)
        b_pert, _ = bi(perturbed_input)
        bi_before = (b_base[:, :cut] - b_pert[:, :cut]).abs().max().item()
    print(f"bidirectional control, change at t < {cut}: {bi_before:.3e}   (must be large)")
    if bi_before < 1e-3:
        problems.append("the bidirectional control did not leak information backwards, so "
                        "the causality test is not actually sensitive")

    # ----------------------------------------- 3. selective scan equivalence
    print(f"\n--- selective_scan_ref vs. an independent naive recurrence ---")
    torch.manual_seed(0)
    b, d, l, n = 2, 6, 9, 4
    u = torch.randn(b, d, l, dtype=torch.float64)
    delta = torch.rand(b, d, l, dtype=torch.float64) * 0.5 + 0.05
    A = -torch.rand(d, n, dtype=torch.float64) - 0.1
    Bm = torch.randn(b, n, l, dtype=torch.float64)
    Cm = torch.randn(b, n, l, dtype=torch.float64)
    Dm = torch.randn(d, dtype=torch.float64)

    ref = selective_scan_ref(u, delta, A, Bm, Cm, Dm, z=None,
                             delta_bias=None, delta_softplus=False)
    naive = naive_selective_scan(u, delta, A, Bm, Cm, Dm)
    err = (ref - naive).abs().max().item()
    print(f"max |selective_scan_ref - naive| = {err:.3e}   (float64)")
    if err > 1e-9:
        problems.append(f"selective_scan_ref disagrees with the naive S6 recurrence "
                        f"by {err:.3e}")

    # ------------------------------------------------------ 4. gradient flow
    print(f"\n--- gradient flow ---")
    train_model = TemporalMamba(input_dim=D, d_model=d_model, n_layers=2, dropout=0.0,
                                pooling="mean").to(dev)
    xt = torch.randn(2, T, D, device=dev, requires_grad=True)
    _, p = train_model(xt)
    loss = p.pow(2).mean()
    loss.backward()
    no_grad = [name for name, prm in train_model.named_parameters()
               if prm.requires_grad and (prm.grad is None or prm.grad.abs().sum() == 0)]
    print(f"loss              : {loss.item():.6f}")
    print(f"input grad finite : {bool(torch.isfinite(xt.grad).all())}")
    print(f"params without gradient: {len(no_grad)}")
    if no_grad:
        print(f"  {no_grad[:8]}")
        problems.append(f"{len(no_grad)} parameters received no gradient: {no_grad[:5]}")
    if not torch.isfinite(xt.grad).all():
        problems.append("input gradient contains NaN/Inf")

    # --------------------------------------------- 5. a bare Mamba mixer too
    print(f"\n--- bare Mamba mixer ---")
    mixer = Mamba(d_model=d_model).to(dev).eval()
    with torch.no_grad():
        out = mixer(torch.randn(2, T, d_model, device=dev))
    print(f"Mamba(d_model={d_model}) : {tuple(out.shape)}  finite={bool(torch.isfinite(out).all())}")
    print(f"  d_inner={mixer.d_inner} dt_rank={mixer.dt_rank} d_state={mixer.d_state}")
    if out.shape != (2, T, d_model):
        problems.append(f"bare Mamba output shape {tuple(out.shape)}")

    save_json({
        "backend": backend,
        "device": device.as_dict(),
        "shapes": {"input": [B, T, D], "sequence": list(seq.shape), "pooled": list(pooled.shape)},
        "forward_ms": round(timer.elapsed * 1000, 2),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "causality": {"max_change_before_cut": before, "max_change_after_cut": after,
                      "bidirectional_control_change_before_cut": bi_before},
        "selective_scan_max_error_vs_naive_float64": err,
        "gradient": {"loss": float(loss.item()),
                     "params_without_gradient": len(no_grad)},
        "mamba_config": {"d_model": d_model, "d_inner": mixer.d_inner,
                         "dt_rank": mixer.dt_rank, "d_state": mixer.d_state},
        "problems": problems,
    }, "artifacts/mamba_check.json")
    print("\nwrote artifacts/mamba_check.json")

    print("\n" + "=" * 72)
    if problems:
        print("CHECKPOINT 7 FAILED:")
        for p in problems:
            print(f"  - {p}")
        print("=" * 72)
        return 1
    print("CHECKPOINT 7 PASSED")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
