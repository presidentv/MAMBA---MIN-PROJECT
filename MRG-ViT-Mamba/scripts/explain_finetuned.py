"""Worked examples for the fine-tuned model: three test clips, three people.

For each clip this draws a real frame with the MediaPipe face markers, the
actual and predicted engagement level, and how the prediction was reached:

  * where the ViT looked on that frame (Grad-CAM for the predicted class);
  * the per-frame reliability r_t and the temporal pooling weight it gives
    each frame, with the learned MRS component weights behind it;
  * how much of the predicted-class score came from the deep (ViT -> Mamba)
    branch and how much from the landmark branch (integrated gradients over
    the fused vector, from an all-zero input);
  * which named landmark features pushed the prediction up or down, compared
    with the average training clip (integrated gradients, baseline = the
    training-set mean the model standardises with);
  * the class probabilities.

The clips are chosen from the test predictions written by run_finetune.py:
the most confident correct prediction, the most confident mistake, and the
rarest remaining class, each from a different person (src/explain.py,
choose_examples). --clips picks them explicitly instead.

Everything is recomputed from the actual video and the actual checkpoint.

Outputs (artifacts/examples_<run>/, git-ignored: the images show DAiSEE
participants, who must not be redistributed, and examples.json pairs clip ids
with their labels):
    example<k>_<clip>_landmarks.png
    example<k>_<clip>_decision.png
    examples_overview.png
    examples.json

Run:
    python scripts/explain_finetuned.py --config configs/config_full.yaml --run full_ft32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.explain_examples import draw_annotated, legend_panel  # noqa: E402
from src.dataset_index import build_index  # noqa: E402
from src.explain import (  # noqa: E402
    build_group_index, choose_examples, explain_clip, pick_frame,
)
from src.face_processing import FaceDetector  # noqa: E402
from src.finetune import build_finetune_datasets  # noqa: E402
from src.landmarks import LandmarkExtractor  # noqa: E402
from src.metrics import DEFAULT_CLASS_NAMES  # noqa: E402
from src.model import MRGViTMamba  # noqa: E402
from src.mrs import COMPONENTS  # noqa: E402
from src.preprocess import stage1_key  # noqa: E402
from src.utils import ensure_dir, load_config, resolve_path, save_json  # noqa: E402
from src.video_sampling import read_frames  # noqa: E402
from src.vit_encoder import ViTFrameEncoder  # noqa: E402

DEEP, LAND, MRSC, GREY, LIGHT = "#1F6F78", "#6B4E9E", "#B4761F", "#5A646E", "#C8D0D8"
OK, BAD, TRUE_C = "#1F6F78", "#C4553B", "#7A9E3B"
COMP_COLS = ["#B4761F", "#1F6F78", "#6B4E9E", "#7A9E3B", "#C4553B"]
GROUP_COLS = {"eye_openness": "#1F6F78", "iris_gaze": "#6B4E9E", "head_pose": "#C4553B",
              "brow": "#B4761F", "mouth_jaw": "#7A9E3B", "cheek_nose": "#A0527E",
              "face_scale": "#3B6EA5", "neutral": "#8C8C8C"}


def crop_rgb(frame_tensor: torch.Tensor, spec) -> np.ndarray:
    """Undo the backbone normalisation: [3, S, S] -> uint8 RGB [S, S, 3]."""
    mean = np.asarray(spec.mean, dtype=np.float32)[:, None, None]
    std = np.asarray(spec.std, dtype=np.float32)[:, None, None]
    img = frame_tensor.float().cpu().numpy() * std + mean
    return (np.clip(img, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)


def verdict(true: int, pred: int) -> tuple[str, str]:
    return ("CORRECT", OK) if true == pred else ("INCORRECT", BAD)


def fig_landmarks(k, rec, annotated, t, source_frame, frame_count, comps, weights, true, pred,
                  probs, out):
    fig = plt.figure(figsize=(14.2, 5.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, .72, 1.15], wspace=.38)
    a0 = fig.add_subplot(gs[0]); a0.axis("off")
    a0.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    a0.set_title(f"{rec.clip_id}   person {rec.subject_id}   "
                 f"frame {source_frame} of {frame_count}", fontsize=10, fontweight="bold")
    legend_panel(fig.add_subplot(gs[1]))

    a2 = fig.add_subplot(gs[2])
    b = a2.barh(np.arange(5), comps, color=COMP_COLS, height=.6)
    a2.bar_label(b, fmt="%.3f", fontsize=8, padding=2)
    a2.set_yticks(np.arange(5), [c.replace("_", "\n") for c in COMPONENTS], fontsize=8)
    a2.invert_yaxis(); a2.set_xlim(0, 1.18); a2.set_xlabel("component value")
    a2.set_title(f"MRS components on this frame\nweighted MRS = {float(comps @ weights):.3f}",
                 fontsize=9.5, fontweight="bold")
    word, col = verdict(true, pred)
    fig.suptitle(f"Example {k}:  actual = {DEFAULT_CLASS_NAMES[true]}   "
                 f"predicted = {DEFAULT_CLASS_NAMES[pred]} ({probs[pred]:.2f})   [{word}]",
                 fontsize=11.5, fontweight="bold", color=col, y=1.03)
    fig.savefig(out); plt.close(fig)


def fig_decision(k, rec, ex, t, crop, weights, out, top_n=8):
    true, pred, probs = ex["true"], ex["pred"], ex["probs"]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.4),
                             gridspec_kw={"hspace": .72, "wspace": .62})

    # 1. where the ViT looked
    a = axes[0, 0]
    cam = cv2.resize(ex["gradcam"][t], (crop.shape[1], crop.shape[0]),
                     interpolation=cv2.INTER_CUBIC)
    a.imshow(crop); a.imshow(np.clip(cam, 0, 1), cmap="jet", alpha=.42)
    a.axis("off")
    a.set_title(f"1. Where the ViT looked (frame t={t})\n"
                f"Grad-CAM for '{DEFAULT_CLASS_NAMES[pred]}', face crop the model saw",
                fontsize=9.2, fontweight="bold")

    # 2. reliability and pooling weight per frame
    a = axes[0, 1]
    T = len(ex["mrs"])
    if ex["pool_weights"] is not None:
        bars = a.bar(range(T), ex["pool_weights"], color=LIGHT, width=.8,
                     label="pooling weight  $r_t / \\Sigma r$")
        bars[t].set_color(DEEP)
        a.set_ylabel("share of the clip embedding")
    a2 = a.twinx()
    a2.plot(range(T), ex["mrs"], "-o", color=MRSC, ms=3, lw=1.5, label="reliability $r_t$")
    a2.set_ylim(0, 1.05); a2.set_ylabel("reliability $r_t$", color=MRSC); a2.grid(False)
    a.set_xlabel("sampled frame")
    a.set_title(f"2. How much each frame counted\n({ex['pooling']} pooling; "
                f"shown frame highlighted)", fontsize=9.2, fontweight="bold")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, fontsize=7.2, loc="upper center", ncol=2,
              bbox_to_anchor=(.5, -.2), frameon=False)

    # 3. learned MRS weights
    a = axes[0, 2]
    order = np.argsort(weights)[::-1]
    a.barh(np.arange(5), weights[order], color=[COMP_COLS[i] for i in order], height=.6)
    a.axvline(.2, color=GREY, ls="--", lw=1.1)
    a.set_yticks(np.arange(5), [COMPONENTS[i].replace("_", "\n") for i in order], fontsize=8)
    a.invert_yaxis(); a.set_xlabel("learned weight (dashed = 0.20 start)")
    a.set_title("3. What reliability is made of\n(learned MRS component weights)",
                fontsize=9.2, fontweight="bold")

    # 4. logit decomposition: deep vs landmark branch
    a = axes[1, 0]
    dec = ex["logit_decomposition"]
    steps = [("zero\ninput", dec["zero_input_logit"], GREY),
             ("+ ViT and\nMamba", dec["deep_branch"], DEEP),
             ("+ land-\nmarks", dec["landmark_branch"], LAND)]
    run, ends = 0.0, [0.0]
    for i, (lab, v, col) in enumerate(steps):
        bottom = 0.0 if i == 0 else run
        a.bar(i, v, bottom=bottom, color=col, width=.6)
        a.text(i, bottom + v, f"{v:+.2f}", ha="center",
               va="bottom" if v >= 0 else "top", fontsize=8)
        run = v if i == 0 else run + v
        ends += [bottom, bottom + v]
    final = dec["predicted_logit"]
    a.bar(3, final, color=OK if true == pred else BAD, width=.6)
    a.text(3, final, f"{final:.2f}", ha="center", va="bottom" if final >= 0 else "top",
           fontsize=8)
    ends.append(final)
    lo, hi = min(ends), max(ends)
    pad = 0.18 * (hi - lo or 1.0)
    a.set_ylim(lo - pad, hi + pad)
    a.axhline(0, color=GREY, lw=.8)
    a.set_xticks(range(4), [s[0] for s in steps] + ["score for\n" + DEFAULT_CLASS_NAMES[pred]],
                 fontsize=7.8)
    a.set_ylabel("logit (score before softmax)")
    a.set_title("4. Where the score came from\n(integrated gradients over the fused vector)",
                fontsize=9.2, fontweight="bold")

    # 5. named landmark features
    a = axes[1, 1]
    la = ex["landmark_attribution"]
    if la is not None:
        attr = la["attribution"]
        idx = np.argsort(-np.abs(attr))[:top_n][::-1]
        group_of = {}
        for g, cols in build_group_index(0, tuple(la["names"])).items():
            for c in cols:
                group_of[c] = g
        cols = [GROUP_COLS.get(group_of.get(i, ""), GREY) for i in idx]
        a.barh(range(len(idx)), attr[idx], color=cols, height=.65)
        a.axvline(0, color=GREY, lw=.8)
        a.set_yticks(range(len(idx)), [la["names"][i] for i in idx], fontsize=7)
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, symmetric=True))
        a.set_xlabel(f"contribution to '{DEFAULT_CLASS_NAMES[pred]}' score")
        shown = sorted({group_of.get(i, "other") for i in idx})
        handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLS.get(g, GREY)) for g in shown]
        a.legend(handles, [g.replace("_", " ") for g in shown], frameon=False, fontsize=6.8,
                 loc="upper center", bbox_to_anchor=(.5, -.2), ncol=min(3, len(shown)))
    else:
        a.text(.5, .5, "landmark branch disabled", ha="center", transform=a.transAxes)
    a.set_title(f"5. Facial behaviour that mattered most\n(top {top_n} landmark features "
                f"vs the average training clip)", fontsize=9.2, fontweight="bold")

    # 6. probabilities
    a = axes[1, 2]
    cols = [LIGHT] * len(probs)
    cols[pred] = OK if pred == true else BAD
    if pred != true:
        cols[true] = TRUE_C
    b = a.bar(range(len(probs)), probs, color=cols)
    a.bar_label(b, fmt="%.3f", fontsize=7.8, padding=2)
    a.set_xticks(range(len(probs)), [n.replace(" ", "\n") for n in DEFAULT_CLASS_NAMES],
                 fontsize=8)
    a.set_ylim(0, 1.08); a.set_ylabel("probability")
    word, col = verdict(true, pred)
    a.set_title(f"6. Output: actual {DEFAULT_CLASS_NAMES[true]}, predicted "
                f"{DEFAULT_CLASS_NAMES[pred]}\n[{word}]", fontsize=9.2, fontweight="bold",
                color=col)

    fig.suptitle(f"Example {k}: how the prediction was reached   ({rec.clip_id}, "
                 f"person {rec.subject_id})", fontsize=12, fontweight="bold", y=.99)
    fig.savefig(out); plt.close(fig)


def fig_overview(panels, out):
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6))
    for a, p in zip(np.atleast_1d(axes), panels):
        a.imshow(cv2.cvtColor(p["image"], cv2.COLOR_BGR2RGB)); a.axis("off")
        word, col = verdict(p["true"], p["pred"])
        a.set_title(f"Example {p['k']}  (person {p['subject']})\n"
                    f"actual: {DEFAULT_CLASS_NAMES[p['true']]}\n"
                    f"predicted: {DEFAULT_CLASS_NAMES[p['pred']]}  "
                    f"(p = {p['prob']:.2f})  [{word}]",
                    fontsize=10, fontweight="bold", color=col)
    fig.suptitle("Three test clips, three different people", fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def summary_text(ex, t) -> str:
    dec = ex["logit_decomposition"]
    true, pred = DEFAULT_CLASS_NAMES[ex["true"]], DEFAULT_CLASS_NAMES[ex["pred"]]
    parts = [f"Actual {true}; predicted {pred} with probability {ex['probs'][ex['pred']]:.2f} "
             f"({'correct' if ex['true'] == ex['pred'] else 'incorrect'})."]
    parts.append(f"Relative to an all-zero fused input (score {dec['zero_input_logit']:+.2f}), "
                 f"the ViT->Mamba branch added {dec['deep_branch']:+.2f} and the landmark "
                 f"branch {dec['landmark_branch']:+.2f} to the {pred} score, "
                 f"giving {dec['predicted_logit']:.2f}.")
    if ex["pool_weights"] is not None:
        top = np.argsort(ex["pool_weights"])[::-1][:3]
        parts.append(f"Mean frame reliability was {ex['mrs'].mean():.2f}; the most heavily "
                     f"weighted frames were {', '.join(str(int(i)) for i in sorted(top))} "
                     f"(frame {t} is drawn).")
    la = ex["landmark_attribution"]
    if la is not None:
        idx = np.argsort(-np.abs(la["attribution"]))[:3]
        feats = ", ".join(f"{la['names'][i]} ({la['attribution'][i]:+.2f})" for i in idx)
        parts.append(f"Compared with the average training clip, the landmark features that "
                     f"moved the {pred} score most were {feats}.")
    return " ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_full.yaml")
    ap.add_argument("--run", default="full_ft32")
    ap.add_argument("--checkpoint", default=None,
                    help="default: checkpoints/<run>/best.pt (the evaluated model)")
    ap.add_argument("--n", type=int, default=3, help="number of examples (one per person)")
    ap.add_argument("--clips", nargs="+", default=None,
                    help="explicit test clip ids instead of the automatic choice")
    ap.add_argument("--out-dir", default=None, help="default: artifacts/examples_<run>")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    index = build_index(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ensure_dir(args.out_dir or Path(cfg["paths"]["artifacts_dir"]) / f"examples_{args.run}")

    ckpt_path = (Path(args.checkpoint) if args.checkpoint else
                 resolve_path(Path(cfg["paths"]["checkpoints_dir"]) / args.run / "best.pt"))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ccfg = ckpt["config"]
    encoder = ViTFrameEncoder(model_name=ccfg["vit"]["model_name"], pretrained=False,
                              freeze=True, chunk_size=int(ccfg["vit"]["batch_size"]))
    model = MRGViTMamba(vit_dim=int(ckpt["vit_dim"]), landmark_dim=int(ckpt["landmark_dim"]),
                        cfg=ccfg, vit_encoder=encoder).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    weights = (model.learnable_mrs.weights().detach().cpu().numpy()
               if model.learnable_mrs is not None else np.full(5, .2))

    ds = build_finetune_datasets(ccfg, index, stage1_key(ccfg), encoder.spec, ("test",))["test"]
    position = {rec.clip_id: i for i, (rec, _) in enumerate(ds.records)}

    preds_path = resolve_path(Path(cfg["paths"]["artifacts_dir"]) /
                              f"per_clip_predictions_{args.run}_test.json")
    if not preds_path.is_file():
        print(f"{preds_path} not found: run scripts/run_finetune.py first; it writes the "
              f"test predictions this script chooses examples from.")
        return 1
    predictions = [p for p in json.loads(preds_path.read_text(encoding="utf-8"))["predictions"]
                   if p["clip_id"] in position]
    if args.clips:
        by_id = {p["clip_id"]: p for p in predictions}
        missing = [c for c in args.clips if c not in by_id]
        if missing:
            print(f"not in the test predictions: {missing}")
            return 1
        chosen = [by_id[c] for c in args.clips]
    else:
        chosen = choose_examples(predictions, args.n)

    models_dir = resolve_path(cfg["paths"]["models_dir"])
    detector = FaceDetector(models_dir / "blaze_face_short_range.tflite",
                            min_confidence=float(cfg["face"]["min_detection_confidence"]))
    landmarker = LandmarkExtractor(dict(cfg["landmarks"]))
    report, panels = [], []

    try:
        for k, choice in enumerate(chosen, start=1):
            rec, _ = ds.records[position[choice["clip_id"]]]
            item = ds[position[choice["clip_id"]]]
            ex = explain_clip(model, item, device)
            if ex["pred"] != choice["pred"]:
                print(f"note: {rec.clip_id} is predicted {DEFAULT_CLASS_NAMES[ex['pred']]} "
                      f"here but {DEFAULT_CLASS_NAMES[choice['pred']]} in the test "
                      f"predictions file - is {ckpt_path.name} the checkpoint that was "
                      f"evaluated?")
            t = pick_frame(ex["pool_weights"] if ex["pool_weights"] is not None else ex["mrs"])

            frames, indices, probe = read_frames(rec.path, int(ccfg["video"]["num_frames"]))
            t = min(t, len(frames) - 1)
            frame = frames[t]
            h, w = frame.shape[:2]
            det, lm = detector.detect(frame), landmarker.extract(frame)
            annotated = draw_annotated(frame, det, lm, w, h) if lm.found else frame
            comps = ex["mrs_components"][t]

            stem = f"example{k}_{rec.stem}"
            f_land, f_dec = out / f"{stem}_landmarks.png", out / f"{stem}_decision.png"
            fig_landmarks(k, rec, annotated, t, int(indices[t]), probe.frame_count, comps,
                          weights, ex["true"], ex["pred"], ex["probs"], f_land)
            fig_decision(k, rec, ex, t, crop_rgb(item["frames"][t], encoder.spec), weights,
                         f_dec)
            panels.append({"k": k, "image": annotated, "true": ex["true"], "pred": ex["pred"],
                           "prob": float(ex["probs"][ex["pred"]]), "subject": rec.subject_id})

            dec, la = ex["logit_decomposition"], ex["landmark_attribution"]
            text = summary_text(ex, t)
            report.append({
                "example": k, "clip_id": rec.clip_id, "subject_id": rec.subject_id,
                "actual": DEFAULT_CLASS_NAMES[ex["true"]],
                "predicted": DEFAULT_CLASS_NAMES[ex["pred"]],
                "correct": ex["true"] == ex["pred"],
                "probabilities": {n: round(float(p), 4)
                                  for n, p in zip(DEFAULT_CLASS_NAMES, ex["probs"])},
                "logits": [round(float(v), 4) for v in ex["logits"]],
                "explanation": text,
                "frame_drawn": {"sampled_index": t, "source_frame": int(indices[t]),
                                "landmarks_found": bool(lm.found),
                                "head_pose_deg": ([round(float(v), 2) for v in lm.head_pose_deg]
                                                  if lm.head_pose_deg is not None else None)},
                "per_frame_reliability": [round(float(v), 4) for v in ex["mrs"]],
                "pooling": ex["pooling"],
                "per_frame_pooling_weight": (None if ex["pool_weights"] is None else
                                             [round(float(v), 4) for v in ex["pool_weights"]]),
                "frame_mrs_components": {c: round(float(v), 4) for c, v in zip(COMPONENTS, comps)},
                "learned_mrs_weights": {c: round(float(v), 4) for c, v in zip(COMPONENTS, weights)},
                "logit_decomposition": {key: round(float(v), 4) for key, v in dec.items()},
                "landmark_group_contribution": (None if la is None else
                                                {g: round(v, 4) for g, v in
                                                 la["group_attribution"].items()}),
                "top_landmark_features": (None if la is None else [
                    {"feature": la["names"][i],
                     "contribution": round(float(la["attribution"][i]), 4),
                     "value": round(float(la["value"][i]), 4),
                     "train_mean": round(float(la["train_mean"][i]), 4)}
                    for i in np.argsort(-np.abs(la["attribution"]))[:10]]),
                "figures": {"landmarks": f_land.name, "decision": f_dec.name},
            })
            print(f"\nexample {k}: {rec.clip_id} (person {rec.subject_id})\n  {text}")
    finally:
        detector.close()
        landmarker.close()

    if panels:
        fig_overview(panels, out / "examples_overview.png")
    save_json({"checkpoint": str(ckpt_path), "run": args.run,
               "class_names": list(DEFAULT_CLASS_NAMES),
               "method": {
                   "gradcam": "Grad-CAM on the ViT's last block (input to its attention), "
                              "target = predicted-class logit of the full model",
                   "logit_decomposition": "integrated gradients over the fused vector "
                                          "[deep || landmark], baseline all zeros, 64 steps",
                   "landmark_features": "integrated gradients over the landmark clip "
                                        "features, baseline = training-set mean, deep "
                                        "branch held at its actual value, 64 steps",
                   "note": "attribution shows what this model relied on, not what causes "
                           "engagement"},
               "examples": report},
              out / "examples.json")
    print(f"\nwrote {len(report)} examples to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
