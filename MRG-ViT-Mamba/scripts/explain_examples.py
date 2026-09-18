"""Annotated worked examples: two test clips from two different people.

For each clip this produces
  * a frame with the MediaPipe face mesh, iris, eye/mouth/brow landmarks, the
    detector box and the decomposed head pose drawn on it,
  * the five MRS components measured on that frame,
  * the model's prediction against the ground truth, with the full decision
    chain: per-frame reliability -> temporal pooling -> branch contributions ->
    class logits and probabilities.

Everything drawn is recomputed from the actual video and the actual checkpoint;
nothing is illustrative.

Outputs (artifacts/report/):
    fig7_example_<clip>_landmarks.png
    fig8_example_<clip>_decision.png
    example_explanations.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.dataset import CachedClipDataset, collate, load_calibration  # noqa: E402
from src.dataset_index import build_index  # noqa: E402
from src.evaluate import load_checkpoint  # noqa: E402
from src.face_processing import FaceDetector  # noqa: E402
from src.landmarks import (  # noqa: E402
    LEFT_EYE, LEFT_IRIS_CENTER, MOUTH, RIGHT_EYE, RIGHT_IRIS_CENTER,
    LEFT_BROW, RIGHT_BROW, NOSE_TIP, FACE_LEFT, FACE_RIGHT, FACE_TOP, FACE_BOTTOM,
    LandmarkExtractor,
)
from src.metrics import DEFAULT_CLASS_NAMES  # noqa: E402
from src.mrs import COMPONENTS, components_from_arrays  # noqa: E402
from src.preprocess import stage1_key, stage1_path, stage2_key  # noqa: E402
from src.utils import ensure_dir, load_config, save_json  # noqa: E402
from src.vit_encoder import ViTFrameEncoder  # noqa: E402
from src.video_sampling import read_frames  # noqa: E402

DEEP, LAND, MRSC, GREY = "#1F6F78", "#6B4E9E", "#B4761F", "#5A646E"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 160, "savefig.bbox": "tight"})


def draw_annotated(frame, det, lm, w, h):
    """Face mesh, iris, key landmarks, detector box and head-pose axes."""
    img = frame.copy()
    pts = lm.landmarks
    px = lambda i: (int(pts[i, 0] * w), int(pts[i, 1] * h))  # noqa: E731

    # full mesh, faint
    for i in range(pts.shape[0]):
        cv2.circle(img, px(i), 1, (150, 150, 150), -1, cv2.LINE_AA)

    # eyes (green) and their corners
    for eye in (RIGHT_EYE, LEFT_EYE):
        for grp in ("top", "bottom"):
            for i in eye[grp]:
                cv2.circle(img, px(i), 2, (60, 220, 60), -1, cv2.LINE_AA)
        cv2.line(img, px(eye["outer"]), px(eye["inner"]), (60, 220, 60), 1, cv2.LINE_AA)
    # iris (cyan)
    for i in (RIGHT_IRIS_CENTER, LEFT_IRIS_CENTER):
        cv2.circle(img, px(i), 4, (255, 230, 60), 2, cv2.LINE_AA)
    # mouth (magenta)
    for a, b in ((MOUTH["left"], MOUTH["right"]), (MOUTH["top"], MOUTH["bottom"])):
        cv2.line(img, px(a), px(b), (220, 60, 220), 1, cv2.LINE_AA)
    for k in MOUTH.values():
        cv2.circle(img, px(k), 2, (220, 60, 220), -1, cv2.LINE_AA)
    # brows (orange)
    for i in (LEFT_BROW, RIGHT_BROW):
        cv2.circle(img, px(i), 3, (40, 150, 240), -1, cv2.LINE_AA)
    # face extent (blue)
    for i in (FACE_LEFT, FACE_RIGHT, FACE_TOP, FACE_BOTTOM):
        cv2.circle(img, px(i), 3, (240, 160, 40), -1, cv2.LINE_AA)

    # detector box
    x0, y0, x1, y1 = det.bbox
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 2, cv2.LINE_AA)
    if det.confidence is not None:
        cv2.putText(img, f"det {det.confidence:.3f}", (x0, max(16, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 200, 255), 1, cv2.LINE_AA)

    # head-pose axes from the nose tip
    if lm.head_pose_deg is not None:
        yaw, pitch, roll = (np.radians(v) for v in lm.head_pose_deg)
        nx, ny = px(NOSE_TIP); L = 70
        cv2.arrowedLine(img, (nx, ny), (int(nx + L * np.cos(yaw)), ny),
                        (0, 0, 255), 2, cv2.LINE_AA, tipLength=.2)     # yaw  X
        cv2.arrowedLine(img, (nx, ny), (nx, int(ny - L * np.cos(pitch))),
                        (0, 255, 0), 2, cv2.LINE_AA, tipLength=.2)     # pitch Y
        cv2.arrowedLine(img, (nx, ny),
                        (int(nx + .5 * L * np.sin(roll)), int(ny + .5 * L * np.cos(roll))),
                        (255, 0, 0), 2, cv2.LINE_AA, tipLength=.2)     # roll  Z
    return img


def legend_panel(ax):
    items = [("face mesh (478 pts)", "#969696"), ("eye contour + corners", "#3CDC3C"),
             ("iris centre", "#3CE6FF"), ("mouth geometry", "#DC3CDC"),
             ("brow points", "#F09628"), ("face extent", "#28A0F0"),
             ("detector box", "#FFC800"), ("head pose  yaw / pitch / roll", "#FF3232")]
    ax.axis("off")
    for i, (lab, col) in enumerate(items):
        ax.add_patch(plt.Rectangle((0.02, .92 - i * .115), .05, .055, color=col,
                                   transform=ax.transAxes, clip_on=False))
        ax.text(.10, .947 - i * .115, lab, transform=ax.transAxes, fontsize=8.4, va="center")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="mini64")
    ap.add_argument("--checkpoint", default="checkpoints/mini64/best.pt")
    ap.add_argument("--frame", type=int, default=6, help="which sampled frame to draw")
    args = ap.parse_args()

    cfg = load_config(); index = build_index(cfg)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ensure_dir(Path(cfg["paths"]["artifacts_dir"]) / "report")
    cal = load_calibration(cfg)

    spec = ViTFrameEncoder(str(cfg["vit"]["model_name"]), pretrained=True, freeze=True).spec
    s1 = stage1_key(cfg); s2 = stage2_key(cfg, s1, spec.name, spec.input_size)
    model, ckpt = load_checkpoint(cfg, args.checkpoint, dev)

    ds = CachedClipDataset(cfg, index, "test", s1, s2, cal, mrs_mode=ckpt.get("mrs_mode"))
    # Two clips from two DIFFERENT subjects.
    chosen, seen = [], set()
    for i, (rec, _, _) in enumerate(ds.records):
        if rec.subject_id not in seen:
            chosen.append(i); seen.add(rec.subject_id)
        if len(chosen) == 2:
            break

    detector = FaceDetector(Path(cfg["paths"]["models_dir"]) / "blaze_face_short_range.tflite",
                            min_confidence=float(cfg["face"]["min_detection_confidence"]))
    landmarker = LandmarkExtractor(dict(cfg["landmarks"]))
    report = []

    for idx in chosen:
        rec, p1, _ = ds.records[idx]
        item = ds[idx]
        batch = collate([item])
        with torch.no_grad():
            logits, mid = model(batch["vit_features"].to(dev), batch["mrs"].to(dev),
                                batch["landmark_features"].to(dev),
                                return_intermediates=True,
                                mrs_components=batch["mrs_components"].to(dev))
        probs = torch.softmax(logits, -1)[0].cpu().numpy()
        pred = int(logits.argmax(-1)[0]); true = int(item["label"])
        r_used = mid["mrs_used"][0].cpu().numpy()

        # --- recompute the frame from the source video and annotate it ---
        frames, indices, probe = read_frames(rec.path, int(cfg["video"]["num_frames"]))
        t = min(args.frame, len(frames) - 1)
        frame = frames[t]; h, w = frame.shape[:2]
        det = detector.detect(frame); lm = landmarker.extract(frame)
        annotated = draw_annotated(frame, det, lm, w, h) if lm.found else frame

        d = np.load(p1)
        comp = components_from_arrays(
            d["blur_raw"], d["face_area_fraction"], d["detector_confidence"],
            d["head_pose_deg"], d["mrs_eye_visibility"], d["mrs_motion_consistency"],
            dict(cfg["mrs"]), cal)
        frame_comps = {c: float(comp[c][t]) for c in COMPONENTS}
        wts = (model.learnable_mrs.weights().detach().cpu().numpy()
               if model.learnable_mrs is not None else np.full(5, .2))

        # ---------------- figure 7: the annotated frame ----------------
        fig = plt.figure(figsize=(13.2, 5.3))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.55, .62, 1.15], wspace=.22)
        a0 = fig.add_subplot(gs[0]); a0.axis("off")
        a0.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        a0.set_title(f"{rec.clip_id}   subject {rec.subject_id}   "
                     f"frame {indices[t]} of {probe.frame_count}",
                     fontsize=10, fontweight="bold")
        legend_panel(fig.add_subplot(gs[1]))

        a2 = fig.add_subplot(gs[2])
        names = [c.replace("_", "\n") for c in COMPONENTS]
        vals = [frame_comps[c] for c in COMPONENTS]
        cols = ["#B4761F", "#1F6F78", "#6B4E9E", "#7A9E3B", "#C4553B"]
        b = a2.barh(np.arange(5), vals, color=cols, height=.6)
        a2.bar_label(b, fmt="%.3f", fontsize=8, padding=2)
        a2.set_yticks(np.arange(5), names, fontsize=8); a2.invert_yaxis()
        a2.set_xlim(0, 1.18); a2.set_xlabel("component value")
        mrs_frame = float((np.array(vals) * wts).sum())
        a2.set_title(f"MRS components on this frame\nweighted MRS = {mrs_frame:.3f}",
                     fontsize=9.5, fontweight="bold")
        fig.suptitle("MediaPipe features extracted from a real DAiSEE frame", fontsize=11, y=1.02)
        f7 = out / f"fig7_example_{rec.stem}_landmarks.png"
        fig.savefig(f7); plt.close(fig)

        # ---------------- figure 8: the decision chain ----------------
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9),
                                 gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]})
        a = axes[0]
        a.plot(range(len(r_used)), r_used, "-o", color=MRSC, ms=4, lw=1.7)
        a.axvline(t, color=DEEP, ls="--", lw=1.3)
        a.annotate(f"frame shown\n(t={t})", (t, r_used[t]), xytext=(6, -18),
                   textcoords="offset points", fontsize=7.6, color=DEEP)
        a.axhline(r_used.mean(), color=GREY, ls=":", lw=1.2)
        a.set_xlabel("sampled frame index  t"); a.set_ylabel("reliability  $r_t$")
        a.set_ylim(0, 1)
        a.set_title(f"Per-frame reliability\nmean {r_used.mean():.3f}", fontsize=9.5,
                    fontweight="bold")

        a = axes[1]
        order = np.argsort(wts)[::-1]
        a.barh(np.arange(5), wts[order], color=[cols[i] for i in order], height=.6)
        a.axvline(.2, color=GREY, ls="--", lw=1.2)
        a.set_yticks(np.arange(5), [COMPONENTS[i].replace("_", "\n") for i in order], fontsize=8)
        a.invert_yaxis(); a.set_xlabel("learned weight")
        a.set_title("Learned MRS weights\n(dashed = 0.20 start)", fontsize=9.5, fontweight="bold")

        a = axes[2]
        bar_cols = ["#C8D0D8"] * 4
        bar_cols[pred] = "#C4553B" if pred != true else "#1F6F78"
        if pred != true:
            bar_cols[true] = "#7A9E3B"
        b = a.bar(np.arange(4), probs, color=bar_cols)
        a.bar_label(b, fmt="%.3f", fontsize=7.6, padding=2)
        a.set_xticks(np.arange(4), [n.replace(" ", "\n") for n in DEFAULT_CLASS_NAMES], fontsize=8)
        a.set_ylim(0, min(1.0, probs.max() * 1.35)); a.set_ylabel("softmax probability")
        ok = "CORRECT" if pred == true else "INCORRECT"
        a.set_title(f"true = {DEFAULT_CLASS_NAMES[true]}   "
                    f"pred = {DEFAULT_CLASS_NAMES[pred]}   [{ok}]",
                    fontsize=9.5, fontweight="bold",
                    color="#1F6F78" if pred == true else "#C4553B")
        fig.suptitle(f"How the prediction was reached  -  {rec.clip_id} "
                     f"(subject {rec.subject_id})", fontsize=11, y=1.04)
        f8 = out / f"fig8_example_{rec.stem}_decision.png"
        fig.savefig(f8); plt.close(fig)

        report.append({
            "clip_id": rec.clip_id, "subject_id": rec.subject_id,
            "frame_drawn": {"sampled_index": t, "source_frame": int(indices[t])},
            "true_class": DEFAULT_CLASS_NAMES[true], "true_index": true,
            "predicted_class": DEFAULT_CLASS_NAMES[pred], "predicted_index": pred,
            "correct": pred == true,
            "probabilities": {n: round(float(p), 4) for n, p in zip(DEFAULT_CLASS_NAMES, probs)},
            "logits": [round(float(v), 4) for v in logits[0].cpu().numpy()],
            "frame_mrs_components": {k: round(v, 4) for k, v in frame_comps.items()},
            "frame_mrs": round(mrs_frame, 4),
            "clip_mrs_mean": round(float(r_used.mean()), 4),
            "clip_mrs_min": round(float(r_used.min()), 4),
            "clip_mrs_max": round(float(r_used.max()), 4),
            "per_frame_mrs": [round(float(v), 4) for v in r_used],
            "learned_weights": {c: round(float(v), 4) for c, v in zip(COMPONENTS, wts)},
            "detector_confidence": (round(float(det.confidence), 4)
                                    if det.confidence is not None else None),
            "head_pose_deg": ([round(float(v), 2) for v in lm.head_pose_deg]
                              if lm.head_pose_deg is not None else None),
            "landmarks_found": bool(lm.found),
            "figures": {"landmarks": str(f7.name), "decision": str(f8.name)},
        })
        print(f"{rec.clip_id}  subject {rec.subject_id}  true={DEFAULT_CLASS_NAMES[true]} "
              f"pred={DEFAULT_CLASS_NAMES[pred]}  {'OK' if pred==true else 'WRONG'}  "
              f"p={probs.max():.3f}  MRS mean {r_used.mean():.3f}")

    detector.close(); landmarker.close()
    save_json({"checkpoint": args.checkpoint, "run": args.run,
               "class_names": list(DEFAULT_CLASS_NAMES), "examples": report},
              out / "example_explanations.json")
    print(f"\nwrote {out/'example_explanations.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
