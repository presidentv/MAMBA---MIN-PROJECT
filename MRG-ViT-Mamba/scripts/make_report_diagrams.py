"""Diagrams required by the BCSE497J report template.

    fig_gantt.png           2.5 Project Plan
    fig_architecture.png    4.1 System Architecture
    fig_dfd.png             4.2.1 Data Flow Diagram  (mandatory)
    fig_usecase.png         4.2.2 Use Case Diagram   (mandatory)
    fig_sequence.png        4.2.4 Sequence Diagram
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Ellipse  # noqa: E402

from src.utils import ensure_dir, load_config  # noqa: E402

DEEP, LAND, MRSC, GREY, INK = "#1F6F78", "#6B4E9E", "#B4761F", "#8A939C", "#161A1F"
DEEP_BG, LAND_BG, MRS_BG, NEUT = "#E6F1F2", "#EFEAF6", "#F7EEDF", "#F2F4F6"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 200,
                     "savefig.bbox": "tight", "savefig.facecolor": "white"})


def box(ax, x, y, w, h, text, fc=NEUT, ec=INK, fs=8.2, bold=False, lw=1.2, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3, linespacing=1.45)


def arrow(ax, p, q, label=None, color=INK, ls="-", lw=1.2, fs=7.2, off=(0, 0), rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=color,
                                 lw=lw, ls=ls, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=3))
    if label:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], label,
                ha="center", va="center", fontsize=fs, color=color, zorder=4,
                bbox=dict(fc="white", ec="none", pad=1.1))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------------- #
def gantt(out):
    tasks = [
        ("Literature review and problem formulation", 0, 3, GREY),
        ("DAiSEE acquisition, audit and split verification", 2, 2, GREY),
        ("Stage 1: face detection, landmarks, MRS", 3, 3, MRSC),
        ("Motion Reliability Score design and calibration", 4, 3, MRSC),
        ("Stage 2: frozen ViT feature extraction", 6, 2, DEEP),
        ("Mamba temporal encoder (vendored reference)", 7, 3, DEEP),
        ("Fusion and MLP classifier", 9, 2, DEEP),
        ("Learnable MRS weighting", 10, 2, MRSC),
        ("Training, ablation and robustness experiments", 11, 3, LAND),
        ("SHAP explainability analysis", 13, 2, LAND),
        ("Report writing and documentation", 14, 2, GREY),
    ]
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    for i, (name, s, d, c) in enumerate(tasks):
        y = len(tasks) - i - 1
        ax.barh(y, d, left=s, height=.58, color=c, alpha=.92, edgecolor="white", lw=.8)
        ax.text(s + d + .12, y, f"W{s+1}-W{s+d}", va="center", fontsize=7, color=GREY)
    ax.set_yticks(range(len(tasks)), [t[0] for t in reversed(tasks)], fontsize=8.2)
    ax.set_xlabel("Project week", fontsize=9)
    ax.set_xlim(0, 18); ax.set_xticks(range(0, 19, 2))
    ax.grid(axis="x", alpha=.25, lw=.6); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Project schedule (Project-I phase)", fontsize=10.5, fontweight="bold", pad=10)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def architecture(out):
    fig, ax = canvas(10.6, 8.4)
    # layer bands
    for y0, y1, lab, c in ((.885, .995, "INPUT LAYER", NEUT), (.665, .875, "PERCEPTION LAYER", MRS_BG),
                           (.415, .655, "REPRESENTATION LAYER", DEEP_BG),
                           (.185, .405, "FUSION LAYER", LAND_BG), (.02, .175, "OUTPUT LAYER", NEUT)):
        ax.add_patch(FancyBboxPatch((.005, y0), .99, y1 - y0,
                                    boxstyle="round,pad=0.002,rounding_size=0.008",
                                    fc=c, ec="none", alpha=.42, zorder=0))
        ax.text(.012, (y0 + y1) / 2, lab, fontsize=6.6, color=GREY, fontweight="bold",
                ha="left", va="center", rotation=90, zorder=1)

    box(ax, .30, .915, .40, .062, "DAiSEE clip   640x480, 30 fps, 10 s\n300 frames", bold=True)
    box(ax, .30, .805, .40, .058, "Uniform temporal sampling\n$i_k$ = round(k(N-1)/(T-1)),  T = 16")
    arrow(ax, (.50, .915), (.50, .863))

    box(ax, .060, .690, .258, .075, "MediaPipe FaceDetector\n(BlazeFace short-range)\nbox + confidence + eye keypoints",
        fc="white", ec=MRSC, fs=7.6)
    box(ax, .371, .690, .258, .075, "Face crop\nroll-aligned, 0.25 padding\n256 px, JPEG cached", fc="white", fs=7.6)
    box(ax, .682, .690, .258, .075, "MediaPipe FaceLandmarker\n478 landmarks + 52 blendshapes\n+ 4x4 pose matrix",
        fc="white", ec=LAND, fs=7.6)
    arrow(ax, (.42, .805), (.19, .765)); arrow(ax, (.58, .805), (.81, .765))
    arrow(ax, (.318, .727), (.371, .727))

    box(ax, .060, .555, .258, .072, "Motion Reliability Score\n$r_t$ = $\\Sigma_c w_c \\cdot c_t$  in [0,1]\n5 components, learned $w$",
        fc=MRS_BG, ec=MRSC, bold=True, fs=7.6)
    box(ax, .371, .555, .258, .072, "ViT-B/16  (frozen)\nImageNet-21k -> 1k\n[16,3,224,224] -> [16,768]",
        fc=DEEP_BG, ec=DEEP, bold=True, fs=7.6)
    box(ax, .682, .555, .258, .072, "Geometric + behavioural\n28 geometry + 52 blendshapes\n[16, 80]",
        fc=LAND_BG, ec=LAND, fs=7.6)
    arrow(ax, (.189, .690), (.189, .627), color=MRSC)
    arrow(ax, (.50, .690), (.50, .627), color=DEEP)
    arrow(ax, (.811, .690), (.811, .627), color=LAND)
    # landmarks also feed MRS (head pose, eye/iris visibility, motion) -- routed
    # below the ViT box so the link is not mistaken for a landmark -> ViT edge.
    ax.add_patch(FancyArrowPatch((.682, .565), (.318, .565), arrowstyle="-|>",
                                 mutation_scale=11, color=MRSC, lw=1.2, zorder=4,
                                 connectionstyle="arc3,rad=0.16", shrinkA=2, shrinkB=3))
    ax.text(.50, .513, "head pose, eye/iris visibility, motion", ha="center", fontsize=6.8,
            color=MRSC, zorder=5, bbox=dict(fc="white", ec="none", pad=1.1))

    box(ax, .30, .440, .40, .058, "Reliability weighting   $f'_t = r_t \\cdot f_t$   [16, 768]",
        fc="white", ec=MRSC, fs=7.8)
    arrow(ax, (.50, .555), (.50, .498), color=DEEP)
    arrow(ax, (.189, .555), (.189, .469), color=MRSC, ls="--")
    arrow(ax, (.189, .469), (.30, .469), color=MRSC, ls="--")

    box(ax, .060, .300, .39, .075, "Linear 768 -> 256\nMamba x 2  (d_state 16, d_conv 4, expand 2)\nreliability-weighted pooling -> [256]",
        fc=DEEP_BG, ec=DEEP, bold=True, fs=7.6)
    box(ax, .555, .300, .40, .075, "mean || std || mean|$\\Delta$|  -> [240]\nStandardScaler (train stats)\nLinear 240 -> 256",
        fc=LAND_BG, ec=LAND, bold=True, fs=7.6)
    arrow(ax, (.40, .440), (.255, .375), color=DEEP)
    arrow(ax, (.811, .555), (.755, .375), color=LAND)

    box(ax, .30, .205, .40, .052, "Concatenation   [256] || [256]  ->  [512]", fc="white", bold=True, fs=8)
    arrow(ax, (.255, .300), (.42, .257), color=DEEP)
    arrow(ax, (.755, .300), (.58, .257), color=LAND)

    box(ax, .225, .085, .265, .058, "MLP classifier\n512 -> 256 -> 128 -> 4", fc="white", fs=7.8)
    box(ax, .530, .085, .245, .058, "4 engagement logits\nVery Low / Low / High / Very High", fc="white", bold=True, fs=7.6)
    arrow(ax, (.44, .205), (.36, .143))
    arrow(ax, (.49, .114), (.53, .114))
    box(ax, .795, .085, .16, .058, "SHAP\nattribution", fc=NEUT, fs=7.8)
    arrow(ax, (.775, .114), (.795, .114), ls="--", color=GREY)

    ax.set_title("Fig. 4.1  System architecture of the MRG-ViT-Mamba engagement recognition pipeline",
                 fontsize=9.6, fontweight="bold", y=1.005)
    fig.savefig(out); plt.close(fig)
    return out


def dfd(out):
    fig, ax = canvas(10.4, 5.0)
    # external entities
    box(ax, .015, .60, .135, .105, "Webcam /\nDAiSEE clip\n(external entity)", fc=NEUT, bold=True, fs=7.6)
    box(ax, .855, .60, .135, .105, "Engagement\nreport\n(end user)", fc=NEUT, bold=True, fs=7.6)

    procs = [(.185, "1.0\nSample &\ndetect face"), (.355, "2.0\nExtract landmarks\n& compute MRS"),
             (.525, "3.0\nEncode frames\n(frozen ViT)"), (.695, "4.0\nTemporal model\n& classify")]
    for x, t in procs:
        ax.add_patch(FancyBboxPatch((x, .585), .145, .135,
                                    boxstyle="round,pad=0.004,rounding_size=0.05",
                                    fc=DEEP_BG, ec=DEEP, lw=1.3, zorder=2))
        ax.text(x + .0725, .6525, t, ha="center", va="center", fontsize=7.6, zorder=3, linespacing=1.5)

    arrow(ax, (.15, .652), (.185, .652), "video")
    arrow(ax, (.33, .652), (.355, .652), "frames +\ncrops", off=(0, .085))
    arrow(ax, (.50, .652), (.525, .652), "$r_t$,\nlandmarks", off=(0, .085))
    arrow(ax, (.67, .652), (.695, .652), "[16,768]", off=(0, .06))
    arrow(ax, (.84, .652), (.855, .652), "class +\nconfidence", off=(0, .085))

    stores = [(.185, "D1  Face-crop cache\n(JPEG, per clip)"), (.365, "D2  MRS raw signals\n+ landmark features"),
              (.545, "D3  ViT feature cache\n[T, 768] per clip"), (.725, "D4  Model checkpoint\n+ train statistics")]
    for x, t in stores:
        ax.add_patch(FancyBboxPatch((x, .175), .175, .095,
                                    boxstyle="square,pad=0.004", fc=MRS_BG, ec=MRSC, lw=1.1, zorder=2))
        ax.plot([x, x + .175], [.245, .245], color=MRSC, lw=1.0, zorder=3)
        ax.text(x + .0875, .208, t, ha="center", va="center", fontsize=7.1, zorder=4, linespacing=1.4)

    arrow(ax, (.2575, .585), (.2725, .270), color=MRSC, lw=1.0)
    arrow(ax, (.4275, .585), (.4525, .270), color=MRSC, lw=1.0)
    arrow(ax, (.5975, .585), (.6325, .270), color=MRSC, lw=1.0)
    arrow(ax, (.8125, .270), (.7825, .585), color=MRSC, lw=1.0, ls="--")
    ax.text(.50, .085, "Solid arrows write to a data store; the dashed arrow reads the trained "
                       "checkpoint and the training-split statistics back into process 4.0.",
            ha="center", fontsize=7.4, color=GREY)
    ax.set_title("Fig. 4.2  Level-0 data flow diagram — engagement inference on one clip",
                 fontsize=9.6, fontweight="bold", y=.99)
    fig.savefig(out); plt.close(fig)
    return out


def usecase(out):
    fig, ax = canvas(10.0, 6.2)
    ax.add_patch(FancyBboxPatch((.235, .045), .53, .90,
                                boxstyle="round,pad=0.004,rounding_size=0.01",
                                fc="white", ec=INK, lw=1.3, zorder=1))
    ax.text(.50, .915, "MRG-ViT-Mamba Engagement Recognition System",
            ha="center", fontsize=8.6, fontweight="bold", zorder=3)

    def actor(x, y, label):
        ax.add_patch(Circle((x, y + .075), .017, fc="white", ec=INK, lw=1.2, zorder=3))
        ax.plot([x, x], [y + .058, y + .012], color=INK, lw=1.2, zorder=3)
        ax.plot([x - .028, x + .028], [y + .045, y + .045], color=INK, lw=1.2, zorder=3)
        ax.plot([x, x - .024], [y + .012, y - .028], color=INK, lw=1.2, zorder=3)
        ax.plot([x, x + .024], [y + .012, y - .028], color=INK, lw=1.2, zorder=3)
        ax.text(x, y - .062, label, ha="center", fontsize=8, fontweight="bold", zorder=3)

    actor(.085, .60, "Researcher /\nInstructor")
    actor(.915, .46, "System\nAdministrator")

    cases = [(.50, .830, "Audit dataset and verify splits", DEEP),
             (.50, .715, "Run preprocessing (Stage 1 / Stage 2)", DEEP),
             (.50, .600, "Train engagement model", DEEP),
             (.50, .485, "Evaluate on validation / test split", DEEP),
             (.50, .370, "Inspect reliability (MRS) diagnostics", MRSC),
             (.50, .255, "Generate SHAP explanations", LAND),
             (.50, .140, "Manage checkpoints and artifacts", GREY)]
    for x, y, t, c in cases:
        ax.add_patch(Ellipse((x, y), .40, .082, fc="white", ec=c, lw=1.25, zorder=2))
        ax.text(x, y, t, ha="center", va="center", fontsize=7.8, zorder=3)

    for y in (.830, .715, .600, .485, .370, .255):
        ax.plot([.115, .30], [.60, y], color=GREY, lw=.9, zorder=1)
    for y in (.140, .715, .830):
        ax.plot([.885, .70], [.46, y], color=GREY, lw=.9, zorder=1)

    ax.annotate("<<include>>", xy=(.50, .428), fontsize=6.8, color=GREY, ha="center", style="italic")
    ax.add_patch(FancyArrowPatch((.50, .459), (.50, .411), arrowstyle="-|>", ls="--",
                                 mutation_scale=9, color=GREY, lw=.9, zorder=1))
    ax.set_title("Fig. 4.3  Use case diagram", fontsize=9.6, fontweight="bold", y=.985)
    fig.savefig(out); plt.close(fig)
    return out


def sequence(out):
    fig, ax = canvas(10.6, 5.6)
    lanes = [(.075, "User"), (.235, "Pipeline\ndriver"), (.395, "MediaPipe\n(detector +\nlandmarker)"),
             (.555, "MRS\nmodule"), (.715, "ViT-B/16\n(frozen)"), (.885, "Mamba +\nfusion + MLP")]
    for x, name in lanes:
        box(ax, x - .068, .885, .136, .085, name, fc=NEUT, bold=True, fs=7.2)
        ax.plot([x, x], [.10, .885], color=GREY, lw=.9, ls=(0, (4, 3)), zorder=0)

    steps = [(.075, .235, .820, "run inference on clip"),
             (.235, .395, .755, "16 uniformly sampled frames"),
             (.395, .235, .690, "boxes, 478 landmarks, blendshapes, pose"),
             (.235, .555, .625, "crops + landmarks + detector confidence"),
             (.555, .235, .560, "5 raw reliability signals per frame"),
             (.235, .715, .495, "aligned 224 px face crops"),
             (.715, .235, .430, "frame embeddings  [16, 768]"),
             (.235, .885, .365, "features, $r_t$, landmark vector [240]"),
             (.885, .885, .300, "$f'_t = r_t f_t$ -> Mamba -> weighted pool"),
             (.885, .885, .235, "concat [512] -> MLP -> 4 logits"),
             (.885, .075, .150, "predicted class + softmax probabilities")]
    for a, b, y, lab in steps:
        if a == b:
            ax.add_patch(FancyArrowPatch((a, y + .028), (a + .052, y + .028), arrowstyle="-",
                                         color=INK, lw=1.0))
            ax.add_patch(FancyArrowPatch((a + .052, y + .028), (a + .052, y), arrowstyle="-",
                                         color=INK, lw=1.0))
            ax.add_patch(FancyArrowPatch((a + .052, y), (a, y), arrowstyle="-|>",
                                         mutation_scale=9, color=INK, lw=1.0))
            ax.text(a - .012, y + .012, lab, ha="right", fontsize=7.1)
        else:
            col = MRSC if "reliability" in lab or "$r_t$" in lab else INK
            ax.add_patch(FancyArrowPatch((a, y), (b, y), arrowstyle="-|>", mutation_scale=10,
                                         color=col, lw=1.1))
            ax.text((a + b) / 2, y + .019, lab, ha="center", fontsize=7.1, color=col,
                    bbox=dict(fc="white", ec="none", pad=1.0))
    ax.set_title("Fig. 4.4  Sequence diagram — engagement inference on a single clip",
                 fontsize=9.6, fontweight="bold", y=.99)
    fig.savefig(out); plt.close(fig)
    return out


def main() -> int:
    cfg = load_config()
    out = ensure_dir(Path(cfg["paths"]["artifacts_dir"]) / "report")
    for f in (gantt(out / "fig_gantt.png"), architecture(out / "fig_architecture.png"),
              dfd(out / "fig_dfd.png"), usecase(out / "fig_usecase.png"),
              sequence(out / "fig_sequence.png")):
        print("wrote", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
