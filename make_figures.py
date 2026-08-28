"""
make_figures.py
Publication-quality 300 DPI figures for the GranularStory paper.
Reads the *_results.csv files (schema: Target, Detected, [Story]) and regenerates
every figure. Drop this in the project root and run:  python make_figures.py

No em dashes anywhere (project style rule). Layman note: this script only READS your
result CSVs and DRAWS charts, it does not run any model, so it is fast and safe to rerun.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Serif"

EMOTIONS = ["Joy", "Fear", "Sadness", "Anger", "Trust", "Surprise", "Anticipation"]

def acc(path):
    d = pd.read_csv(path)
    return accuracy_score(d["Target"], d["Detected"]) * 100

def per_emotion_f1(path):
    d = pd.read_csv(path)
    return {e: f1_score(d["Target"], d["Detected"], labels=[e], average="micro",
                        zero_division=0) for e in EMOTIONS}

# ----------------------------------------------------------------------
# FIGURE 1: full-config accuracy bar chart
# Edit this list as you add new runs. (name, csv_path, color)
# gray = capacity-only, blue = scale, green = steer, red = main hybrid
# ----------------------------------------------------------------------
# Two manifests are in play and they are NOT directly comparable. The original runs
# used a single fixed story stem ("The") for all 160 generations. The revision runs use
# 20 held-out neutral stems times 8 emotions. Mixing them in one undivided axis would
# invite the false reading that top-p (47.50, old) beat greedy (44.38, new). They are
# drawn as two labelled groups, and the old-manifest bars are hatched so the split
# survives greyscale printing.
FIG1_OLD_MANIFEST = [
    ("Med+LoRA r8\n(baseline)",   "evaluation_results_baseline.csv",       "#8c8c8c"),
    ("Med+LoRA r64",              "evaluation_results_optimized.csv",       "#8c8c8c"),
    ("Med+r32\nspecial tok",      "evaluation_results_special.csv",         "#8c8c8c"),
    ("Med+r128\ninstruct",        "evaluation_results_instruction.csv",     "#8c8c8c"),
    ("Large+r32\ninstruct",       "evaluation_results_large_optimized.csv", "#5b9bd5"),
    ("Med+Steer\n(b=5)",          "steered_results.csv",                    "#70ad47"),
    ("Large+Hybrid\n(b=5)",       "hybrid_results.csv",                     "#c00000"),
    ("Large+Aggr\n(b=15)",        "aggressive_steered_results.csv",         "#8c8c8c"),
]

FIG1_NEW_MANIFEST = [
    ("Greedy",          "decoding_greedy_results.csv", "#c00000"),
    ("Top-p (p=.92) = Lexicon 15", "decoding_topp_results.csv", "#70ad47"),
    ("Top-k (k=50)",    "decoding_topk_results.csv",   "#70ad47"),
    ("Lexicon 10",      "lexicon_10_results.csv",      "#5b9bd5"),
    ("Lexicon 5",       "lexicon_5_results.csv",       "#5b9bd5"),
    ("Steer b=1",       "steered_b1_results.csv",      "#8c8c8c"),
]

def fig1():
    old = [(n, acc(p), c) for n, p, c in FIG1_OLD_MANIFEST if os.path.exists(p)]
    new = [(n, acc(p), c) for n, p, c in FIG1_NEW_MANIFEST if os.path.exists(p)]
    gap = 1.2
    x_old = np.arange(len(old))
    x_new = np.arange(len(new)) + len(old) + gap

    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=300)
    for x, (name, value, color) in zip(x_old, old):
        ax.bar(x, value, color=color, edgecolor="black", lw=0.8, width=0.65, hatch="//")
    for x, (name, value, color) in zip(x_new, new):
        ax.bar(x, value, color=color, edgecolor="black", lw=0.8, width=0.65)

    for x, (name, value, color) in list(zip(x_old, old)) + list(zip(x_new, new)):
        ax.text(x, value + 0.6, f"{value:.2f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    all_vals = [v for _, v, _ in old + new]
    top = max(all_vals) + 10
    divider = len(old) + gap / 2 - 0.5
    ax.axvline(divider, color="black", lw=1.2, ls=":", alpha=0.8)
    ax.axhline(12.5, ls="--", color="gray", lw=1, alpha=0.7)
    ax.text(-0.4, 13.2, "random chance (8-class)", ha="left", fontsize=8,
            color="gray", style="italic")

    ax.text(divider - 0.6, top - 2.5, "Original manifest\n(single fixed stem \"The\")",
            ha="right", va="top", fontsize=9.5, style="italic", color="#404040")
    ax.text(divider + 0.6, top - 2.5, "Revision manifest\n(20 held-out neutral stems)",
            ha="left", va="top", fontsize=9.5, style="italic", color="#404040")

    # Rotate: several labels are long enough to collide at this bar spacing.
    ax.set_xticks(list(x_old) + list(x_new))
    ax.set_xticklabels(
        [n.replace("\n", " ") for n, _, _ in old] + [n for n, _, _ in new],
        fontsize=8.5, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("Emotional Controllability Across All Tested Configurations",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, top)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/Fig1_Full_Comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Fig1 saved (two manifests drawn as separate groups)")

def fig2(hybrid="hybrid_results.csv"):
    d = pd.read_csv(hybrid)
    order = ["Joy","Trust","Fear","Surprise","Sadness","Disgust","Anger","Anticipation"]
    present = [e for e in order if e in set(d["Target"]) | set(d["Detected"])]
    if "Neutral" in set(d["Detected"]):
        present = present + ["Neutral"]
    cm = confusion_matrix(d["Target"], d["Detected"], labels=present).astype(float)
    cm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=present,
                yticklabels=present, cbar_kws={"label": "Row-normalized proportion"},
                linewidths=0.5, linecolor="gray", ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Detected Emotion (RoBERTa Judge)", fontsize=12)
    ax.set_ylabel("Target Emotion (Prompt)", fontsize=12)
    ax.set_title("Confusion Matrix: GPT-2 Large + Hybrid Steering",
                 fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUT}/Fig2_Confusion_Matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Fig2 saved")

def fig3():
    base = per_emotion_f1("evaluation_results_baseline.csv")
    steer = per_emotion_f1("steered_results.csv")
    hyb = per_emotion_f1("hybrid_results.csv")
    x = np.arange(len(EMOTIONS)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    ax.bar(x - w, [base[e] for e in EMOTIONS], w, label="Baseline (Med, LoRA r8)",
           color="#a6a6a6", edgecolor="black", lw=0.6)
    ax.bar(x, [steer[e] for e in EMOTIONS], w, label="Med + Steering (b=5)",
           color="#70ad47", edgecolor="black", lw=0.6)
    ax.bar(x + w, [hyb[e] for e in EMOTIONS], w, label="Large + Hybrid (b=5)",
           color="#c00000", edgecolor="black", lw=0.6)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(EMOTIONS, fontsize=10)
    ax.set_title("Per-Emotion F1: Effect of Adding Steering and Scale\n"
                 "(original manifest, single fixed stem)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, frameon=True); ax.set_ylim(0, 0.95)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/Fig3_PerEmotion_F1.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Fig3 saved")


def fig3b():
    """
    Per-emotion F1 on the REVISION manifest only. Kept separate from fig3 rather than
    merged into it: those runs use the original single fixed stem and these use 20
    held-out neutral stems, so grouping them per emotion would imply a comparison the
    data does not support. Everything inside this figure shares one manifest.
    """
    series = [
        ("Greedy (b=5)",        "decoding_greedy_results.csv", "#c00000"),
        ("Top-p (b=5)",         "decoding_topp_results.csv",   "#70ad47"),
        ("Lexicon 5 (b=5)",     "lexicon_5_results.csv",       "#5b9bd5"),
        ("Steer b=1",           "steered_b1_results.csv",      "#a6a6a6"),
    ]
    present = [(n, per_emotion_f1(p), c) for n, p, c in series if os.path.exists(p)]
    if len(present) < 2:
        print("Fig3b skipped (revision CSVs not present yet)")
        return
    x = np.arange(len(EMOTIONS))
    w = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    for i, (name, scores, color) in enumerate(present):
        offset = (i - (len(present) - 1) / 2) * w
        ax.bar(x + offset, [scores[e] for e in EMOTIONS], w, label=name,
               color=color, edgecolor="black", lw=0.6)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(EMOTIONS, fontsize=10)
    ax.set_title("Per-Emotion F1 by Decoding and Lexicon Size\n"
                 "(revision manifest, 20 held-out neutral stems)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9, frameon=True, ncol=2); ax.set_ylim(0, 0.95)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/Fig3b_PerEmotion_F1_Revision.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Fig3b saved")

def fig4():
    # Beta ablation, now on a SINGLE manifest. All three points are measured on the
    # revision manifest (20 held-out neutral stems), so the trend line has one footing
    # and needs no asterisk. Previously b=5 read hybrid_results.csv (47.50, original
    # manifest) and b=15 read aggressive_steered_results.csv (6.67, original manifest),
    # while b=1 was already revision manifest. A line through mixed manifests is exactly
    # the defect the Fig1 split exists to prevent, so it is removed here too.
    #   b=1  steered_b1_results.csv
    #   b=5  decoding_topp_results.csv (identical config to the 47.50% run, new manifest)
    #   b=15 steered_b15_results.csv
    b1 = acc("steered_b1_results.csv")
    b5 = acc("decoding_topp_results.csv")
    b15 = acc("steered_b15_results.csv")
    beta = [1.0, 5.0, 15.0]; accs = [b1, b5, b15]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.plot(beta, accs, "o-", color="#c00000", lw=2.5, markersize=11,
            markerfacecolor="white", markeredgewidth=2.5)
    for b, a in zip(beta, accs):
        ax.annotate(f"{a:.2f}%", (b, a), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Boost Factor (beta)", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.axhline(12.5, ls="--", color="gray", lw=1, alpha=0.7)
    ax.text(15.0, 13.4, "random chance (8-class)", ha="right", fontsize=8,
            color="gray", style="italic")
    ax.set_title("Steering Strength Ablation (GPT-2 Large + Hybrid)\n"
                 "(all points on the revision manifest, 20 held-out neutral stems)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, max(accs) + 10); ax.set_xticks(beta); ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/Fig4_Beta_Ablation.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Fig4 saved")

def fig5():
    # New lightweight baselines vs Hybrid. Only draws bars for files that exist.
    candidates = [
        ("Phi-3 Mini",   "baseline_phi3_results.csv", "#8c8c8c"),
        ("Qwen2.5 1.5B", "baseline_qwen_results.csv", "#8c8c8c"),
        ("PPLM (GPT-2)", "baseline_pplm_results.csv", "#ed7d31"),
        ("Ours: Hybrid", "hybrid_results.csv",        "#c00000"),
    ]
    labels, vals, colors = [], [], []
    for name, path, color in candidates:
        if os.path.exists(path):
            labels.append(name); vals.append(acc(path)); colors.append(color)
    if len(labels) < 2:
        print("Fig5 skipped (new baseline CSVs not present yet)")
        return
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.2f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(12.5, ls="--", color="gray", lw=1, alpha=0.7)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("Our Method vs Newer Lightweight and CTG Baselines",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, max(vals) + 7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/Fig5_New_Baselines.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Fig5 saved")

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig3b(); fig4(); fig5()
    print(f"\nAll figures written to {OUT}/ at 300 DPI.")
