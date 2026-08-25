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
FIG1_CONFIGS = [
    ("Med+LoRA r8\n(baseline)",   "evaluation_results_baseline.csv",       "#8c8c8c"),
    ("Med+LoRA r64",              "evaluation_results_optimized.csv",       "#8c8c8c"),
    ("Med+r32\nspecial tok",      "evaluation_results_special.csv",         "#8c8c8c"),
    ("Med+r128\ninstruct",        "evaluation_results_instruction.csv",     "#8c8c8c"),
    ("Large+r32\ninstruct",       "evaluation_results_large_optimized.csv", "#5b9bd5"),
    ("Med+Steer\n(b=5)",          "steered_results.csv",                    "#70ad47"),
    ("Large+Hybrid\n(b=5)",       "hybrid_results.csv",                     "#c00000"),
    ("Large+Aggr\n(b=15)",        "aggressive_steered_results.csv",         "#8c8c8c"),
]

def fig1():
    labels, vals, colors = [], [], []
    for name, path, color in FIG1_CONFIGS:
        if os.path.exists(path):
            labels.append(name); vals.append(acc(path)); colors.append(color)
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8, width=0.65)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.2f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(12.5, ls="--", color="gray", lw=1, alpha=0.7)
    ax.text(len(labels)-0.6, 13.2, "random chance (8-class)", ha="right",
            fontsize=8, color="gray", style="italic")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("Emotional Controllability Across All Tested Configurations",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, max(vals) + 7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT}/Fig1_Full_Comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Fig1 saved")

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
    ax.set_title("Per-Emotion F1: Effect of Adding Steering and Scale",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, frameon=True); ax.set_ylim(0, 0.95)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/Fig3_PerEmotion_F1.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Fig3 saved")

def fig4():
    # beta ablation. b=1 reverts to ~large baseline, b=5 sweet spot, b=15 collapse.
    beta = [1.0, 5.0, 15.0]; accs = [13.12, 47.50, 6.67]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.plot(beta, accs, "o-", color="#c00000", lw=2.5, markersize=11,
            markerfacecolor="white", markeredgewidth=2.5)
    for b, a in zip(beta, accs):
        ax.annotate(f"{a:.2f}%", (b, a), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Boost Factor (beta)", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("Steering Strength Ablation (GPT-2 Large + Hybrid)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 54); ax.set_xticks(beta); ax.grid(True, alpha=0.3)
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
    fig1(); fig2(); fig3(); fig4(); fig5()
    print(f"\nAll figures written to {OUT}/ at 300 DPI.")
