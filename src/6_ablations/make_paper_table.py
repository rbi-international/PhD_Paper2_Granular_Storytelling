"""
make_paper_table.py

Builds the consolidated results table for the paper FROM results/all_configs_summary.csv.

Layman note: this reads the numbers that the experiments actually produced and formats
them into one table. It never contains a hand-typed accuracy, so the table cannot drift
away from the runs.

WHY THIS EXISTS (read before reaching for the old script). The original generator,
src/5_evaluation/09_generate_paper_tables.py, hardcoded every value as a literal string,
including "6.67%" for the GPT-2 Medium baseline where the correct figure is 10.00%. It
wrote straight to results/comparisons/final_paper_table.csv. That stale file was deleted,
but deleting the output does not remove the trap, because re-running that script recreates
it with the wrong number. That script sits under the frozen "do not edit" pipeline, so it
was deliberately left untouched. Do NOT run it. Run this instead.

The baseline row is the one number that cannot come from the summary, because the GPT-2
Medium baseline predates the ablation harness. It is stated once here, at 10.00% and N=70
per CLAUDE.md, with the 6.67% error recorded next to it so nobody reintroduces it.

Usage:
    python src/6_ablations/make_paper_table.py

No em dashes anywhere (project style rule).
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

OUT_DIR = os.path.join(common.PROJECT_ROOT, "results", "comparisons")
OUT_PATH = os.path.join(OUT_DIR, "final_paper_table.csv")

# Historical rows that predate the ablation harness, so they are not in the summary.
# CLAUDE.md is the authority on these. The baseline is 10.00% at N=70. The submitted
# paper said 6.67%, which was wrong and unsupported; the only correct use of 6.67% is the
# aggressive-steering b=15 collapse run at N=120 on the ORIGINAL manifest.
LEGACY_ROWS = [
    {
        "config_name": "baseline_medium_r8",
        "display": "GPT-2 Medium (baseline)",
        "params": "355M",
        "technique": "LoRA r8, no steering",
        "N": 70,
        "top1_accuracy": 10.00,
        "manifest": "original (fixed stem)",
        "note": "corrected from the 6.67% printed in the rejected submission",
    },
    {
        "config_name": "hybrid_original_manifest",
        "display": "GPT-2 Large + Hybrid steering",
        "params": "774M",
        "technique": "LoRA r32 + instruction + lexical steering (b=5.0)",
        "N": 160,
        "top1_accuracy": 47.50,
        "manifest": "original (fixed stem)",
        "note": "headline result of the original submission, fixed identical stems",
    },
]

# How the summary's config_name maps into the paper table, and in what order.
DISPLAY = [
    ("rank_32",         "GPT-2 Large + Hybrid (revision manifest)", "774M",
     "LoRA r32 + lexical steering (b=5.0)"),
    ("decoding_greedy", "  greedy decoding",       "774M", "LoRA r32 + steering, greedy"),
    ("decoding_topp",   "  top-p decoding",        "774M", "LoRA r32 + steering, top-p"),
    ("decoding_topk",   "  top-k decoding",        "774M", "LoRA r32 + steering, top-k"),
    ("lexicon_15",      "  lexicon tier 15",       "774M", "LoRA r32 + steering, 15 words"),
    ("lexicon_10",      "  lexicon tier 10",       "774M", "LoRA r32 + steering, 10 words"),
    ("lexicon_5",       "  lexicon tier 5",        "774M", "LoRA r32 + steering, 5 words"),
    ("rank_8",          "  LoRA rank 8",           "774M", "LoRA r8 + steering (b=5.0)"),
    ("beta_1",          "  steering b=1",          "774M", "LoRA r32 + steering, b=1"),
    ("beta_15",         "  steering b=15",         "774M", "LoRA r32 + steering, b=15"),
    ("baseline_pplm",   "PPLM (GPT-2 Large)",      "774M", "gradient-based CTG"),
    ("baseline_phi3",   "Phi-3-mini-4k-instruct",  "3.8B", "prompting only, 4-bit"),
    ("baseline_qwen",   "Qwen2.5-1.5B-Instruct",   "1.5B", "prompting only, 4-bit"),
]

FIELDS = ["Model", "Parameters", "Technique", "N", "Top-1 Accuracy",
          "Joy F1", "Fear F1", "Sadness F1", "Anger F1",
          "Trust F1", "Surprise F1", "Anticipation F1", "Manifest", "Note"]


def read_summary():
    if not os.path.exists(common.SUMMARY_PATH):
        raise SystemExit(f"Missing {common.SUMMARY_PATH}. Run the experiments first.")
    with open(common.SUMMARY_PATH, newline="", encoding="utf-8") as handle:
        return {row["config_name"]: row for row in csv.DictReader(handle)}


def build_rows():
    summary = read_summary()
    rows, missing = [], []

    for legacy in LEGACY_ROWS:
        row = {key: "" for key in FIELDS}
        row.update({
            "Model": legacy["display"],
            "Parameters": legacy["params"],
            "Technique": legacy["technique"],
            "N": legacy["N"],
            "Top-1 Accuracy": f"{legacy['top1_accuracy']:.2f}%",
            "Manifest": legacy["manifest"],
            "Note": legacy["note"],
        })
        rows.append(row)

    for config_name, display, params, technique in DISPLAY:
        source = summary.get(config_name)
        if source is None:
            missing.append(config_name)
            continue
        row = {
            "Model": display,
            "Parameters": params,
            "Technique": technique,
            "N": source["N"],
            "Top-1 Accuracy": f"{float(source['top1_accuracy']):.2f}%",
            "Manifest": "revision (20 neutral stems)",
            "Note": source.get("caveat", "")[:120],
        }
        for emotion in common.EMOTIONS:
            row[f"{emotion} F1"] = f"{float(source[f'f1_{emotion}']):.4f}"
        rows.append(row)

    return rows, missing


def main():
    rows, missing = build_rows()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})

    print(f"--- consolidated paper table, {len(rows)} rows ---")
    for row in rows:
        print(f"  {row['Model']:<44} {row['Top-1 Accuracy']:>8}  N={row['N']}")
    if missing:
        print("\n  not in the summary yet, so omitted rather than invented:")
        for name in missing:
            print(f"    {name}")
    print(f"\n  written to {os.path.relpath(OUT_PATH, common.PROJECT_ROOT)}")
    print("  Every number above was read from all_configs_summary.csv, except the two")
    print("  legacy rows, which are stated once in this file per CLAUDE.md.")
    print("  Baseline is 10.00%, never 6.67%.")


if __name__ == "__main__":
    main()
