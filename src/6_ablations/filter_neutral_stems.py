"""
filter_neutral_stems.py

Layman note: our story openings are supposed to be emotionally blank, so that any
emotion in the finished story came from our method and not from the opening line.
This script checks that assumption by showing every candidate opening to the same
automatic marker that grades the results, and throwing out any opening that already
carries an emotion.

Why this is needed (the gap it closes):
The held-out pool was labelled Neutral by the teacher, but the teacher scored the
FULL story, never the six-word opening. A long story can average to Neutral while
opening on an air raid siren. Inspecting the first selection showed exactly that:
"The air raid siren howled into", "Blood dripped off the hammer staining",
"Explosions and maniacal laughter - terrifying". Those stems would inflate Fear and
depress Joy and Trust in EVERY run, skewing the per-emotion F1 that feeds Fig3.

We filter the POOL and then select from it, rather than selecting first and patching
rejects. That keeps the selection rule to one clean sentence a reviewer can check:
"20 stems drawn at random, with a fixed seed, from the judge-verified-neutral pool."

Neutrality criterion (BOTH conditions must hold, stated here so it is reproducible):
  1. classify(stem) == "Neutral". The top sigmoid label maps to Neutral under
     EMOTION_MAP. This is the exact function that scores the generated stories, so
     the filter and the evaluator agree by construction.
  2. max P(any Plutchik-mapped emotion) < MAX_EMOTION_PROB. This catches the narrow
     case where Neutral wins at 0.31 while Fear sits at 0.29, which condition 1
     alone would wave through.

Run this BEFORE make_test_prompts.py. It is the only GPU step in the data layer.

Output: data/processed/neutral_stem_pool.csv
        (stem, top_label, neutral_prob, max_emotion_label, max_emotion_prob, passes)

No em dashes anywhere (project style rule).
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import judge
import make_test_prompts as mtp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "neutral_stem_pool.csv")

# Condition 2 cutoff. 0.20 is deliberately conservative: sigmoid outputs on this model
# are not a softmax, so a genuinely blank sentence sits far below this for every emotion.
MAX_EMOTION_PROB = 0.20


def main():
    judge.verify_against_source()
    print("EMOTION_MAP verified identical to 07_evaluate_hybrid.py (the 47.50% run).\n")

    pool, n_held, labels = mtp.build_clean_pool()
    print(f"Held-out pool      {n_held} rows, teacher labels: {', '.join(labels)}")
    print(f"Clean stem pool    {len(pool)} distinct {mtp.STEM_WORDS}-word stems")
    print(f"\nScoring every stem in isolation with {judge.TEACHER_NAME}")
    print(f"Criterion: top label is Neutral AND max emotion prob < {MAX_EMOTION_PROB}\n")

    model, tokenizer, device = judge.load_judge()
    print(f"Judge loaded on {device}\n")

    rows = []
    n_pass = 0
    for index, stem in enumerate(pool):
        profile = judge.emotion_profile(stem, model, tokenizer, device)
        top_ok = profile["top_label"] == "Neutral"
        margin_ok = profile["max_emotion_prob"] < MAX_EMOTION_PROB
        passes = top_ok and margin_ok
        n_pass += int(passes)
        rows.append({
            "stem": stem,
            "top_label": profile["top_label"],
            "neutral_prob": round(profile["neutral_prob"], 6),
            "max_emotion_label": profile["max_emotion_label"],
            "max_emotion_prob": round(profile["max_emotion_prob"], 6),
            "top_is_neutral": int(top_ok),
            "margin_ok": int(margin_ok),
            "passes": int(passes),
        })
        if (index + 1) % 100 == 0:
            print(f"  scored {index + 1}/{len(pool)}, passing so far: {n_pass}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # --- Final summary, so the numbers can be sanity checked before they reach the paper ---
    n_top_only = sum(r["top_is_neutral"] for r in rows)
    n_margin_only = sum(r["margin_ok"] for r in rows)
    print(f"\nWrote {len(rows)} scored stems to {os.path.relpath(OUT_PATH, PROJECT_ROOT)}\n")
    print("Filter sensitivity:")
    print(f"  condition 1 alone (top label Neutral)        {n_top_only:>4} / {len(rows)}")
    print(f"  condition 2 alone (max emotion < {MAX_EMOTION_PROB})     {n_margin_only:>4} / {len(rows)}")
    print(f"  BOTH (the applied filter)                    {n_pass:>4} / {len(rows)}")
    print(f"\n  need 20 stems, headroom factor {n_pass / 20:.1f}x" if n_pass else "\n  WARNING: nothing passed")

    rejected = [r for r in rows if not r["passes"]]
    if rejected:
        rejected.sort(key=lambda r: -r["max_emotion_prob"])
        print("\nMost emotional stems rejected (these are what the filter exists to remove):")
        for r in rejected[:6]:
            print(f"  {r['max_emotion_prob']:.3f} {r['max_emotion_label']:<13} {r['stem']}")

    accepted = [r for r in rows if r["passes"]]
    if accepted:
        accepted.sort(key=lambda r: r["max_emotion_prob"])
        print("\nBlankest stems accepted:")
        for r in accepted[:6]:
            print(f"  {r['max_emotion_prob']:.3f} {r['max_emotion_label']:<13} {r['stem']}")


if __name__ == "__main__":
    main()
