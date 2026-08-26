"""
make_test_prompts.py

Layman note: this writes down, once and for all, the exact list of 160 things we
ask the model to write. Every experiment from now on reads this same list, so any
difference in results comes from the method being tested, not from a different set
of questions.

Why this file has to exist:
The original hybrid run (07_evaluate_hybrid.py, 47.50% top-1) did not have a test
set. It looped 8 emotions times 20 samples, and all 160 used the identical story
start "The". The 160 differed only by sampling randomness, and no seed was set.
So there was nothing on disk to reuse and nothing to make a rerun reproducible.

This script freezes that same design (same 8 emotions, same 20 samples, same "The"
story start) and adds one deterministic seed per row. New runs become exactly
reproducible. The original 47.50% run stays comparable in aggregate, but it is NOT
seed matched to these rows and we never claim that it is.

Output: data/processed/test_prompts_160.csv

No em dashes anywhere (project style rule).
"""
import csv
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test_prompts_160.csv")

# Order and counts copied from 07_evaluate_hybrid.py so the layout matches the
# run that produced 47.50%. Disgust is generated but scored as Anger, which is
# why Anger ends up with 40 rows and every other emotion with 20.
EMOTION_ORDER = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation"]
SAMPLES_PER_EMOTION = 20
STORY_START = "The"      # neutral single token, exactly as in the 47.50% run
BASE_SEED = 20260826     # fixed once, never changed, so reruns reproduce exactly


def build_rows():
    rows = []
    prompt_id = 0
    for lexicon_key in EMOTION_ORDER:
        # Disgust was merged into Anger for scoring (8 generated, 7 scored classes).
        target_label = "Anger" if lexicon_key == "Disgust" else lexicon_key
        for replicate in range(SAMPLES_PER_EMOTION):
            rows.append({
                "prompt_id": prompt_id,
                "lexicon_key": lexicon_key,
                "target_label": target_label,
                "replicate": replicate,
                "story_start": STORY_START,
                "prompt_text": f"Emotion: {lexicon_key} | Story: {STORY_START}",
                "seed": BASE_SEED + prompt_id,
            })
            prompt_id += 1
    return rows


def main():
    rows = build_rows()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Final summary so the numbers can be sanity checked before they reach the paper.
    print(f"Wrote {len(rows)} prompts to {os.path.relpath(OUT_PATH, PROJECT_ROOT)}\n")
    counts = {}
    for row in rows:
        counts[row["target_label"]] = counts.get(row["target_label"], 0) + 1
    print("Rows per SCORED target label:")
    for label in sorted(counts, key=lambda k: -counts[k]):
        print(f"  {label:<14} {counts[label]:>3}")
    print(f"\n  total          {len(rows):>3}")
    print(f"  distinct seeds {len(set(r['seed'] for r in rows)):>3}")
    print(f"  story start    '{STORY_START}' for all rows, matching the 47.50% run")


if __name__ == "__main__":
    main()
