"""
make_test_prompts.py

Layman note: this writes down, once and for all, the exact list of 160 things we
ask the model to write. Every experiment reads this same list, so any difference
in results comes from the method being tested, not from different questions.

Why the stems are real held-out text and not "The":
The original hybrid run (07_evaluate_hybrid.py, 47.50% top-1) had no test set on
disk. It looped 8 emotions times 20 samples, and all 160 used the identical story
start "The", with no seed set. Verified three ways: the constant is hardcoded at
line 17, the loop variable is discarded and the script reads no file, and all 160
stories in hybrid_results.csv begin with the single word "The".

Identical inputs break the decoding ablation. Greedy decoding is deterministic, so
160 identical prompts would collapse to 8 distinct stories (one per emotion) while
being reported as N=160, overstating the sample size twentyfold.

Design: 20 stems times 8 emotions (paired)
The SAME 20 stems are used for every emotion. All 160 rows are therefore distinct
(stem, emotion) inputs, so greedy produces 160 distinct generations. Because every
emotion sees an identical stem set, an accuracy difference between emotions is
attributable to the emotion conditioning and not to which stems that emotion
happened to draw. Giving each emotion its own stems would confound the two.

Stem provenance (this is what a reviewer will ask about):
Stems come from data/annotated/annotated_train.csv rows that are NOT in
data/processed/final_train_ready.csv, that is the held-out remainder never seen in
training. All 1407 held-out rows carry the teacher label "Neutral", because
02_balance_and_format.py balanced the training set by dropping surplus Neutral
rows. So the stems are real WritingPrompts text, unseen in training, and judged
emotionally neutral by the SAME RoBERTa teacher that scores the outputs. The stem
therefore cannot leak the target emotion into the result.

Output: data/processed/test_prompts_160.csv
        data/processed/test_prompts_160_PROVENANCE.md

No em dashes anywhere (project style rule).
"""
import csv
import os
import random
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANNOTATED = os.path.join(PROJECT_ROOT, "data", "annotated", "annotated_train.csv")
TRAIN_USED = os.path.join(PROJECT_ROOT, "data", "processed", "final_train_ready.csv")
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test_prompts_160.csv")
PROV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test_prompts_160_PROVENANCE.md")
NEUTRAL_POOL_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "neutral_stem_pool.csv")

# Order and counts copied from 07_evaluate_hybrid.py so the layout matches the run
# that produced 47.50%. Disgust is generated but scored as Anger, which is why
# Anger ends up with 40 rows and every other emotion with 20.
EMOTION_ORDER = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation"]
N_STEMS = 20
STEM_WORDS = 6          # short enough not to dilute the emotional signal in 60 new tokens
BASE_SEED = 20260826    # fixed once, never changed, so reruns reproduce exactly

# Characters we allow inside a stem. Anything else means scraped junk.
STEM_ALLOWED = re.compile(r"[A-Za-z0-9 ,'.\-]+")
STEM_REJECT = re.compile(r"http|www\.|reddit|\[\s*prompt\s*\]|\\n", re.I)


def load_held_out():
    """Rows present in the annotated set but absent from the training set."""
    import pandas as pd
    annotated = pd.read_csv(ANNOTATED)
    trained = pd.read_csv(TRAIN_USED)
    used = set(trained["text"].astype(str))
    held = annotated[~annotated["text"].astype(str).isin(used)]
    return held


def is_clean(text):
    """
    Keep only stems that read as a clean sentence opening. We reject encoding
    artefacts, scraped markup and fragments, because a malformed stem would show up
    in the published CSV and invite questions that have nothing to do with the method.
    """
    text = str(text)
    if "�" in text:
        return False
    if STEM_REJECT.search(text):
        return False
    words = text.split()
    if len(words) < STEM_WORDS + 6:
        return False
    stem = " ".join(words[:STEM_WORDS])
    if not STEM_ALLOWED.fullmatch(stem):
        return False
    return stem[0].isupper()


def build_clean_pool():
    """
    The candidate stem pool before any judge scoring. Exposed so that
    filter_neutral_stems.py scores exactly this pool, and nothing drifts between
    what gets scored and what gets selected from.
    """
    held = load_held_out()
    labels = sorted(set(held["emotion_label"].astype(str)))
    clean = held[held["text"].map(is_clean)]
    # sorted() so the pool order does not depend on pandas or file ordering.
    pool = sorted({" ".join(str(t).split()[:STEM_WORDS]) for t in clean["text"]})
    return pool, len(held), labels


def load_neutral_pool():
    """
    Read the judge-verified-neutral stems. filter_neutral_stems.py must run first.
    We filter the POOL and then select from it, rather than selecting first and
    patching rejects, so the selection rule stays one sentence a reviewer can check.
    """
    import pandas as pd
    if not os.path.exists(NEUTRAL_POOL_PATH):
        raise SystemExit(
            "Missing " + os.path.relpath(NEUTRAL_POOL_PATH, PROJECT_ROOT) + "\n"
            "Run this first:  python src/6_ablations/filter_neutral_stems.py\n"
            "That script scores every candidate stem in isolation with the RoBERTa judge, "
            "so the manifest can be built from stems verified to carry no target emotion."
        )
    scored = pd.read_csv(NEUTRAL_POOL_PATH)
    passing = sorted(scored[scored["passes"] == 1]["stem"].astype(str))
    return passing, len(scored)


def select_stems():
    """Deterministically pick N_STEMS stems from the judge-verified-neutral pool."""
    pool, n_held, labels = build_clean_pool()
    neutral, n_scored = load_neutral_pool()
    if len(neutral) < N_STEMS:
        raise SystemExit(
            f"Only {len(neutral)} stems passed the neutrality filter, need {N_STEMS}. "
            "Relax MAX_EMOTION_PROB in filter_neutral_stems.py or widen the quality filter."
        )
    # seeded sample over a sorted list, so this selection is byte reproducible.
    stems = random.Random(BASE_SEED).sample(neutral, N_STEMS)
    return stems, n_held, len(pool), len(neutral), labels


def build_rows(stems):
    rows = []
    prompt_id = 0
    for lexicon_key in EMOTION_ORDER:
        target_label = "Anger" if lexicon_key == "Disgust" else lexicon_key
        for stem_id, stem in enumerate(stems):
            rows.append({
                "prompt_id": prompt_id,
                "lexicon_key": lexicon_key,
                "target_label": target_label,
                "stem_id": stem_id,
                "story_start": stem,
                "prompt_text": f"Emotion: {lexicon_key} | Story: {stem}",
                "seed": BASE_SEED + prompt_id,
            })
            prompt_id += 1
    return rows


def write_provenance(stems, n_held, n_pool, n_neutral, labels):
    lines = []
    lines.append("# Provenance: test_prompts_160.csv\n")
    lines.append("Generated by `src/6_ablations/make_test_prompts.py`. Do not edit by hand.\n")
    lines.append("## Design\n")
    lines.append("20 stems times 8 emotions (paired), giving 160 distinct (stem, emotion) rows.")
    lines.append("Every emotion uses the SAME 20 stems, so accuracy differences between emotions")
    lines.append("reflect the emotion conditioning rather than which stems that emotion drew.\n")
    lines.append("Scored label distribution: Anger 40 (Disgust merged in), all other emotions 20,")
    lines.append("N=160. This matches `hybrid_results.csv` exactly.\n")
    lines.append("## Stem source\n")
    lines.append("| Step | Value |")
    lines.append("|---|---|")
    lines.append("| Source file | `data/annotated/annotated_train.csv` |")
    lines.append("| Excluded | every row appearing in `data/processed/final_train_ready.csv` |")
    lines.append(f"| Held-out pool | {n_held} rows |")
    lines.append(f"| Teacher labels in pool | {', '.join(labels)} |")
    lines.append(f"| After quality filter | {n_pool} distinct {STEM_WORDS}-word stems |")
    lines.append(f"| After neutrality filter | {n_neutral} stems verified neutral by the judge |")
    lines.append(f"| Selected | {N_STEMS}, via `random.Random({BASE_SEED}).sample(sorted(neutral_pool), {N_STEMS})` |\n")
    lines.append("Every held-out row carries the label `Neutral` because `02_balance_and_format.py`")
    lines.append("balanced the training set by dropping surplus Neutral rows. That label applies to the")
    lines.append("FULL story though, not to the six-word opening, so it is not sufficient on its own.")
    lines.append("A long story can average to Neutral while opening on an air raid siren. The stems are")
    lines.append("therefore scored a second time, in isolation, by the neutrality filter below.\n")
    lines.append(f"Quality filter rejects encoding artefacts, URLs, reddit markup, escaped newlines,")
    lines.append(f"stems under {STEM_WORDS + 6} words, stems with characters outside")
    lines.append("`[A-Za-z0-9 ,'.-]`, and stems not starting with a capital.\n")
    lines.append("## Neutrality filter\n")
    lines.append("Applied by `src/6_ablations/filter_neutral_stems.py`, which scores every candidate")
    lines.append("stem in isolation and writes `data/processed/neutral_stem_pool.csv` with the full")
    lines.append("per-stem scores. The pool is filtered first and the selection drawn from it, rather")
    lines.append("than selecting first and patching rejects, so the selection rule stays checkable.\n")
    lines.append("A stem is accepted only if BOTH conditions hold:\n")
    lines.append("1. `classify(stem) == \"Neutral\"`. The top sigmoid label maps to Neutral under")
    lines.append("   `EMOTION_MAP`. This is the exact function that scores the generated stories, so")
    lines.append("   the filter and the evaluator agree by construction.")
    lines.append("2. `max P(any Plutchik-mapped emotion) < 0.20`. This catches the narrow case where")
    lines.append("   Neutral wins at 0.31 while Fear sits at 0.29, which condition 1 alone would pass.\n")
    lines.append(f"Judge: `SamLowe/roberta-base-go_emotions`, mapped to Plutchik-8, seed {BASE_SEED}.\n")
    lines.append("The paper can therefore state, literally and checkably, that the stems themselves")
    lines.append("carry no target-emotion signal.\n")
    lines.append("## Difference from the 47.50% headline run\n")
    lines.append('That run used the fixed stem "The" for all 160 generations. These stems are varied.')
    lines.append("The ablation numbers are therefore internally comparable to each other exactly, and")
    lines.append("comparable to the 47.50% anchor in aggregate only. This needs a footnote in the")
    lines.append("ablation table. Re-running the hybrid config on this manifest would make the two")
    lines.append("directly stackable, which has not been done.\n")
    lines.append("## The 20 stems\n")
    for i, stem in enumerate(stems):
        lines.append(f"{i:>3}. {stem}")
    with open(PROV_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    stems, n_held, n_pool, n_neutral, labels = select_stems()
    rows = build_rows(stems)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_provenance(stems, n_held, n_pool, n_neutral, labels)

    print(f"Held-out pool      {n_held} rows, labels: {', '.join(labels)}")
    print(f"Clean stem pool    {n_pool} distinct {STEM_WORDS}-word stems")
    print(f"Judge-neutral pool {n_neutral} stems passed the neutrality filter")
    print(f"Selected           {len(stems)} stems, seed {BASE_SEED}")
    print(f"\nWrote {len(rows)} prompts to {os.path.relpath(OUT_PATH, PROJECT_ROOT)}")
    print(f"Wrote provenance to {os.path.relpath(PROV_PATH, PROJECT_ROOT)}\n")

    counts = {}
    for row in rows:
        counts[row["target_label"]] = counts.get(row["target_label"], 0) + 1
    print("Rows per SCORED target label:")
    for label in sorted(counts, key=lambda k: -counts[k]):
        print(f"  {label:<14} {counts[label]:>3}")
    print(f"\n  total              {len(rows):>3}")
    print(f"  distinct stems     {len(set(r['story_start'] for r in rows)):>3}")
    print(f"  distinct prompts   {len(set(r['prompt_text'] for r in rows)):>3}  (greedy safe if 160)")
    print(f"  distinct seeds     {len(set(r['seed'] for r in rows)):>3}")


if __name__ == "__main__":
    main()
