"""
lexicon_tiers.py

Layman note: our method nudges the model towards a small list of emotion words.
This file answers "does the size of that word list matter?" by building smaller
versions of the SAME list, then letting us run each one.

Technical note: PRODUCTION_LEXICON below is a verbatim copy of the EMOTION_LEXICON
in src/5_evaluation/07_evaluate_hybrid.py, the run that produced 47.50% top-1.
It is copied rather than imported because directory names starting with a digit
(5_evaluation) are not importable as Python modules. verify_against_source()
guards that copy against drift, and every runner calls it at startup.

Subset rule: tiers take the FIRST N words in the published order. We do not
re-rank or hand-pick. A cherry-picked subset would invite the objection that we
kept only the words that happened to work.

No em dashes anywhere (project style rule).
"""
import os
import re

# --- The production lexicon, exactly as used in the 47.50% hybrid run ---
PRODUCTION_LEXICON = {
    "Joy": ["smile", "laugh", "happy", "delight", "wonderful", "bright", "cheer", "excited", "love", "joy", "grin", "warmth", "beautiful", "sun", "light"],
    "Trust": ["friend", "trust", "believe", "safe", "rely", "honor", "truth", "accept", "help", "support", "agree", "faith", "promise", "steady"],
    "Fear": ["dark", "scared", "afraid", "shiver", "scream", "run", "terror", "shadow", "cold", "panic", "threat", "nervous", "hide", "creep", "blood"],
    "Surprise": ["shock", "gasp", "sudden", "unexpected", "stunned", "wow", "flash", "blink", "amazed", "jolt", "startle", "wide", "abrupt"],
    "Sadness": ["cry", "tear", "loss", "grief", "sob", "sad", "alone", "miss", "pain", "hurt", "empty", "sorry", "dark", "weep", "broken"],
    "Disgust": ["gross", "nasty", "sick", "vile", "foul", "ugly", "hate", "repulse", "dirty", "rotten", "trash", "slime", "smell", "puke"],
    "Anger": ["rage", "furious", "hate", "yell", "shout", "mad", "angry", "fist", "hit", "burn", "fight", "stupid", "annoy", "kill", "slam"],
    "Anticipation": ["wait", "hope", "ready", "soon", "plan", "wish", "dream", "expect", "watch", "look", "forward", "prepare", "future"],
}

# Per-emotion counts are NOT uniform. They range from 13 (Surprise, Anticipation)
# to 15 (Joy, Fear, Sadness, Anger), mean 14.25. The paper must say
# "13 to 15 words per emotion", not "15 words per emotion".
TIER_FULL = "15"   # keeps the filename the spec asks for (lexicon_15_results.csv)
TIER_10 = "10"
TIER_5 = "5"
TIERS = [TIER_FULL, TIER_10, TIER_5]


def get_lexicon(tier):
    """Return the lexicon for a tier. 'full'/'15' is the untouched production list."""
    tier = str(tier)
    if tier in (TIER_FULL, "full"):
        return {e: list(w) for e, w in PRODUCTION_LEXICON.items()}
    n = int(tier)
    return {e: list(w[:n]) for e, w in PRODUCTION_LEXICON.items()}


def tier_word_counts(tier):
    """Per-emotion word counts for a tier, for logging into config.yaml."""
    return {e: len(w) for e, w in get_lexicon(tier).items()}


def verify_against_source(project_root=None):
    """
    Guard against drift: re-parse EMOTION_LEXICON out of 07_evaluate_hybrid.py and
    confirm our copy still matches it word for word. Raises if they diverge, so a
    silent edit to either file can never quietly change what the ablation means.
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source = os.path.join(project_root, "src", "5_evaluation", "07_evaluate_hybrid.py")
    text = open(source, encoding="utf-8").read()
    body = re.search(r"EMOTION_LEXICON = \{(.*?)\n\}", text, re.S)
    if not body:
        raise RuntimeError(f"Could not locate EMOTION_LEXICON in {source}")
    namespace = {}
    exec("SOURCE_LEXICON = {" + body.group(1) + "\n}", namespace)
    source_lexicon = namespace["SOURCE_LEXICON"]
    if source_lexicon != PRODUCTION_LEXICON:
        raise RuntimeError(
            "PRODUCTION_LEXICON has drifted from 07_evaluate_hybrid.py. "
            "The ablation would no longer be measured against the 47.50% config. "
            "Reconcile the two before running."
        )
    return True


if __name__ == "__main__":
    verify_against_source()
    print("Lexicon verified identical to 07_evaluate_hybrid.py (the 47.50% run).\n")
    print(f"{'Emotion':<14} {'full':>6} {'10':>6} {'5':>6}")
    print("-" * 36)
    for emotion in PRODUCTION_LEXICON:
        print(f"{emotion:<14} {len(get_lexicon('15')[emotion]):>6} "
              f"{len(get_lexicon('10')[emotion]):>6} {len(get_lexicon('5')[emotion]):>6}")
    total_full = sum(tier_word_counts('15').values())
    print("-" * 36)
    print(f"{'TOTAL':<14} {total_full:>6} {sum(tier_word_counts('10').values()):>6} "
          f"{sum(tier_word_counts('5').values()):>6}")
    print(f"\nFull tier is 13 to 15 words per emotion (mean {total_full / 8:.2f}), not a uniform 15.")
