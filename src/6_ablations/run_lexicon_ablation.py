"""
run_lexicon_ablation.py

Task A1, reviewer comment 4. Does the SIZE of the emotion word list matter?

Layman note: we shrink the list of emotion words the model is nudged towards, and see
whether control gets worse. If a 5-word list works as well as the full one, the method
is cheaper than we claimed. If it collapses, list size is a real design parameter.

The tiers are strict subsets of the published list, taking the first N words in the
published order. No new words are invented, so nobody can argue we kept only the words
that happened to work.

Usage:
    python src/6_ablations/run_lexicon_ablation.py --tier 15
    python src/6_ablations/run_lexicon_ablation.py --tier 10
    python src/6_ablations/run_lexicon_ablation.py --tier 5

No em dashes anywhere (project style rule).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def main():
    parser = argparse.ArgumentParser(description="Lexicon size ablation")
    parser.add_argument("--tier", required=True, choices=["15", "10", "5"],
                        help="15 is the full published lexicon (13 to 15 words per emotion)")
    args = parser.parse_args()

    caveat = ("tier 15 is the full published lexicon, which holds 13 to 15 words per "
              "emotion (mean 14.25), not a uniform 15") if args.tier == "15" else ""

    common.execute_run(
        config_name=f"lexicon_{args.tier}",
        output_csv=f"lexicon_{args.tier}_results.csv",
        technique=f"LoRA + lexical steering (b=5.0), lexicon tier {args.tier}",
        tier=args.tier,
        boost_factor=5.0,
        decoding="topp",
        caveat=caveat,
    )


if __name__ == "__main__":
    main()
