"""
run_decoding_ablation.py

Task A2, reviewer comment 4. Does the decoding strategy change emotional control?

Layman note: there are several ways to pick the next word. Greedy always takes the most
likely one. Top-k and top-p sample from a shortlist. We hold everything else fixed and
change only this, so any difference is caused by the decoding rule alone.

Greedy is deterministic. It is valid at N=160 here only because the manifest now holds
160 distinct prompts (20 held-out stems times 8 emotions). With the original fixed stem
"The" it would have collapsed to 8 distinct stories reported as 160.

Usage:
    python src/6_ablations/run_decoding_ablation.py --decoding topp
    python src/6_ablations/run_decoding_ablation.py --decoding topk
    python src/6_ablations/run_decoding_ablation.py --decoding greedy

No em dashes anywhere (project style rule).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def main():
    parser = argparse.ArgumentParser(description="Decoding strategy ablation")
    parser.add_argument("--decoding", required=True, choices=["greedy", "topk", "topp"],
                        help="topp (p=0.92) is the configuration that produced 47.50%")
    args = parser.parse_args()

    caveat = ("greedy decoding is deterministic, so each of the 160 rows is a distinct "
              "prompt but repeat runs are identical") if args.decoding == "greedy" else ""

    common.execute_run(
        config_name=f"decoding_{args.decoding}",
        output_csv=f"decoding_{args.decoding}_results.csv",
        technique=f"LoRA + lexical steering (b=5.0), {args.decoding} decoding",
        tier="15",
        boost_factor=5.0,
        decoding=args.decoding,
        caveat=caveat,
    )


if __name__ == "__main__":
    main()
