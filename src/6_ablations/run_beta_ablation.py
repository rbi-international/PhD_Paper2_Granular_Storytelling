"""
run_beta_ablation.py

Task A.5. Measure the low end of the steering strength curve instead of assuming it.

Layman note: the boost factor controls how hard we push the model towards emotion words.
At b=5 it works (47.50%). At b=15 it breaks the writing (6.67%). The b=1 point was never
measured; the figure borrowed the unsteered model's 13.12% as a stand-in. This run
replaces that guess with a real number.

make_figures.py fig4() already reads steered_b1_results.csv automatically when present.

Usage:
    python src/6_ablations/run_beta_ablation.py --beta 1.0

No em dashes anywhere (project style rule).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def main():
    parser = argparse.ArgumentParser(description="Steering strength (beta) ablation")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="logit boost applied to emotion tokens")
    args = parser.parse_args()

    label = f"{args.beta:g}".replace(".", "p")
    output = "steered_b1_results.csv" if args.beta == 1.0 else f"steered_b{label}_results.csv"

    common.execute_run(
        config_name=f"beta_{label}",
        output_csv=output,
        technique=f"LoRA + lexical steering (b={args.beta:g})",
        tier="15",
        boost_factor=args.beta,
        decoding="topp",
        caveat="replaces the hardcoded 13.12 proxy previously used for b=1 in fig4" if args.beta == 1.0 else "",
    )


if __name__ == "__main__":
    main()
