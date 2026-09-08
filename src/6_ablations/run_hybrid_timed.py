"""
run_hybrid_timed.py

Measures the per-story latency of our own method, so the PPLM cost comparison has a
measured denominator instead of an asserted one.

Layman note: we already knew how long PPLM takes per story (56.9 seconds). We had been
saying PPLM is "6.2x more expensive than ours" without ever having timed ours. This run
times ours, on exactly the same settings that produced our headline numbers, so the
comparison is real.

WHY THIS EXISTS. experiment_013 (PPLM) was the ONLY run of thirteen that recorded timing,
because execute_run never measured it. The hybrid denominator existed nowhere in the repo.
56.9 / 6.2 implies about 9.2 s/story, a number nothing in this project ever produced. The
ratio is now computed from two measurements taken the same afternoon on the same machine
state, rather than one measurement and one number of unknown origin.

APPLES TO APPLES. This is the EXACT production decoding_topp configuration: GPT-2 Large
plus the production adapter, steering on at boost 5.0, the full tier-15 lexicon, the
160-stem revision manifest, max_new_tokens=60, top-p sampling. The only difference from
the existing decoding_topp run is that this one records mean_seconds_per_story. If any
generation parameter differed, the ratio would not be clean.

Timing excludes the judge on purpose. PPLM's 56.9 s/story measures generation only, so
folding judging time into our denominator would understate the ratio.

FREE SANITY CHECK. Because this is the identical configuration to decoding_topp (38.12%)
and rank_32 (41.25%), its accuracy is a THIRD independent draw of the production recipe.
It must land in the 38 to 41 band. If it does, the retraining noise floor gains a third
point. If it lands well outside, something is wrong with this run and its latency should
not be trusted.

Usage:
    python src/6_ablations/run_hybrid_timed.py

No em dashes anywhere (project style rule).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

BAND = (38.0, 41.5)


def main():
    metrics = common.execute_run(
        config_name="hybrid_timed",
        output_csv="hybrid_timed_results.csv",
        technique="LoRA + lexical steering (b=5.0), top-p, production config, latency timed",
        tier="15",
        boost_factor=5.0,
        decoding="topp",
        model_label="GPT-2 Large + LoRA r32",
        caveat=(
            "identical configuration to decoding_topp and rank_32, run to measure "
            "per-story latency. Its accuracy is a THIRD independent draw of the "
            "production recipe and so extends the retraining noise-floor measurement."
        ),
        extra_provenance={
            "purpose": (
                "measure the hybrid's per-story generation latency, to supply the "
                "denominator for the PPLM cost comparison"
            ),
            "timing_scope": (
                "generation only, judge excluded, matching how PPLM's "
                "mean_seconds_per_story was measured"
            ),
            "comparison_numerator": (
                "PPLM 56.9 s/story from experiments/experiment_013_baseline_pplm/"
                "metrics.json, re-confirmed on the same machine state"
            ),
        },
    )

    accuracy = metrics["top1_accuracy"]
    latency = metrics.get("mean_seconds_per_story")

    print("\n--- sanity check: third draw of the production config ---")
    print(f"  decoding_topp  38.12%")
    print(f"  rank_32        41.25%")
    print(f"  this run       {accuracy:.2f}%")
    if BAND[0] <= accuracy <= BAND[1]:
        print(f"  IN BAND ({BAND[0]} to {BAND[1]}). Latency is trustworthy and the noise")
        print("  floor now rests on three independent retrainings.")
    else:
        print(f"  OUT OF BAND ({BAND[0]} to {BAND[1]}). Investigate before trusting this")
        print("  run's latency: an accuracy this far off suggests the configuration")
        print("  differs from production in some way that would also affect timing.")

    if latency:
        print(f"\n--- latency ---")
        print(f"  hybrid  {latency:.2f} s per story (measured here)")
        print(f"  PPLM    56.90 s per story (experiment_013, re-confirm on this machine)")
        print(f"  ratio   {56.9 / latency:.2f}x")
        print("\n  Report the measured ratio, never the previously asserted 6.2x.")


if __name__ == "__main__":
    main()
