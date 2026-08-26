# Revision experiments (Tasks A, A.5, B, C, D)

Code for the SNCS-D-26-04624 revision. Everything here is additive. Nothing in
`src/1_data_ingestion` through `src/5_evaluation` is modified, and GPT-2 remains
the central fine-tuned model.

## Why a frozen prompt manifest exists

The original hybrid run (`src/5_evaluation/07_evaluate_hybrid.py`, 47.50% top-1)
had no test set on disk. It looped 8 emotions times 20 samples, all 160 using the
identical story start `"The"`, with no seed set. The 160 differed only by sampling
randomness.

`data/processed/test_prompts_160.csv` freezes that same design and adds one
deterministic seed per row. Every run below reads it. Results are therefore
comparable to each other exactly, and comparable to the original 47.50% run in
aggregate only. We never claim the original was seed matched to these rows.

## Files

| File | Purpose |
|---|---|
| `make_test_prompts.py` | Writes the frozen 160-row manifest. Run once. |
| `lexicon_tiers.py` | Lexicon subsets, plus a drift check against `07_evaluate_hybrid.py`. |
| `common.py` | Shared steering processor, RoBERTa judge, model loading, crash-safe writer. |
| `run_lexicon_ablation.py` | Task A1. `--tier 15\|10\|5` |
| `run_decoding_ablation.py` | Task A2. `--decoding greedy\|topk\|topp` |
| `run_beta_ablation.py` | Task A.5. `--beta 1.0` |

## Two reference implementations exist, we use the second

`src/4_generation/02_steered_inference.py` has `LexicalSteeringProcessor`
(smaller lexicon, no capitalised variants, boost 3.0, story start
`"The door opened slowly,"`).

`src/5_evaluation/07_evaluate_hybrid.py` has `SteeringProcessor` (full lexicon,
capitalised variants included, boost 5.0, story start `"The"`).

The 47.50% headline came from the second. `common.py` therefore canonicalises on
`SteeringProcessor`, so the ablations are measured against the published config.

## Lexicon sizes are not uniform

The production lexicon holds 13 to 15 words per emotion (mean 14.25), not 15:

| Emotion | Words | | Emotion | Words |
|---|---|---|---|---|
| Joy | 15 | | Sadness | 15 |
| Fear | 15 | | Anger | 15 |
| Trust | 14 | | Disgust | 14 |
| Surprise | 13 | | Anticipation | 13 |

The `15` tier is the untouched production list. The `10` and `5` tiers take the
first N words in the published order, no re-ranking and no hand-picking, so the
subset cannot be accused of keeping only the words that worked. The paper should
say "13 to 15 words per emotion".

## Provenance

Each run writes `experiments/experiment_XXX/` containing `config.yaml` (seed, git
commit, model and adapter paths, decoding params, exact word list, hardware,
CUDA and package versions, run command, timestamp), `partial_results.csv`
(flushed per row so a crash never loses a run), `results.csv` in the
`[Target, Detected, Story]` schema, `metrics.json`, and `run.log`.
