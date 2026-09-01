# Task: clean LoRA-rank sweep (final experiment, reviewer comment 4, 4th angle)

Read CLAUDE.md and HANDOFF.md first, then this. IMPORTANT: HANDOFF.md is stale on this
point. It says "do NOT re-run rank, just read existing CSVs." That guidance is OVERTURNED.
The old rank CSVs (N 70/105/160/160, mixed formats, 2 missing Story, original manifest) are
unusable as a sweep and must NOT be added to the summary. Re-run rank cleanly on the
revision manifest instead. Update HANDOFF.md to reflect this before starting.

## Two housekeeping fixes first (do before the sweep)
1. DELETE results/comparisons/final_paper_table.csv. It is a stale pre-revision artifact
   (dated Feb 10, carries 6.67% for the GPT-2 Medium baseline; CLAUDE.md mandates 10.00%).
   Nothing reads it. It is a trap if copied into the manuscript. The consolidated paper
   table will be regenerated ONCE at the very end from all_configs_summary.csv at 10.00%.
2. Do NOT add the old rank CSVs to all_configs_summary.csv. The clean re-run replaces them.

## The sweep: vary ONLY rank
Rank in {8, 32, 64, 128}. Everything else identical across all four.

alpha strategy (RESOLVED, do not change): hold lora_alpha = 64 FIXED across all four ranks,
matching production (models/gpt2_large_optimized used r=32, alpha=64). Consequence to state
plainly in the paper: the training convention was alpha = 2*rank, so holding alpha=64 fixed
makes effective scaling (alpha/rank) swing from 8.0 at rank-8 to 0.5 at rank-128. This is
INTENTIONAL and reportable: "rank swept at fixed alpha=64 matching production; effective
scaling therefore co-varies. A flat sweep despite 16x scaling variation strengthens the
finding that neither adapter capacity nor scaling is the operative factor."

FREEZE identical across all four (copy from 06_train_gpt2_large_optimized.py):
- training data (same set), seed, epochs=10, LR=2e-4, batch=1, grad-accum=32
- base model GPT-2 Large, same modules_to_save, same target modules
- everything except r

EVALUATE identical to the Task A pipeline (via common.py):
- revision manifest (test_prompts_160.csv), same 160 stems
- same RoBERTa judge / classify()
- max_new_tokens=60
- top-p decoding (match production; NOT greedy, so rank and decoding do not entangle)
- STEERING ON (production hybrid config, boost=5.0, full lexicon)

## Anchor check (CRITICAL, gates the other three)
Run rank-32 FIRST, alone. It must land near 38.12% (production hybrid on revision manifest).
If rank-32-steered does NOT come out near 38.12%, something differs from production, STOP
and reconcile before training the other three. Also use rank-32 to TIME one training run
so you know the real per-rank cost before committing to four.

## rank-128 OOM tolerance
Run 128 LAST. It may not fit alongside modules_to_save at 6GB. If it OOMs, that is not a
blocker: report a 3-point sweep (8/32/64) and note "rank-128 exceeded the 6GB budget," which
is itself on-thesis for a resource-constrained paper. Do not let 128 block the other three.

## Hypothesis to state BEFORE seeing numbers (keeps interpretation honest)
With steering ON, rank is expected to matter little (flat-ish sweep). The paper's thesis is
that steering, not capacity, is the dominant lever. A flat sweep CONFIRMS this from a 4th
axis (others: model size Medium vs Large, lexicon-size cliff, decoding). If rank matters a
lot even with steering, that is a surprise that must be engaged honestly, not smoothed.

## Outputs
rank_8_results.csv, rank_32_results.csv, rank_64_results.csv, rank_128_results.csv
([Target, Detected, Story]). One experiments/experiment_XXX/ per rank with full provenance
(seed, commit hash, git_tree_clean, versions, hardware, the alpha/scaling note, training
time). Resume wired in (this is training, long, wire the reuse path as with PPLM/b=15).

## Sequence
1. Update HANDOFF.md (mark A/B/C done, this sweep as the active final task).
2. Delete the stale final_paper_table.csv.
3. Confirm tree clean, note commit hash.
4. Write the rank training+eval runner (reuse common.py; adapt training from
   06_train_gpt2_large_optimized.py). Confirm tree clean and commit the runner BEFORE
   training, so every rank's config.yaml carries an honest committed hash.
5. Train + eval rank-32 FIRST. Report: anchor near 38.12%? training time? Then I approve
   the other three.
6. Train + eval 8, 64, then 128 (OOM-tolerant).
7. Add all four rows to all_configs_summary.csv with per-emotion F1.
8. Regenerate figures (make_figures.py) so Fig1 and any rank figure include the clean rows.
   Update FIG1_CONFIGS to the clean rank CSVs, not the old ones.
9. Regenerate the consolidated paper table ONCE at 10.00% from the summary.
10. Commit and push everything.

## Report back after rank-32 (step 5), then again after the full sweep:
- rank-32 anchor value (near 38.12%?) and per-rank training time
- final four top-1 values + per-emotion F1
- whether the sweep is flat (thesis confirmed) or not (engage honestly)
- whether rank-128 fit or OOMed

Start at step 1. STOP after step 5 (rank-32 anchor) and report before training the rest.
