# HANDOFF.md  (living work order, updated between sessions)

Read this after CLAUDE.md. This is the current state and next actions. When a section is
done, move it to "Completed" at the bottom. When a decision changes, edit it here.

Last updated: 2026-09-01

## Where we are
- Repo on GitHub (private), tree clean. NOTE: local is AHEAD of origin, the rank-sweep
  commits have not been pushed yet. Push at the end with the finishing work.
- Tasks A, B and C are all DONE and committed (see Completed).
- rank-32 anchor DONE and committed: 41.25%, 7.41 h, peak 5.65 GiB.
- 12 runs are in results/all_configs_summary.csv with per-emotion F1. Figures were last
  regenerated on 2026-08-31 and are now STALE, they do not include rank_32 or the ranks
  still to come.
- Remaining: ranks 8, 64 and 128 (one overnight chain), then figures, the new paper-table
  generator, push, and then writing. These are the LAST experiments in the paper.

## Run log for the sweep so far
    timing probe  r32   39.52 s per optimizer step, 760 steps, estimated 8.35 h
    rank 32             41.25%, actual 7.41 h (probe overestimated by 11%), peak 5.65 GiB
    rank 8              not started, expect about 7.5 h
    rank 64             not started, expect about 7.5 h, predicted peak about 5.96 GiB
    rank 128            not started, expect OOM within minutes (needs about 0.9 GiB more
                        than the 5.65 GiB measured at r32, against about 6.0 GiB usable)

## NEXT ACTION: say "run", and the overnight rank chain starts

Everything is staged and tested. Nothing is running right now. The single command is:

    bash src/6_ablations/run_rank_chain.sh

That trains and evaluates rank 8, then 64, then 128, back to back, committing each rank as
it lands. Expect about 7.5 h per rank (rank-32 took 7.41 h), so roughly 15 h for 8 and 64,
plus minutes for 128 to hit its expected OOM. Launch it in the background, not in a
foreground shell.

State at the time of writing: tree clean, HEAD at the rank-32 commit, GPU idle, 12 of 15
planned runs complete. rank-32 is DONE and committed (41.25%). Ranks 8, 64 and 128 are the
only experiments left in the entire paper.

### What was verified before staging the chain (so it is not merely asserted)
- The OOM bookkeeping path was FORCE-TESTED, not read: it exits 2, leaves
  all_configs_summary.csv byte-identical, and writes a full config.yaml to its own
  experiment folder. The synthetic test artifact was deleted afterwards so no fake OOM
  record sits in experiments/.
- A real gap was found and fixed while doing that. Trainer.__init__ moves the model onto
  the GPU, which is the first large allocation and a likely OOM site at rank 128, and it
  originally sat OUTSIDE the try block. An OOM there would have escaped uncaught with no
  provenance written, which is exactly the failure the guard exists to prevent. Trainer
  construction is now inside the guard.
- The chain deliberately does NOT use "set -e", because rank-128's OOM must not kill the
  bookkeeping or the ranks after it.
- Per-rank commits, so a crash at hour 12 cannot take completed ranks down with it.
- Checkpoints are NOT auto-deleted (about 3.5 GB per rank, disk has room). An unattended
  script should not delete things while nobody is watching. Manual cleanup afterwards:
  rm -rf models/rank_sweep/checkpoints

## The rank sweep itself (see RANK_SWEEP_TASK.md for the full spec)

RANK_SWEEP_TASK.md is the authority on this task. Summary of what changed and why:

REVERSAL, recorded here so the old note cannot mislead later. Earlier versions of this
file said "LoRA rank ablation (8/32/64/128) already exists in prior CSVs; do NOT re-run,
just have the summary script read them." That guidance is OVERTURNED. The old rank CSVs
(evaluation_results_baseline / optimized / special / instruction) are unusable as a sweep:
they have inconsistent N (70, 105, 160, 160), mixed formats, two of them lack the Story
column, and they were produced on the ORIGINAL fixed-stem manifest, not the revision
manifest. They also vary model size and technique alongside rank, so they do not isolate
rank at all. They must NOT be added to all_configs_summary.csv. Rank is being re-run
cleanly on the revision manifest instead.

The sweep varies ONLY rank, over {8, 32, 64, 128}:
- lora_alpha held FIXED at 64 across all four, matching production (r=32, alpha=64).
  The training convention was alpha = 2*rank, so fixed alpha makes effective scaling
  (alpha/rank) swing from 8.0 at rank-8 to 0.5 at rank-128. This is intentional and gets
  stated plainly in the paper: a flat sweep despite 16x scaling variation strengthens the
  finding that neither adapter capacity nor scaling is the operative factor.
- Everything else frozen from 06_train_gpt2_large_optimized.py: same data, seed,
  epochs=10, LR=2e-4, batch=1, grad-accum=32, same target modules and modules_to_save.
- Evaluation identical to Task A via common.py: revision manifest, same RoBERTa judge,
  max_new_tokens=60, top-p decoding (match production, NOT greedy, so rank and decoding
  do not entangle), steering ON at boost=5.0 with the full lexicon.

HYPOTHESIS, recorded before seeing any numbers so interpretation stays honest: with
steering ON, rank should matter little (flat-ish sweep). The paper's thesis is that
steering, not capacity, is the dominant lever. A flat sweep confirms this from a 4th axis
(the others being model size Medium vs Large, the lexicon-size cliff, and decoding). If
rank matters a lot even with steering, that is a surprise to engage honestly, not smooth.

Gates and tolerances:
- rank-32 runs FIRST and ALONE. It must land near 38.12% (production hybrid on the
  revision manifest, the decoding_topp / lexicon_15 value). If it does not, something
  differs from production: STOP and reconcile before training the other three. rank-32
  also times one training run so the real per-rank cost is known before committing to four.
- rank-128 runs LAST and is OOM-tolerant. If it does not fit alongside modules_to_save in
  6GB, report a 3-point sweep (8/32/64) and note "rank-128 exceeded the 6GB budget",
  which is itself on-thesis for a resource-constrained paper. It must not block the others.

Outputs: rank_8_results.csv, rank_32_results.csv, rank_64_results.csv,
rank_128_results.csv (Target, Detected, Story), one experiments/experiment_XXX/ per rank
with full provenance including training time and the alpha/scaling note.

## Then: consolidate, figures, paper table
- Add all four rank rows to results/all_configs_summary.csv with per-emotion F1.
- Regenerate results/figures/ at 300 DPI via make_figures.py. Point FIG1_CONFIGS at the
  CLEAN rank CSVs, not the old ones.
- Regenerate the consolidated paper table ONCE, at the end, from all_configs_summary.csv,
  with the GPT-2 Medium baseline at 10.00% (never 6.67%). The old
  results/comparisons/final_paper_table.csv has been DELETED as a stale pre-revision
  artifact that carried the wrong 6.67%.
- IMPORTANT, found while deleting it: the file was not orphaned, it had a GENERATOR.
  src/5_evaluation/09_generate_paper_tables.py hardcodes all four table rows as literal
  strings, including "6.67%" at line 25, and writes them straight to that path. Deleting
  the CSV alone does not remove the trap, because running that script recreates it with
  the wrong number. That script lives under the frozen "do not edit" original pipeline,
  so it has been left untouched deliberately. Consequence: do NOT run
  09_generate_paper_tables.py again. The end-of-project table gets a NEW generator in
  src/6_ablations/ that reads all_configs_summary.csv instead of hardcoding, so the
  number cannot drift from the runs again.

## THE THREE WRITE-UP POINTS THAT MUST LAND EXACTLY AS FRAMED

These were settled deliberately, two of them before any numbers existed. Do not soften or
re-derive them at writing time.

### 1. The measured retraining noise floor, about 3 points (report this prominently)
rank_32 (41.25%) and decoding_topp / lexicon_15 (38.12%) are INDEPENDENT RETRAININGS of
one configuration, differing only in random init: production never seeded, run_rank_sweep
does. Their 3.13-point difference is a measured retraining variance for this setup, which
is 5 stories out of 160 and 0.81 standard errors (one SE at p=0.4, N=160 is 3.87 points).

This does double duty and reframes the WHOLE results table, not just the rank rows:
  a. it makes the rank sweep interpretable, since any rank gap under about 3 points is
     indistinguishable from retraining the same config; and
  b. it retroactively justifies every "inside noise" call made elsewhere. lexicon_10 at
     38.75 against lexicon_15 at 38.12 is a 0.63-point gap, now MEASURED as negligible
     rather than merely asserted.
Put one line in methods or results establishing the roughly 3-point retraining variance,
then reference it wherever a gap is called negligible. Most papers in this space do not
bother, and it costs nothing because the measurement already exists.

### 2. The two summary rows for one config are deliberate, keep both
all_configs_summary.csv holds rank_32 at 41.25 and decoding_topp at 38.12, which look like
a duplicate somebody forgot to remove. They are not. Keep both, label them as independent
retrainings, and cite them AS the noise-floor measurement. The sentence:
  "rank_32 and the tier-15 / top-p run are independent retrainings of the production
   configuration; their 3.13-point difference establishes the retraining noise floor."
That converts an apparent redundancy into the deliberate measurement it actually is.

### 3. The capacity claim stays NARROW (locked before any numbers existed)
Rank was swept 16x at fixed alpha=64 with lm_head and wte trained in full throughout.
Because those two dominate the parameter count, total trainable capacity varied only 1.66x
(134.5M at r8 to 223.0M at r128). Therefore:
  SUPPORTED:     "LoRA rank specifically does not matter under steering."
  NOT SUPPORTED: "adapter capacity is not the operative factor."
Capacity is never actually made small, so the broader claim would be an overreach a
reviewer could catch with the same arithmetic. Hold this line even if a flat sweep makes
the bigger claim tempting. The capacity-in-general argument is carried by the model-size
result (Medium vs Large), the lexicon cliff and decoding. Keep the two claims in separate
lanes. The paper sentence:
  "Rank was swept 16x at fixed alpha=64 with lm_head and wte trained in full throughout;
   because those two modules dominate the parameter count, total trainable capacity varied
   only 1.66x (134.5M to 223.0M). The sweep therefore isolates the effect of LoRA rank
   specifically, not adapter capacity in general."

## Paper-writing consequences to record later (not code)
- Ablation prompt regime differs from the original 47.50% headline run (which used fixed
  identical stems, all starting with "The"). The ablation table needs a footnote. I chose
  NOT to re-run the hybrid on the varied set, so the footnote does the reconciliation. (If
  I later want ablations directly stackable with the 47.50% anchor, one hybrid run on the
  varied set would be needed. Not now.)
- Provenance sentence the paper can now make (kept literally true by the neutrality filter):
  "we verified the stems themselves carry no target-emotion signal."
- The rank sweep's fixed-alpha choice needs its one honest sentence (see above).
- Old rank CSVs are NOT a rank ablation and must not be presented as one. If any earlier
  draft text implies they are, fix that text.

## Completed

### Step 0: check whether original 47.50% run used varied prompts  (DONE)
Finding: prompts were genuinely identical. Hardcoded constant, discarded loop variable, no
file read anywhere; all 160 stories started with the single word "The". Nothing to recover,
so Step 1 (regenerate manifest) applied.

### Step 1: regenerate manifest from held-out pool  (DONE)
Held-out pool: 1,407 rows never seen in training, all labeled Neutral because
02_balance_and_format.py balanced by dropping the Neutral surplus. Quality filtering left
885 clean stems. Manifest kept the exact label distribution (Anger 40, others 20, N=160),
selected deterministically, verified byte-identical across two runs.

### Step 1b: stem neutrality filter  (DONE, decision: apply the filter)
Problem found: the teacher labeled the full STORY Neutral, never the six-word opening.
Scoring the selected stems in isolation, several read as clearly emotional (air-raid siren,
blood on a hammer, explosions and laughter), which would skew per-emotion F1 in Fig3.
Fix applied: score the 885-stem pool in isolation with the RoBERTa judge, keep only the
judge-verified-neutral stems, then select from that filtered pool (filter the pool THEN
select, not select-then-patch, so the selection rule stays one clean sentence).

THRESHOLD AS ACTUALLY RUN (this answers the earlier note to self). A stem is accepted only
if BOTH conditions hold, and both are recorded in test_prompts_160_PROVENANCE.md:
  1. classify(stem) == "Neutral", that is the top sigmoid label maps to Neutral under
     EMOTION_MAP. This is the exact function that scores the generated stories, so the
     filter and the evaluator agree by construction.
  2. max P(any Plutchik-mapped emotion) < 0.20, which catches the narrow case where Neutral
     wins at 0.31 while Fear sits at 0.29 and condition 1 alone would pass it.
Judge: SamLowe/roberta-base-go_emotions mapped to Plutchik-8. Seed 20260826.

Numbers: 885 scored, 764 pass condition 1 alone, 686 pass condition 2 alone, 686 pass both.
Condition 2 turned out to be strictly stricter on this data (every stem passing 2 also
passed 1), so condition 2 is the binding one. 686 passing gives 34x headroom for 20 stems.
Per-stem scores are committed in data/processed/neutral_stem_pool.csv so the filter is
auditable rather than merely asserted.

Verified: all four originally flagged stems are gone (air raid siren, blood on a hammer,
maniacal laughter, injected the sleeping two year). Manifest still 160 distinct prompts,
20 distinct stems, 160 distinct seeds, label distribution unchanged.

Honest caveat: a few surviving stems still read as eventful to a human ("Five shots. Five
shots rang out", "Captain Rick slammed on the brakes"). The judge scores every emotion
below 0.20 for them. Since the judge is what grades the outputs, agreement with the judge
is the criterion that matters, but do not overclaim these as human-neutral in the paper.

NOTE: kept the paired design at 20 stems times 8 emotions. The line "select 160
deterministically" would mean 160 distinct stems, which is the confounded option we
explicitly rejected. Flagging in case that was intended differently.

### Step 2: src/6_ablations/common.py and the runners  (DONE)
Shared spine built and in use by every runner since: SteeringProcessor loaded live from
07_evaluate_hybrid.py with a SHA256 drift guard, load_judge/classify, load_hybrid_model,
load_prompt_set, ResumableWriter (per-row flush, resume keyed on frozen prompt_id),
write_provenance, summarize/append_to_summary, and the shared execute_run loop.

### Task A: seven ablation runs  (DONE, commits 04dee57 and f3dc1eb)
All 160 rows, revision manifest, Target/Detected/Story, one experiment folder each with
config.yaml provenance. Results in results/all_configs_summary.csv:
  decoding_greedy 44.38 | lexicon_10 38.75 | lexicon_15 38.12 | decoding_topp 38.12
  decoding_topk 36.25 | lexicon_5 22.50 | beta_1 11.88 | beta_15 18.12
b=15 was added on the revision manifest so Fig4 sits on a single footing: fig4 now reads
b=1 from steered_b1_results.csv, b=5 from decoding_topp_results.csv and b=15 from
steered_b15_results.csv, all three on the same manifest, so the beta curve needs no
asterisk. Note the 6.67% b=15 figure in CLAUDE.md refers to the ORIGINAL-manifest
aggressive run (N=120), which is a different run from this one.

### Task B: Phi-3 and Qwen baselines  (DONE, commit f612c4d)
Inference-only, 4-bit, correct chat template each, same 160 prompts, equal 60-token
generation budget. baseline_phi3 25.62, baseline_qwen 21.88. Both lose to steered GPT-2.

### Task C: PPLM baseline  (DONE, commit 97b86e0)
PPLM on gpt2-large using a linear attribute head over frozen hidden states, same 160
prompts. 35.62%, peak 3.67 GiB, 56.9 s/story. It runs in the 6GB budget and lands just
below our control at 6.2x the per-story cost, which is the comparison the paper wants.
GeDi skipped as planned.

### Housekeeping alongside the rank sweep  (DONE)
Deleted results/comparisons/final_paper_table.csv: a stale pre-revision artifact dated
Feb 10 that carried 6.67% for the GPT-2 Medium baseline against the CLAUDE.md mandate of
10.00%. Nothing read it (make_figures.py computes accuracy from the raw result CSVs), and
it was a trap if copied into the manuscript. The consolidated table gets regenerated once
at the end from all_configs_summary.csv.
