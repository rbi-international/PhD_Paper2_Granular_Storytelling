# HANDOFF.md  (living work order, updated between sessions)

Read this after CLAUDE.md. This is the current state and next actions. When a section is
done, move it to "Completed" at the bottom. When a decision changes, edit it here.

Last updated: 2026-09-07

## Where we are
ALL EXPERIMENTS ARE FINISHED. Nothing is running. The next phase is writing, not code.

- Tasks A, B and C all done and committed. The LoRA-rank sweep is done as a TWO-POINT
  sweep (r8 and r32) plus a measured compute envelope for r64 and r128.
- results/all_configs_summary.csv holds 13 runs with per-emotion F1 throughout.
- All seven figures regenerated at 300 DPI, including the new Fig6 rank sweep.
- results/comparisons/final_paper_table.csv regenerated from the summary, baseline at
  10.00%, by a NEW generator that reads the summary instead of hardcoding.

## THE ONE CORRECTION THAT MUST SURVIVE INTO THE PAPER

**THE RANK SWEEP IS NOT FLAT. DO NOT WRITE THAT IT IS FLAT.**

We pre-registered the prediction, before any numbers existed, that the sweep would be flat
under steering, and that flatness would be a fourth confirmation of "capacity is not the
lever". THE DATA DISAGREED. We report what the data says.

    rank 8    35.62%
    rank 32   41.25%
    gap       5.63 points, against a MEASURED retraining noise floor of 3.13 points

The gap clears the noise floor, so the direction is real: rank-32 is genuinely better than
rank-8. Per-emotion F1 moves the same direction on all seven emotions, which is consistent
with a real effect rather than a lucky draw on one class.

We agreed in advance, in RANK_SWEEP_TASK.md, that "if rank matters a lot even with
steering, that is a surprise that must be engaged honestly, not smoothed." This is that
case. Pretending otherwise would be the single dishonest move in an otherwise clean
revision, and it would be caught by anyone who reads our own noise floor.

### The sentence to write instead
Not "rank does not matter (flat sweep)" but:

  "LoRA rank has a measurable effect (approximately 5 to 6 points between r8 and r32,
   exceeding the measured retraining noise floor of about 3 points), which is nonetheless
   small relative to the approximately 27-point effect of steering (11.88% at b=1 versus
   38 to 41% at b=5). Adapter capacity within the feasible range is a secondary factor;
   steering is the primary one."

Steering's effect is roughly 5x the rank effect. Rank mattering somewhat does not dent
"steering dominates". The nuanced version is more credible than the convenient one: a
reviewer trusts a paper that reports "our own ablation shows rank matters somewhat, but
less than our main mechanism" far more than one where everything confirms the thesis.

### Precision caution (do not over-claim the magnitude)
r8 and r32 are ONE training run each, and the noise floor is about 3 points, so the true
r8 value could plausibly sit anywhere from about 33 to 39. The DIRECTION is above noise and
therefore real; the exact magnitude rests on a single draw per point. Write "approximately
5 to 6 points, exceeding the measured retraining noise floor", never "exactly 5.63 points".
We do not have the replicates to justify that precision.

## The compute envelope (a measured finding, not a gap)
r64 and r128 are absent for a reason we measured, and it is on-thesis for a paper about
resource-constrained hardware:

    r8    31 s/optimizer step    fits in 6GB      6h 38m total
    r32   39 s/optimizer step    fits in 6GB      7h 25m total, peak 5.65 GiB
    r64   174 s/optimizer step   SPILLS to system RAM, 4.4x slower, about 37h projected
    r128  not attempted to completion, would spill harder

IMPORTANT MECHANISM CORRECTION. We predicted r128 would raise a clean CUDA OOM. It does
NOT, and neither does r64. On Windows WDDM the driver spills to system RAM over PCIe
instead of raising OOM, so the failure mode is a 4.4x slowdown, not a crash. The OOM guard
in run_rank_sweep.py is still correct and still tested, it simply is not the path this
hardware takes. Describe the limit as "exceeds the VRAM budget and spills to host memory",
not as "out of memory".

r64 was stopped deliberately after 16 of 760 steps. Rationale: 37 hours for one refinement
point on a curve that is not load-bearing (r32 is the production config, and the thesis is
about steering, not rank shape) was a bad trade against ETP3 time.

## NEXT: writing, no more experiments
1. Results section covering all three reviewer comments, using
   results/comparisons/final_paper_table.csv and results/figures/.
2. Apply the four write-up points below verbatim.
3. Submission. Then ETP3.

## THE FOUR WRITE-UP POINTS THAT MUST LAND EXACTLY AS FRAMED

### 1. The rank result is NOT flat (see the correction above, this is the big one)

### 2. The measured retraining noise floor, about 3 points (report prominently)
rank_32 (41.25%) and decoding_topp / lexicon_15 (38.12%) are INDEPENDENT RETRAININGS of
one configuration, differing only in random init: production never seeded, run_rank_sweep
does. Their 3.13-point difference is a measured retraining variance, which is 5 stories out
of 160 and 0.81 standard errors (one SE at p=0.4, N=160 is 3.87 points).

It does double duty and reframes the WHOLE results table:
  a. it makes the rank sweep interpretable, and is what lets us say the 5.63-point rank
     gap is real rather than noise; and
  b. it retroactively justifies every "inside noise" call elsewhere. lexicon_10 at 38.75
     against lexicon_15 at 38.12 is a 0.63-point gap, now MEASURED as negligible rather
     than merely asserted.
One line in methods or results establishing it, then reference it wherever a gap is called
negligible. Most papers in this space do not bother, and it costs nothing here.

### 3. The two summary rows for one config are deliberate, keep both
The summary holds rank_32 at 41.25 and decoding_topp at 38.12, which look like a duplicate.
They are not. Keep both, label them as independent retrainings, cite them AS the noise-floor
measurement:
  "rank_32 and the tier-15 / top-p run are independent retrainings of the production
   configuration; their 3.13-point difference establishes the retraining noise floor."

### 4. The capacity claim stays NARROW (locked before any numbers existed)
Rank was swept at fixed alpha=64 with lm_head and wte trained in full throughout. Because
those two dominate the parameter count, total trainable capacity varied only 1.66x
(134,556,160 at r8, verified against the real model, to 152,250,880 at r32).
  SUPPORTED:     "LoRA rank specifically has a modest effect under steering."
  NOT SUPPORTED: "adapter capacity is not the operative factor."
Capacity is never made small, so the broader claim would be an overreach a reviewer could
catch with the same arithmetic. The capacity-in-general argument is carried by the
model-size result, the lexicon cliff and decoding. Keep the claims in separate lanes.

## Paper-writing consequences to record later (not code)
- Ablation prompt regime differs from the original 47.50% headline run (fixed identical
  stems, all starting with "The"). The ablation table needs a footnote; the reconciliation
  is done by the footnote, not by re-running the hybrid on the varied set.
- Provenance sentence the neutrality filter earns: "we verified the stems themselves carry
  no target-emotion signal."
- Old rank CSVs (evaluation_results_*) are NOT a rank ablation and must never be presented
  as one. They vary model size and technique alongside rank, have inconsistent N, and sit
  on the original manifest. Fig1 draws them only in the clearly separated old-manifest
  group.
- DO NOT run src/5_evaluation/09_generate_paper_tables.py. It hardcodes every row including
  6.67% for the baseline and would recreate the stale table. It is under the frozen
  do-not-edit pipeline so it was left untouched. Use src/6_ablations/make_paper_table.py,
  which reads all_configs_summary.csv and states the baseline once at 10.00%.

## Completed

### Step 0: original 47.50% run used fixed prompts  (DONE)
Prompts were genuinely identical: hardcoded constant, discarded loop variable, all 160
stories started with the single word "The". Nothing to recover, so the manifest was
regenerated.

### Step 1 and 1b: manifest regenerated and neutrality-filtered  (DONE)
885 clean held-out stems scored in isolation by the RoBERTa judge; kept only stems where
classify(stem) == Neutral AND max P(any Plutchik emotion) < 0.20. 686 of 885 passed both,
condition 2 binding. Selection then drawn from the filtered pool, preserving the label
distribution (Anger 40, others 20, N=160), 20 stems times 8 emotions, byte-identical across
two runs. Per-stem scores committed in data/processed/neutral_stem_pool.csv.
Honest caveat: a few surviving stems still read as eventful to a human ("Five shots. Five
shots rang out"). The judge scores every emotion below 0.20 for them, and the judge is what
grades the outputs, but do not overclaim these as human-neutral.

### Step 2: src/6_ablations/common.py and the runners  (DONE)
Shared spine: SteeringProcessor loaded live from 07_evaluate_hybrid.py with a SHA256 drift
guard, load_judge/classify, load_hybrid_model, load_prompt_set, ResumableWriter (per-row
flush, resume keyed on frozen prompt_id), write_provenance, summarize/append_to_summary,
execute_run. Later extended additively with adapter_path and extra_provenance for the rank
sweep, so all earlier runs stay valid unchanged.

### Task A: seven ablation runs  (DONE, commits 04dee57 and f3dc1eb)
  decoding_greedy 44.38 | lexicon_10 38.75 | lexicon_15 38.12 | decoding_topp 38.12
  decoding_topk 36.25 | lexicon_5 22.50 | beta_15 18.12 | beta_1 11.88
Fig4 reads b=1, b=5 and b=15 all from revision-manifest runs, so the beta curve needs no
asterisk. The 6.67% b=15 figure in CLAUDE.md refers to the ORIGINAL-manifest aggressive run
(N=120), a different run.

### Task B: Phi-3 and Qwen baselines  (DONE, commit f612c4d)
Inference-only, 4-bit, correct chat template each, equal 60-token budget.
baseline_phi3 25.62, baseline_qwen 21.88. Both lose to steered GPT-2.

### Task C: PPLM baseline  (DONE, commit 97b86e0)
PPLM on gpt2-large, linear attribute head over frozen hidden states. 35.62%, peak 3.67 GiB,
56.9 s/story. Runs in the 6GB budget, lands just below our control at 6.2x the per-story
cost. GeDi skipped as planned.

### Rank sweep  (DONE as a two-point sweep, commits 2d987f5 and 548290c)
rank_32 41.25% (7h 25m) and rank_8 35.62% (6h 38m), both on the revision manifest with
steering on, everything except r frozen from the production recipe and checked at runtime
by verify_training_source(). r64 stopped at step 16 of 760 after measuring 174 s/step;
r128 not completed. See the NOT FLAT correction and the compute envelope above.

### Housekeeping  (DONE)
- Deleted the stale results/comparisons/final_paper_table.csv (Feb 10 artifact carrying
  6.67%), and wrote a replacement generator that reads the summary.
- Confirmed git author is the user on every commit; the Claude attribution appears only as
  a message trailer. No history rewrite was performed, deliberately, because config.yaml
  provenance files reference commit hashes and rewriting would invalidate them.
