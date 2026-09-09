# HANDOFF.md  (living work order, updated between sessions)

Read this after CLAUDE.md. This is the current state and next actions. When a section is
done, move it to "Completed" at the bottom. When a decision changes, edit it here.

Last updated: 2026-09-09

## Where we are
ALL EXPERIMENTS ARE FINISHED. Nothing is running. Nothing is left to measure.
THE ONLY REMAINING WORK IS WRITING THE PAPER.

State at save time: tree clean, origin in sync, HEAD at 1bae45c (Fig5 fix).
GPU idle. Ollama is installed and may be running for the unrelated AcademiaHumanify
project; it holds no VRAM when idle, but it will contend for the 6GB card if used.

- Tasks A, B and C all done and committed. The LoRA-rank sweep is done as a TWO-POINT
  sweep (r8 and r32) plus a measured compute envelope for r64 and r128.
- results/all_configs_summary.csv holds 14 runs with per-emotion F1 throughout.
- Latency is now measured and logged by execute_run for every future run.
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

## LATENCY: MEASURED, DONE (2026-09-08). The ratio is about 4.6x, NOT 6.2x.

    hybrid   12.37 s/story   peak 4.03 GiB   (160 rows, generation only, judge excluded)
    PPLM     56.4  s/story   peak 3.66 GiB   (20 rows, re-timed the same session)
    PPLM     56.9  s/story                   (original 160-row run, 0.9% drift)
    RATIO    4.56x using both of today's numbers, 4.60x using the original PPLM

Report it as "approximately 4.6x". The retracted 6.2x implied a hybrid of 9.18 s/story;
the true figure is 12.37, so the old claim overstated our advantage by about 35%. Measuring
it was the right call.

BOTH numbers come from the same machine state, measured back to back, so the ratio has no
cross-session confound. The 0.9% drift between today's PPLM and the original confirms the
machine is in a comparable state to the Task C session.

### An honest nuance that must go in the paper
PPLM uses LESS peak VRAM than we do (3.66 GiB against our 4.03 GiB). The correct claim is
therefore about SPEED, not resources in general: "PPLM is approximately 4.6x slower per
story than our method at comparable peak memory (3.66 GiB against 4.03 GiB)." Do not write
that our method is cheaper across the board, because on memory it is marginally not.

### CORRECTION: hybrid_timed is NOT a third noise-floor draw
It was staged as one. It is not, and the run proved it: hybrid_timed is BYTE-IDENTICAL to
decoding_topp, every story and every detection, accuracy 38.12% with identical per-emotion
F1. generate_one seeds per row with torch.manual_seed and this run used the SAME production
adapter, so identical output was guaranteed. rank_32 differs only because it is a different
adapter from a separate training run.

  THE NOISE FLOOR STILL RESTS ON TWO RETRAININGS: decoding_topp 38.12 and rank_32 41.25,
  giving 3.13 points. It did NOT become three.

The silver lining is real and worth one sentence in the paper: a full 160-story
regeneration, in a different process on a different day, reproduced a prior run bit for
bit. That is an exact-reproducibility demonstration stronger than most papers offer.

## Superseded: the latency measurement plan (kept for the reasoning, now executed)

DONE. Executed 2026-09-08, results above. Kept because the reasoning explains why the
claim was retracted and how the measurement was designed.

### Why: the 6.2x claim currently has NO measured denominator
"PPLM costs 6.2x more per story than our method" has been repeated in the Task C commit
message, in this file, and in conversation. Tracing it: PPLM's numerator IS recorded
(experiments/experiment_013_baseline_pplm/metrics.json holds mean_seconds_per_story 56.9
and peak_vram_gib 3.67). The hybrid DENOMINATOR is recorded NOWHERE. 56.9 / 6.2 implies
about 9.2 s/story for the hybrid, and that number does not exist in the repo. Note also
that an earlier "about 2 s/story" figure floated in conversation was never measured either.
experiment_013 is the ONLY one of thirteen runs that recorded timing at all, because
execute_run never measured it.

This is the cleanest answer to reviewer comment 3, and it is exactly the kind of crisp
checkable claim a reviewer divides out. If the denominator does not exist, the latency
argument collapses on contact. About 25 minutes of GPU converts it from a liability into
a strength.

### DO NOT anchor on 6.2
Report whatever the measurement says. If hybrid is about 9 s/story the ratio is about 6x;
if hybrid is much faster the ratio is much larger. The point of measuring is that we do
not know the answer yet. Delete "6.2x" from this file and from any draft once the real
number exists.

### The plan (model-touching, so it is GATED, confirm before writing)
1. Add to common.execute_run's metrics, for every future run:
     mean_seconds_per_story   wall clock around generate_one only, judge excluded
     peak_vram_gib            torch.cuda.max_memory_allocated
   Logged by default from now on so this gap cannot recur, and ETP3 inherits it free.
2. New runner: timed hybrid on the EXACT production decoding_topp config. Same GPT-2 Large,
   same adapter, steering on at boost 5.0, full tier-15 lexicon, same 160-stem revision
   manifest, max_new_tokens=60, top-p. The ONLY difference from the existing decoding_topp
   run is that this one records latency. If any generation parameter differs the ratio is
   not clean. Output config name: hybrid_timed.
3. Re-time PPLM on the SAME machine state, back to back. PPLM's 56.9 was measured in the
   Task C session weeks ago; a ratio built from two numbers measured under unknown
   conditions is weak. Even 20 stories is enough to confirm 56.9 still holds today. Use
   run_pplm_baseline.py --smoke 20.
4. Compute and report the REAL ratio.

### Free bonus: a THIRD independent retraining for the noise floor
hybrid_timed is the identical config to decoding_topp (38.12) and rank_32 (41.25), so its
accuracy is a third draw of the same configuration. Sanity check: it must land in the 38 to
41 band. If it does, the noise floor gets a third point and is stronger. If it lands wildly
outside, something is wrong with the timing run and its latency should NOT be trusted.

## VERIFIED ENVIRONMENT (read from the system and all 14 configs on 2026-09-09)

Use this block verbatim for the reproducibility statement. Every value was read from the
machine or from committed configs, none from memory. All 14 experiment configs agree
unanimously on every recorded field (each returned 14/14), so the environment did not
drift across the runs and the currently installed env is what produced the results.

    HARDWARE
      GPU                NVIDIA GeForce RTX 3060 Laptop GPU  (Laptop variant, confirmed)
      Compute capability 8.6 (Ampere GA106)
      VRAM               6,441,926,656 bytes = 6.44 GB = 6.00 GiB = 6144 MiB
      NVIDIA driver      591.74      [recorded post hoc, NOT captured in run provenance]

    OPERATING SYSTEM
      Microsoft Windows 11 Home Single Language, Build 26200
      Display driver model: WDDM

    CUDA / cuDNN
      CUDA (PyTorch build)  12.8
      cuDNN                 9.10.2

    CORE SOFTWARE  (identical in all 14 configs)
      Python 3.11.14 (conda env "study_torch"), PyTorch 2.10.0+cu128,
      transformers 5.3.0, peft 0.18.1

    SUPPORTING  [installed-now; NOT captured in run provenance]
      datasets 4.6.1, bitsandbytes 0.49.2, numpy 2.4.2, pandas 3.0.1,
      scikit-learn 1.8.0  (computes every accuracy and F1 in the results)

    TRAINING HYPERPARAMETERS
      LoRA r=32, alpha=64, dropout 0.1, target c_attn/c_proj/c_fc,
      modules_to_save lm_head+wte, LR 2e-4, 10 epochs, seq len 256,
      batch 1 x grad-accum 32, fp16=True, gradient checkpointing, seed 42
      NOT SET, so library defaults applied: warmup_steps 0, warmup_ratio None,
      weight_decay 0.0, lr_scheduler linear, max_grad_norm 1.0,
      optim adamw_torch_fused, adam betas 0.9/0.999, eps 1e-8
      Report these as DEFAULTS, not as tuned choices.

    INFERENCE
      boost 5.0, top-p (temp 0.8, p 0.92), max_new_tokens 60,
      repetition_penalty 1.2, full tier-15 lexicon, N=160,
      judge SamLowe/roberta-base-go_emotions -> Plutchik-8

### FOUR TRAPS in the environment data, do not copy these verbatim
1. platform reports "Windows-10-10.0.26200-SP0" but the machine is WINDOWS 11. Windows 11
   still reports a 10.0.x kernel version. systeminfo confirms Windows 11 Build 26200.
   Write Windows 11. This also matters for the WDDM spillover in the rank sweep.
2. nvidia-smi shows "CUDA Version: 13.1". That is the DRIVER'S MAXIMUM SUPPORTED runtime,
   not what ran. The runs used CUDA 12.8 (PyTorch build). Never report 13.1.
3. cuDNN is stored in configs as the integer 91002, which decodes to 9.10.2
   (major = v//10000, minor = (v%10000)//100, patch = v%100). Report 9.10.2.
4. 6.44 GB and 6.00 GiB are the SAME quantity, decimal versus binary. Never write
   "6.44 GiB", which would be a unit error.

### Provenance gaps to be honest about
collect_environment() records only python, platform, torch, transformers, peft,
cuda_available, cuda, cudnn, gpu, vram. It does NOT record the NVIDIA driver, nor
scikit-learn, numpy, pandas, datasets or bitsandbytes. scikit-learn computes every metric
in the paper, so its version is an inference from today's install, not a measurement from
run time. A two-line extension to collect_environment() would close this before ETP3.

## STALE ARTIFACTS: do not hand these off, do not run these
Checked 2026-09-09. The repo contains several pre-revision leftovers that would put wrong
numbers into the manuscript.

    results/figures/            <- CURRENT, 300 DPI, the ONLY figures to use
    ./Fig*.png                  STALE (Aug 13, 4 files, pre-revision)
    ./files (2)/                STALE duplicate tree, includes an old make_figures.py
                                that hardcodes accs = [13.12, 47.50, 6.67]
    results/plots/Figure*.png   STALE (Feb 10, original submission)

Two scripts under the frozen do-not-edit pipeline will regenerate wrong numbers if run:
  - src/5_evaluation/09_generate_paper_tables.py   hardcodes 6.67% baseline and 47.50%
  - src/5_evaluation/10_visualize_results.py       hardcodes [6.67, 20.00, 13.12, 47.50]
                                                   and wrote the stale results/plots/
DO NOT RUN EITHER. Use src/6_ablations/make_paper_table.py (reads the summary, baseline
stated once at 10.00%) and make_figures.py.

## FILES FOR WRITING (all relative to the repo root)
    results/all_configs_summary.csv          14 runs, per-emotion F1, caveat column
    results/comparisons/final_paper_table.csv 15 rows, regenerated from the summary
    results/figures/Fig1_Full_Comparison.png  all configs, two manifest groups (hatched)
    results/figures/Fig2_Confusion_Matrix.png hybrid confusion matrix
    results/figures/Fig3_PerEmotion_F1.png    baseline vs steered vs hybrid (ORIGINAL manifest)
    results/figures/Fig3b_PerEmotion_F1_Revision.png  revision-manifest runs only
    results/figures/Fig4_Beta_Ablation.png    b=1/5/15, all revision manifest
    results/figures/Fig5_New_Baselines.png    ours vs Phi-3/Qwen/PPLM, all revision manifest
    results/figures/Fig6_Rank_Sweep.png       rank sweep with noise band shaded
    experiments/experiment_0XX_*/metrics.json  per-run metrics (14)
    experiments/experiment_0XX_*/config.yaml   per-run provenance (14)
    src/5_evaluation/07_evaluate_hybrid.py     Equation 2 (SteeringProcessor, lines 32-52)
                                               Algorithm 1 (generation loop, lines 98-132)
Experiment numbering skips 005 and 007: aborted runs, never renumbered.

Equation 2, stated from the source: an additive pre-softmax logit bias, uniform over all
lexicon token ids, no decay and no context dependence:
    s'[t] = s[t] + beta * 1[t in T_e],  beta = 5.0
T_e expands three surface forms per word (bare, leading space, capitalised) and
de-duplicates, so the bias reaches BPE variants. It is a TOKEN-level bias, not sequence
level, and the paper should say so.

## NEXT: writing, no more experiments
1. Full paper as a FORMATTED DOCUMENT (delivery choice; say so if plain prose is wanted
   instead). Title, abstract, all sections, tables, figure callouts, framed for the
   applied/systems venue, integration-not-invention, varied-stem headline with fixed-stem
   context, the honest NOT-FLAT rank finding, the noise floor as a rigor feature, and the
   measured latency ratio anchoring comment 3.
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
56.9 s/story (recorded in metrics.json, NOT in config.yaml). Runs in the 6GB budget and
lands just below our control. GeDi skipped as planned.
CORRECTION: earlier versions of this line said "6.2x the per-story cost". That ratio was
never measured, because the hybrid denominator was never recorded. Claim removed pending
the measurement described under NEXT ACTION. Do not reinstate it from memory.

### Rank sweep  (DONE as a two-point sweep, commits 2d987f5 and 548290c)
rank_32 41.25% (7h 25m) and rank_8 35.62% (6h 38m), both on the revision manifest with
steering on, everything except r frozen from the production recipe and checked at runtime
by verify_training_source(). r64 stopped at step 16 of 760 after measuring 174 s/step;
r128 not completed. See the NOT FLAT correction and the compute envelope above.

### Latency measured  (DONE 2026-09-08, commit 4460429)
hybrid 12.37 s/story, PPLM 56.4 re-timed the same session. Ratio about 4.6x. See the
LATENCY section above, including the correction that hybrid_timed is NOT a third
noise-floor draw and the nuance that PPLM uses LESS peak VRAM than we do.

### Fig5 corrected  (DONE 2026-09-09, commit 1bae45c)
Fig5 showed "Ours: Hybrid" at 47.50%, the ORIGINAL fixed-stem number, contradicting the
manuscript which reports the revision manifest throughout. Root cause was NOT a hardcoded
value: fig5() read hybrid_results.csv (the original-manifest file) and computed 47.50 from
it honestly, which is why grepping for a literal 47.5 in make_figures.py found nothing.

Fixed by routing all four bars through a new summary_accuracy() helper that reads
results/all_configs_summary.csv by config_name, so figure and manuscript share one source
and cannot diverge again. "Ours: Hybrid" is now rank_32 at 41.25%. The title now names the
manifest and N, which is newly accurate: previously the "Ours" bar sat on a different
manifest from the other three bars, silently mixing evaluations.

Fig1 was audited bar by bar and LEFT UNCHANGED BY DESIGN. It shows 47.50 exactly once, in
the hatched separately labelled original-manifest group next to 10.00 baseline, 13.12
Large+instruct and 6.67 b=15. That is a labelled historical comparison, not a headline
claim, and it is correct in context. No other figure contained a stale 47.50.

### Environment verified  (DONE 2026-09-09)
Full reproducibility block recorded above, read from the system and from all 14 configs,
with the four unit and version traps documented.

### Housekeeping  (DONE)
- Deleted the stale results/comparisons/final_paper_table.csv (Feb 10 artifact carrying
  6.67%), and wrote a replacement generator that reads the summary.
- Confirmed git author is the user on every commit; the Claude attribution appears only as
  a message trailer. No history rewrite was performed, deliberately, because config.yaml
  provenance files reference commit hashes and rewriting would invalidate them.
