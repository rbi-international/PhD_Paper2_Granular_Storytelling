# HANDOFF.md  (living work order, updated between sessions)

Read this after CLAUDE.md. This is the current state and next actions. When a section is
done, move it to "Completed" at the bottom. When a decision changes, edit it here.

Last updated: 2026-08-26

## Where we are
- Repo on GitHub (private), local and origin in sync.
- Task A scaffolding done: frozen 160-prompt manifest, lexicon tiers with drift guard,
  experiment folder structure.
- Manifest regenerated from held-out WritingPrompts and neutrality-filtered (see Completed).
- Environment verified (study_torch, bitsandbytes present, no installs needed).
- CLAUDE.md corrected: export syntax for Git Bash, cuDNN 9.10.2.
- NOT yet built: src/6_ablations/common.py and the three runners.

## NEXT ACTION: commit the data layer, then build common.py

### Commit first (pure data + docs, clean checkpoint before result-producing code):
Commit together: the neutrality-filtered manifest, its provenance/sidecar file, this
HANDOFF update, and the corrected CLAUDE.md. Reason: every run stamps the commit hash of
the code that made it, so freeze the INPUT manifest in its own commit before the code that
consumes it. If a run looks wrong later, you can tell manifest-changed from code-changed by
looking at two commits instead of one.

### Then build src/6_ablations/common.py (GATED: touches models, show plan + structure first)
Shared spine imported by all runners:
- SteeringProcessor: copy from 07_evaluate_hybrid.py (the 47.50% version), guard vs drift.
- load_judge() / classify(): the RoBERTa GoEmotions to Plutchik-8 judge, unchanged.
- load_hybrid_model(): GPT-2 Large + models/gpt2_large_optimized.
- load_prompt_set(): reads the frozen manifest.
- ResumableWriter: flush per row, resume from last prompt_id after a crash.
- write_provenance(): config.yaml with seed, commit hash, versions, hardware.

## Then: the seven Task A runs (about 1.5 to 2 hours on the 3060)
All read the frozen manifest. All output Target,Detected,Story CSVs at repo root.
1. Lexicon tier 15 (full)   -> lexicon_15_results.csv   (tier holds 13-15 words, note in config)
2. Lexicon tier 10          -> lexicon_10_results.csv
3. Lexicon tier 5           -> lexicon_5_results.csv
4. Decoding greedy          -> decoding_greedy_results.csv   (valid at N=160: stems are distinct now)
5. Decoding top-k (k=50)    -> decoding_topk_results.csv
6. Decoding top-p (p=0.92)  -> decoding_topp_results.csv
7. Beta b=1.0               -> steered_b1_results.csv        (replaces hardcoded 13.12 proxy in fig4)
LoRA rank ablation (8/32/64/128) already exists in prior CSVs; do NOT re-run, just have the
summary script read them.

## Then: consolidate + figures
Append each run to results/all_configs_summary.csv (config, model, technique, N, top1_acc,
then per-emotion F1 for Joy, Fear, Sadness, Anger, Trust, Surprise, Anticipation).
Regenerate results/figures/ at 300 DPI via make_figures.py (already fig4-patched to read
steered_b1_results.csv automatically).

## Paper-writing consequences to record later (not code)
- Ablation prompt regime differs from the original 47.50% headline run (which used fixed
  identical stems, all starting with "The"). The ablation table needs a footnote. I chose
  NOT to re-run the hybrid on the varied set, so the footnote does the reconciliation. (If
  I later want ablations directly stackable with the 47.50% anchor, one hybrid run on the
  varied set would be needed. Not now.)
- Provenance sentence the paper can now make (kept literally true by the neutrality filter):
  "we verified the stems themselves carry no target-emotion signal."

## Later (Tasks B and C, not started, separate gating)
- Task B: Phi-3-mini-4k-instruct and Qwen2.5-1.5B-Instruct, inference-only, 4-bit, same
  160 prompts, correct chat template each. -> baseline_phi3_results.csv, baseline_qwen_results.csv
- Task C: PPLM on GPT-2 using the RoBERTa judge as attribute model, same 160 prompts.
  Timebox it; if it will not run stably in 6GB, stop and fall back to A+B. Skip GeDi.

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
