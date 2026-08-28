# HANDOFF addition: Task B (baselines) and clean LoRA-rank re-run

Paste/merge this into HANDOFF.md. Order decided: Task B first, then rank re-run, then
decide on PPLM (Task C).

## Task B: newer lightweight model baselines (reviewer comment 2)  [DO FIRST]
Inference only, no fine-tuning. bitsandbytes 0.49.2 already installed.
Models: microsoft/Phi-3-mini-4k-instruct, Qwen/Qwen2.5-1.5B-Instruct (4-bit).

NON-NEGOTIABLE controls (or the comparison is unusable):
- SAME revision manifest (test_prompts_160.csv), the same 160 varied stems.
- SAME judge (RoBERTa GoEmotions -> Plutchik-8), same target labels, same classify().
- SAME max_new_tokens=60 as the GPT-2 runs. This is a deliberate equal-generation-budget
  fairness constraint; document it as such, not as an oversight. Longer generations would
  give the judge more signal and unfairly favor the newer models.
- Native chat template per model (Phi-3 and Qwen are instruction-tuned; feeding them the
  raw "Emotion: X | Story:" format would handicap them and invite "you prompted them
  wrong"). Document that prompt format varies per model BY DESIGN while task and manifest
  are held constant; forcing one format on all would itself be the confound.
Output: baseline_phi3_results.csv, baseline_qwen_results.csv  ([Target, Detected, Story]).

Framing to remember: Fig5 is "steered GPT-2 vs UNASSISTED newer models", not "GPT-2 vs
newer models". That is the correct comparison for the paper's claim (our method on
constrained hardware vs newer models without it), but state it precisely. If a baseline
BEATS 38.12% unassisted, report it honestly, it reframes the contribution as "competitive
on minimal hardware" rather than "superior". Do not decide which sentence you are writing
until the number is in.

## Clean LoRA-rank re-run (reviewer comment 4, replaces the unusable old 4 CSVs)  [DO SECOND]
The old rank CSVs (N 70/105/160/160, differing formats, 2 missing Story, original manifest)
are NOT a clean sweep and are dropped. Re-run properly.

Vary ONLY: rank in {8, 32, 64, 128}.
DECIDE FIRST (blocking): alpha strategy. Check what alpha the production hybrid (rank-32,
the 38.12% / 47.50% config) actually used. Submitted Table 1 says alpha=64.
  - If production used alpha=64: hold alpha=64 FIXED across the sweep. This makes the
    rank-32 point coincide with production (a built-in anchor). Effective scaling
    alpha/rank co-varies; state "rank effect at fixed alpha=64, matching production;
    scaling therefore co-varies" plainly.
  - Only if you want pure-capacity isolation instead: scale alpha with rank to hold
    alpha/rank constant, but then no point coincides with the real trained model.
  Lean: fixed alpha=64 to keep the production anchor. CONFIRM production alpha before running.
FREEZE (identical across all four): training data, seed, epochs, LR, batch size,
gradient accumulation, everything except rank.
EVALUATE (identical to Task A pipeline): revision manifest, same judge, max_new_tokens=60,
top-p decoding (match production, NOT greedy, so rank and decoding do not entangle),
STEERING ON (production config).
ANCHOR CHECK: rank-32-steered should land near 38.12%. If it does not, something differs
from production, stop and reconcile before trusting the other three points.
Output: rank_8_results.csv, rank_32_results.csv, rank_64_results.csv, rank_128_results.csv.

Hypothesis to state BEFORE seeing numbers (keeps it honest): with steering ON, rank is
expected to matter little (flat-ish sweep), because the paper's thesis is that steering,
not capacity, is the dominant lever. A flat sweep CONFIRMS that thesis from a second axis
(the first was model size, Medium vs Large). If rank matters a lot even with steering,
that is a surprise that complicates the thesis and must be engaged honestly, not smoothed.

## Then: Fig5, all_configs_summary update, commit, decide on PPLM (Task C)
Fig5 becomes real once Task B lands (steered GPT-2 vs Phi-3 vs Qwen, one frame, same
manifest). Rank sweep gets its own figure or a clean sub-table, single manifest, one
variable. Then reassess whether PPLM (timeboxed, most likely to fail in 6GB, resume fix
now in place) is worth attempting or whether comments 2+4 answered well is enough.

## Deliverables to send after B + rank (for Results drafting, no round-trips):
1. Final all_configs_summary.csv (all runs incl. baselines and rank sweep, all N, all F1).
2. Full seven-emotion greedy vs top-p breakdown on revision manifest (all 7, not 4).
3. Fig4 three points on single-manifest footing + actual b=15 value + observed failure mode.
4. Baselines: Phi-3 and Qwen top-1 + per-emotion F1, plus whether either beats 38.12%.
5. Rank sweep: four top-1 values + whether rank-32 anchored near 38.12%.
6. If cheaply available: mean generation length per config (judge-independent collapse
   evidence, b=15 should pin near the 60-token cap if EOS-suppression story holds).
