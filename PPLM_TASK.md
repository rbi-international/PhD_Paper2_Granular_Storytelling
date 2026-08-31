# Task C: PPLM controllable-generation baseline (reviewer comment 3)

Read CLAUDE.md and HANDOFF.md first. This is the last experiment before the paper is
written. Comment 3 asked for a stronger controllable-generation (CTG) baseline beyond
GPT-2 variants. PPLM is that baseline.

## Goal and success definition (read this before writing code)
The goal is an HONEST OUTCOME, not a working PPLM at all costs. Two outcomes both succeed:
  A) PPLM runs on the 160-prompt revision manifest and produces judgeable stories.
     -> we get real CTG baseline numbers for the comparison table.
  B) PPLM cannot run in 6GB (OOM) or is impractically slow.
     -> we record the SPECIFIC failure (exact error, memory ceiling, per-token latency)
        and report it as an on-thesis finding: gradient-based CTG is infeasible on the
        constrained hardware this paper targets, which motivates our forward-pass-only
        steering. This is a legitimate, strong answer to comment 3, not a failure.

Either way we win. Do NOT debug PPLM into working past the timebox.

## HARD TIMEBOX
If PPLM is neither cleanly running (producing coherent output over the manifest) nor
cleanly failing (a clear, documented OOM/latency wall) within about 2 hours of work,
STOP. Write up whatever behavior was observed and fall back to outcome B. Report to me
before spending more time. The paper does not need PPLM to succeed; it needs an honest,
specific answer to comment 3.

## What PPLM is (so the implementation is faithful, not a strawman)
PPLM (Dathathri et al. 2020, Plug and Play Language Models) steers a frozen LM using a
separate attribute model. At each decoding step it runs forward, computes a gradient of
the attribute model's loss w.r.t. the LM's past key/value activations, perturbs those
activations toward the desired attribute, and re-decodes. The per-token backward pass is
the expensive part and the likely 6GB failure point.

## Controls (same discipline as Task A and Task B, non-negotiable)
- Backbone: GPT-2 (same family as the paper's models). Use GPT-2 Large if it fits with the
  per-token backward pass; if Large OOMs immediately, fall back to GPT-2 Medium and RECORD
  that Large did not fit (that itself is outcome-B evidence). State clearly which backbone
  produced the reported numbers.
- Attribute model: reuse the existing RoBERTa GoEmotions -> Plutchik-8 judge as PPLM's
  attribute classifier where architecturally possible. If RoBERTa cannot serve as the
  gradient-providing attribute model directly (PPLM classically uses a small classifier
  head on the LM's hidden states, not a separate encoder), implement the standard PPLM
  attribute head trained on the LM hidden states, and DOCUMENT the deviation. Do not
  silently substitute something weaker; note exactly what the attribute model is.
- Manifest: the SAME test_prompts_160.csv, same 160 varied stems.
- Judge: the SAME classify() for evaluation (RoBERTa), so PPLM is scored identically to
  every other config. Keep attribute-model (steering) and judge (evaluation) roles
  conceptually separate even if both are RoBERTa-based; note any shared-model circularity
  as a limitation.
- Budget: SAME max_new_tokens=60.
- Output: baseline_pplm_results.csv in [Target, Detected, Story] schema.
- Provenance: same experiments/experiment_XXX/ folder with config.yaml (seed, commit hash,
  git_tree_clean, versions, hardware), metrics.json.

## Resume (this run is the most likely to crash, so this matters)
Wire the ResumableWriter reuse path into the PPLM runner explicitly, the same fix that
recovered the b=15 run (reuse an existing incomplete experiment directory for the same
config, skip completed prompt_ids, resume rather than starting a new folder). Verify it
works on PPLM specifically before launching the full 160: run a couple of rows, kill it,
relaunch, confirm it resumes.

## Instrumentation to capture regardless of outcome (needed for the write-up)
- Peak VRAM during a PPLM step (torch.cuda.max_memory_allocated).
- Wall-clock per generated story (mean seconds/story), so we can report latency vs the
  forward-only configs (which ran at ~normal speed).
- If OOM: the exact error, at what backbone/rank, and the memory ceiling hit.
These are the numbers that make outcome B a measured finding rather than an assertion.

## Steps
1. Confirm tree clean, note starting commit hash.
2. Write src/6_ablations/run_pplm_baseline.py reusing common.py (manifest, judge,
   ResumableWriter, provenance). Add only the PPLM steering loop and attribute model.
3. Smoke test: 2 rows end-to-end, confirm schema, confirm resume, capture peak VRAM.
   Report the smoke result and the VRAM/latency reading BEFORE running all 160.
4. If smoke is viable: run all 160 (resume-safe). If smoke OOMs/crawls: stop, that is
   outcome B, write it up.
5. Commit the CSV + provenance (and the runner) before reporting final numbers.

## Report back to me
- Which outcome (A or B), with the specific numbers: either PPLM top-1 + per-emotion F1,
  or the exact failure (backbone, peak VRAM, error, latency).
- Whether PPLM (if it ran) beats, matches, or loses to the steered GPT-2 at 38.12%.
- Peak VRAM and mean latency per story either way.
Then I draft the Results section covering comments 2, 3, and 4 together.

Start with steps 1 to 3 and report the smoke test before committing to the full run.
