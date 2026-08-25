# Claude Code Prompt: GranularStory paper revision (new baselines, ablations, 300 DPI figures)

Copy everything below the line into Claude Code, opened in the project root
(E:/PHDETP2/PhD_Paper2_Granular_Storytelling).

===============================================================================

# Project: GranularStory paper revision, new baselines and expanded ablations

## Context
I am revising a REJECTED journal paper (SN Computer Science, manuscript SNCS-D-26-04624).
New title: "GranularStory: Enhancing Emotional Fidelity in Narrative Generation using
Hybrid NSIT on Resource-Constrained Hardware."

The paper studies emotion-controlled story generation on a single RTX 3060 (6GB VRAM),
32GB RAM, Windows. The core method combines LoRA instruction-tuning of GPT-2 with
inference-time lexical steering (a logit boost on emotion-specific words). GPT-2 MUST
remain the central fine-tuned model (my PhD Objective 2 requires it). Do NOT replace it.
Newer models are ADDED as comparison baselines only, never as replacements.

Reviewers rejected it and asked for: (2) comparison against newer lightweight models,
(3) stronger controllable-generation baselines beyond GPT-2 variants, (4) an expanded
ablation study. I am NOT doing human evaluation. This prompt covers ONLY the code and
experiment work. The paper writing happens separately.

## Environment (already exists on my machine)
Activate with BOTH commands every session (the second prevents an OpenMP crash on Windows):
    conda activate study_torch
    set KMP_DUPLICATE_LIB_OK=TRUE

Confirmed stack: Python (conda), PyTorch 2.10.0+cu128, CUDA 12.8, cuDNN 9.15,
transformers, peft, datasets, on an RTX 3060 Laptop (6.44GB VRAM), Windows.

Before Task B, verify bitsandbytes is installed (needed for 4-bit Phi-3/Qwen):
    python -c "import bitsandbytes; print(bitsandbytes.__version__)"
If it errors, install it WITHOUT downgrading torch or transformers (this environment
produced my original 47.50% result and must stay reproducible). If any install risks
the existing stack, clone the env first:
    conda create --name granular-revision --clone study_torch

## My existing repo structure (do not break relative paths)
    project_root/
      data/processed/            # final_train_ready.csv, instruction_train_ready.csv
      models/                    # trained LoRA adapters (gpt2_large_optimized, etc.)
      src/
        1_data_ingestion/
        2_annotation/            # teacher annotate (RoBERTa GoEmotions -> Plutchik-8)
        3_training/              # 01_train_lora.py ... 06_train_gpt2_large_optimized.py
        4_generation/            # 01_generate_stories.py, 02_steered_inference.py,
                                 #   03_aggressive_steering.py
        5_evaluation/            # 01..10 evaluate/visualize scripts
      *.csv                      # result files at root (hybrid_results.csv etc.)
      results/

Scripts use PROJECT_ROOT-relative paths. My existing steering logic is in
src/4_generation/02_steered_inference.py, reuse its LexicalSteeringProcessor and
EMOTION_LEXICON as the reference implementation. My RoBERTa emotion judge
(SamLowe/roberta-base-go_emotions, mapped to Plutchik-8) is the evaluator used across
src/5_evaluation, reuse it, do not build a new judge.

## The fixed test set (critical for fair comparison)
Every new run MUST use the SAME 160-prompt test set that produced hybrid_results.csv:
20 stories per emotion for Joy, Trust, Fear, Surprise, Sadness, Anticipation, and 40
for Anger (Disgust merged into Anger). Prompt format: "Emotion: [TARGET] | Story: [PROMPT]".
Output CSV schema for every run: columns EXACTLY [Target, Detected, Story]. This keeps
new results drop-in compatible with my existing evaluation and figure code.

## IMPORTANT correction to bake in
My previously submitted paper listed the baseline (GPT-2 Medium, LoRA r8) at 6.67%.
On re-verification against the raw CSV the correct figure is 10.00% (N=70). Use 10.00%
everywhere. Do not reintroduce 6.67% except where it correctly refers to the
aggressive-steering b=15 collapse run (which genuinely is 6.67%, N=120).

## Tasks (build in this order)

### Task A, reviewer comment 4: expanded ablation (fastest, reuses existing code)
1. Lexicon-size ablation. Create small / medium / large versions of EMOTION_LEXICON
   (about 5, 15, 30 words per emotion). Run steered inference with each on the 160-prompt
   set. Output: lexicon_small_results.csv, lexicon_medium_results.csv, lexicon_large_results.csv.
2. Decoding-strategy ablation. Re-run the Hybrid config (GPT-2 Large + LoRA + steering,
   boost=5.0) with greedy, top-k (k=50), and top-p (p=0.92, current default) decoding.
   Output: decoding_greedy_results.csv, decoding_topk_results.csv, decoding_topp_results.csv.
3. LoRA rank ablation (8/32/64/128) already exists in my prior CSVs. Do NOT re-run it,
   just make sure the summary script (below) reads those existing files.

### Task B, reviewer comment 2: newer lightweight model baselines (inference only, NO fine-tuning)
Prompt these models on the SAME 160 prompts, then score with my RoBERTa judge. All must
run inference-only in 6GB VRAM (use 4-bit quantization via bitsandbytes where needed).
Start with these two (add Gemma-2B or TinyLlama-1.1B later only if time allows):
  - microsoft/Phi-3-mini-4k-instruct
  - Qwen/Qwen2.5-1.5B-Instruct
Wrap my prompt content in each model's correct chat template (do NOT feed raw text to
instruction-tuned models). Output: baseline_phi3_results.csv, baseline_qwen_results.csv.

### Task C, reviewer comment 3: PPLM controllable-generation baseline (the real new work)
Implement PPLM (Plug and Play Language Model) as one additional CTG baseline beyond GPT-2
variants. Use my existing RoBERTa emotion classifier as PPLM's attribute model. Run on
GPT-2 (same backbone family, fair comparison), same 160 prompts.
Output: baseline_pplm_results.csv. Skip GeDi entirely, one strong CTG baseline is enough.
Timebox PPLM: if it will not run stably in 6GB or gets too slow, stop and tell me, then
fall back to delivering Tasks A and B only. Do not burn hours forcing it.

### Task D: consolidated summary + 300 DPI figures
After each run, append a row to results/all_configs_summary.csv with columns:
  config_name, model, technique, N, top1_accuracy, then per-emotion F1 for
  Joy, Fear, Sadness, Anger, Trust, Surprise, Anticipation.
Then regenerate all publication figures at 300 DPI into results/figures/ using a single
script (results/make_figures.py). Figures needed:
  - Fig1: full-config Top-1 accuracy bar chart (all runs including new baselines),
    with a dashed 12.5% random-chance line.
  - Fig2: row-normalized confusion matrix for the Hybrid model (Blues colormap).
  - Fig3: per-emotion F1 grouped bars (baseline vs steered vs hybrid, plus new baselines).
  - Fig4: beta ablation line (b=1 no effect, b=5 sweet spot, b=15 collapse).
  - Fig5 (new): the new lightweight baselines (Phi-3, Qwen, PPLM) vs my Hybrid, one bar chart.
Use serif fonts, dpi=300, tight_layout, bbox_inches='tight', save as PNG. I have a
reference make_figures.py (attached separately) you can adapt for Figs 1 to 4.

## Engineering standards (apply to all tasks)
- Follow my "research paper mindset": each experiment folder saves config.yaml, seed,
  git commit hash, model/dataset versions, hardware, CUDA/Python/package versions, run
  command, timestamp. Use experiments/experiment_XXX/ structure. Reproducible from one command.
- Continuous local checkpoint/output saving so a crash never loses a run.
- Heavily commented, layman-first then technical, one idea at a time.
- No em dashes anywhere in code comments, docstrings, or output text. Use commas or parentheses.
- Every script prints a final summary: N, Top-1 accuracy, per-emotion F1, so I can
  sanity-check before the numbers ever reach the paper.

## What I will bring back for the paper (do not write paper text here)
Once these run, I will upload the new *_results.csv files, all_configs_summary.csv, and
the results/figures/*.png to a separate paper-writing session. Keep every output CSV in
the [Target, Detected, Story] schema.

Start with Task A. Confirm the plan and show me the file structure you will create
BEFORE writing any model-loading or training code.
