# CLAUDE.md

Claude Code reads this file automatically at the start of every session. It holds the
stable rules and context for this project. For the CURRENT task and open decisions, read
HANDOFF.md (updated between sessions).

## What this project is
Revision of a REJECTED journal paper (SN Computer Science, manuscript SNCS-D-26-04624).
New title: "GranularStory: Enhancing Emotional Fidelity in Narrative Generation using
Hybrid NSIT on Resource-Constrained Hardware."

The paper studies emotion-controlled story generation on a single RTX 3060 (6GB VRAM).
Core method: LoRA instruction-tuning of GPT-2 plus inference-time lexical steering (a
logit boost on emotion-specific words). This revision adds baselines and ablations the
reviewers asked for. No human evaluation is being done (my decision).

## Hard constraints (never violate)
- GPT-2 MUST remain the central fine-tuned model. My PhD Objective 2 requires it. Newer
  models (Phi-3, Qwen, etc.) are comparison baselines ONLY, never replacements.
- NO em dashes anywhere: not in code, comments, docstrings, printed output, or markdown.
  Use commas, parentheses, or plain text. This is a firm rule for all my work.
- Baseline (GPT-2 Medium, LoRA r8) accuracy is 10.00%, N=70. The submitted paper said
  6.67%; that was wrong and unsupported. Use 10.00% everywhere. The only correct use of
  6.67% is the aggressive-steering b=15 collapse run (N=120), which genuinely is 6.67%.
- Every result CSV uses columns EXACTLY: Target, Detected, Story. This keeps all runs
  drop-in compatible with the evaluation and figure scripts.

## Environment
Activate with BOTH commands every session (second prevents an OpenMP crash on Windows):
    conda activate study_torch
    export KMP_DUPLICATE_LIB_OK=TRUE
Verified stack: PyTorch 2.10.0+cu128, CUDA 12.8, cuDNN 9.10.2, transformers, peft,
datasets, bitsandbytes (already installed, no installs needed). RTX 3060 Laptop 6.44GB
VRAM, Windows. Do NOT downgrade torch/transformers; this env produced the 47.50% result.

## Repo layout
    data/processed/            final_train_ready.csv, instruction_train_ready.csv,
                               test_prompts_160.csv (frozen manifest)
    models/                    trained LoRA adapters (NOT in git, backed up separately)
    src/1_data_ingestion ... 5_evaluation/    original pipeline (do not edit)
    src/6_ablations/           this revision's ablation code
    experiments/               one folder per run: experiment_XXX/ with config.yaml etc.
    results/figures/           300 DPI figures
Scripts use PROJECT_ROOT-relative paths. Directories start with a digit, so they are not
importable as Python modules; copy shared code rather than importing across them, and
guard copies against drift from the source (see lexicon_tiers.py for the pattern).

## Reference implementations (source of truth, do not re-derive)
- Steering: src/5_evaluation/07_evaluate_hybrid.py (the 47.50% version). Its
  SteeringProcessor and EMOTION_LEXICON are canonical.
- Judge: RoBERTa GoEmotions mapped to Plutchik-8, as used across src/5_evaluation.
  Reuse it. Do not build a new judge.

## Engineering standards (research-paper mindset)
Every run saves full provenance: experiments/experiment_XXX/config.yaml with seed, git
commit hash, model and dataset versions, hardware, CUDA/Python/package versions, run
command, timestamp. Continuous per-row output flushing so a crash never loses a run
(resumable). Heavily commented, layman-first then technical. Every script prints a final
summary (N, Top-1 accuracy, per-emotion F1) so I can sanity-check before numbers reach
the paper. Reproducible from a single command.

## Workflow rule
Before writing model-loading or training code, confirm the plan and show the file
structure first. I gate the model-touching code. Pure data/scaffolding code you can build
without gating.
