"""
run_lm_baseline.py

Task B, reviewer comment 2: how do newer lightweight models do on the same task?

Layman note: we give a modern small model the same 160 story openings and the same
emotion targets, then score it with the same automatic marker. No fine-tuning, no
steering. This answers "could you have just used a newer model instead?"

Inference only. 4-bit quantisation so Phi-3 fits in 6GB.

What is held constant (or the comparison is unusable)
  - the same revision manifest, the same 160 varied held-out stems
  - the same RoBERTa judge, the same classify(), the same target labels
  - the same max_new_tokens=60 as every GPT-2 run
The equal generation budget is a deliberate fairness constraint, not an oversight.
Longer generations would give the judge more text to work with and would unfairly
favour whichever model was allowed to write more.

What varies BY DESIGN, and why that is correct
Prompt format varies per model. Phi-3 and Qwen are instruction tuned, so they get
their own native chat template. Feeding them the raw "Emotion: X | Story:" tag that
GPT-2 was fine-tuned on would handicap them and invite the objection that we prompted
them wrong. Forcing one format on every model would itself be the confound.

Two deliberate choices in the prompt, both recorded here so the asymmetry is on record

1. The stem PRIMES THE ASSISTANT TURN rather than being requested in the instruction.
   We append the stem to the assistant's response opening, so the model continues from
   it exactly as GPT-2 does. Asking an instruction-tuned model to "begin with exactly
   X" would impose an instruction-following burden GPT-2 never had to meet, and models
   often add a preamble ("Sure! Here's a story...") that would then be scored by the
   judge. Priming removes the reproduction burden entirely.

2. The instruction is at least as explicit as GPT-2's conditioning, never less.
   GPT-2 received a bare "Emotion: X | Story:" tag. The baselines receive an explicit
   directive to express the emotion. That tilts slightly in the baselines' favour,
   which is the safe direction: we would rather understate our own model's advantage
   than overstate it. If a baseline still loses despite the more explicit instruction,
   the result is stronger. If it wins, the asymmetry is already disclosed.

Usage:
    python src/6_ablations/run_lm_baseline.py --model qwen
    python src/6_ablations/run_lm_baseline.py --model phi3
    python src/6_ablations/run_lm_baseline.py --model qwen --limit 3   (smoke test)

Output: baseline_qwen_results.csv, baseline_phi3_results.csv  [Target, Detected, Story]

No em dashes anywhere (project style rule).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import judge

MODELS = {
    "phi3": {
        "id": "microsoft/Phi-3-mini-4k-instruct",
        "label": "Phi-3-mini-4k-instruct (4-bit, no fine-tuning)",
    },
    "qwen": {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "label": "Qwen2.5-1.5B-Instruct (4-bit, no fine-tuning)",
    },
}

# Disgust is generated but scored as Anger, matching the manifest and every GPT-2 run.
INSTRUCTION = (
    "Continue the story so that it clearly expresses the emotion {emotion}. "
    "Write two or three sentences of narrative prose. "
    "Do not explain, label, or comment on the emotion, just tell the story."
)


def load_baseline(model_key, device=None):
    """Load an instruction-tuned baseline in 4-bit. Inference only, never trained."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    model_id = MODELS[model_key]["id"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    # trust_remote_code is deliberately OFF. The Phi-3 repo ships a vendored
    # modeling_phi3.py that predates this transformers version and crashes with
    # KeyError: 'type' on config.rope_scaling, because the rope config format changed.
    # transformers 5.3.0 implements Phi-3 and Qwen2 natively, so the built-in classes
    # are both correct and current. This avoids touching the environment, which has to
    # stay reproducible since it produced the 47.50% result.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer, device


def build_prompt(tokenizer, emotion, stem):
    """
    Native chat template, with the stem PRIMING the assistant turn.

    The assistant's reply is opened with the stem, so the model continues from it the
    way GPT-2 does, instead of being asked to reproduce it. Returns the full prompt
    string, which already ends with the stem.
    """
    messages = [{"role": "user", "content": INSTRUCTION.format(emotion=emotion)}]
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prefix + stem


def generate_one(model, tokenizer, device, row):
    """
    One continuation. Seeded per row from the frozen manifest, exactly as the GPT-2
    runs are, so a crash resume reproduces identical text.

    Only the NEW tokens are decoded, then the stem is prepended. The chat template
    scaffolding therefore never reaches the judge, and the Story column has the same
    shape as every GPT-2 run: stem followed by continuation.
    """
    import torch

    torch.manual_seed(row["seed"])
    prompt = build_prompt(tokenizer, row["lexicon_key"], row["story_start"])
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=common.MAX_NEW_TOKENS,
            repetition_penalty=common.REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **common.DECODING_PRESETS["topp"],
        )
    continuation = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

    # Join fix. Phi-3's tokenizer does not emit a leading space when continuing, so a
    # naive concatenation fuses words at the seam ("the bell" + "above" -> "bellabove").
    # Insert a space only when the continuation opens on an alphanumeric character, so
    # a continuation that legitimately starts with punctuation (a comma, a full stop,
    # a quote) is left alone. Qwen already emits the leading space, so this is a no-op
    # there and its committed results are unaffected.
    if continuation and continuation[0].isalnum():
        continuation = " " + continuation
    return (row["story_start"] + continuation).strip()


def main():
    parser = argparse.ArgumentParser(description="Newer lightweight model baselines")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--limit", type=int, default=None,
                        help="only run the first N prompts, for a smoke test")
    args = parser.parse_args()

    spec = MODELS[args.model]
    config_name = f"baseline_{args.model}"
    print(f"=== {config_name}: {spec['id']} ===\n")

    judge.verify_against_source()
    print("EMOTION_MAP verified identical to 07_evaluate_hybrid.py (the 47.50% run).\n")

    rows = common.load_prompt_set()
    if args.limit:
        rows = rows[:args.limit]
        print(f"SMOKE TEST: first {len(rows)} prompts only, nothing will be committed.\n")

    exp_dir = common.find_or_create_experiment_dir(config_name, len(rows))
    common.write_provenance(exp_dir, {
        "config_name": config_name,
        "technique": "prompting only, no fine-tuning, no steering",
        "model": spec["label"],
        "model_id": spec["id"],
        "quantization": "4-bit nf4, double quant, fp16 compute",
        "prompt_style": "native chat template, stem primes the assistant turn",
        "instruction": INSTRUCTION,
        "decoding": "topp",
        "decoding_params": common.DECODING_PRESETS["topp"],
        "max_new_tokens": common.MAX_NEW_TOKENS,
        "repetition_penalty": common.REPETITION_PENALTY,
        "manifest": "data/processed/test_prompts_160.csv",
        "manifest_rows": len(rows),
        "judge": judge.TEACHER_NAME,
        "fairness_notes": [
            "equal generation budget: max_new_tokens=60, same as every GPT-2 run",
            "same manifest, same judge, same target labels",
            "prompt format varies per model by design, forcing one format would confound",
            "stem primes the assistant turn, so no instruction-following burden GPT-2 never had",
            "instruction is at least as explicit as GPT-2's bare conditioning tag, never less",
        ],
    })

    writer = common.ResumableWriter(
        os.path.join(exp_dir, "partial_results.csv"),
        os.path.join(exp_dir, "results.csv"),
    )
    done = writer.completed_prompt_ids()
    if done:
        print(f"  resuming, {len(done)} of {len(rows)} rows already complete\n")

    model, tokenizer, device = load_baseline(args.model)
    judge_model, judge_tokenizer, _ = judge.load_judge(device)
    print(f"  baseline and judge loaded on {device}\n")

    for index, row in enumerate(rows):
        if row["prompt_id"] in done:
            continue
        story = generate_one(model, tokenizer, device, row)
        detected = judge.classify(story, judge_model, judge_tokenizer, device)
        writer.append(row["prompt_id"], row["target_label"], detected, story)
        if args.limit:
            print(f"  --- prompt {row['prompt_id']} target={row['target_label']} ---")
            print(f"      stem     : {row['story_start']}")
            print(f"      story    : {' '.join(story.split())[:180]}")
            print(f"      starts with stem verbatim: {story.startswith(row['story_start'])}")
            print(f"      detected : {detected}\n")
        elif (index + 1) % 20 == 0:
            print(f"  {index + 1}/{len(rows)} generated")

    if args.limit:
        print("Smoke test done. Inspect the stem handling above, then run without --limit.")
        return

    root_copy = os.path.join(common.PROJECT_ROOT, f"{config_name}_results.csv")
    count, _ = writer.finalize(extra_paths=[root_copy])
    final = writer.rows()
    metrics = common.summarize([r["target_label"] for r in final],
                               [r["detected"] for r in final])
    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    common.print_summary(config_name, metrics)
    common.append_to_summary(config_name, spec["label"],
                             "prompting only, no fine-tuning, no steering", metrics,
                             caveat="unassisted baseline, equal generation budget (60 tokens)")
    print(f"\n  wrote {count} rows to {config_name}_results.csv")
    print(f"  provenance: {os.path.relpath(exp_dir, common.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
