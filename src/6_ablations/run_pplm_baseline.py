"""
run_pplm_baseline.py

Task C, reviewer comment 3: a stronger controllable-generation baseline beyond GPT-2
variants. PPLM (Plug and Play Language Models, Dathathri et al. 2020).

Layman note: our own method nudges the model by adding a bonus to emotion words, which
costs nothing extra because it happens during the normal forward pass. PPLM does
something far more expensive. At every single word it generates, it asks a separate
classifier "how emotional is this so far?", works out which direction would make the
answer stronger, and edits the model's internal memory in that direction before
committing to the word. That means running the model backwards, once per word.

Why this is the right baseline for comment 3: PPLM is gradient-based CTG, which is a
genuinely different mechanism from our forward-pass-only lexical steering. Whether it
even runs on 6GB is itself the interesting question for a paper about constrained
hardware.

THE ATTRIBUTE MODEL, AND AN HONEST NOTE ABOUT IT
PPLM needs an attribute model whose loss is differentiable with respect to the language
model's activations. Our RoBERTa judge cannot do that job: it consumes discrete token
ids, so there is no gradient path from its loss back into GPT-2's past key/values. We
therefore implement the standard PPLM discriminator, a linear classifier over GPT-2's
own mean-pooled hidden states, which is exactly what the original paper uses.

This is faithful to PPLM rather than a deviation from it, and it has a side benefit: the
attribute model (GPT-2 head) and the evaluation judge (RoBERTa) stay separate models, so
PPLM is not being graded by the same network that steers it. Every other config in this
project is judged by RoBERTa too, so the comparison stays like for like.

Controls, identical to Task A and Task B
  - the same revision manifest, the same 160 varied held-out stems
  - the same RoBERTa judge and classify() for evaluation
  - the same max_new_tokens=60 generation budget
  - output in the [Target, Detected, Story] schema

Usage:
    python src/6_ablations/run_pplm_baseline.py --smoke 2      (smoke test, VRAM + latency)
    python src/6_ablations/run_pplm_baseline.py                (full 160, resume safe)
    python src/6_ablations/run_pplm_baseline.py --backbone gpt2-medium

No em dashes anywhere (project style rule).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import judge

# PPLM hyperparameters, from Dathathri et al. 2020 for the discriminator variant.
NUM_ITERATIONS = 3      # gradient steps per generated token, the expensive loop
STEP_SIZE = 0.02        # perturbation step
GAMMA = 1.0             # gradient normalisation exponent
GM_SCALE = 0.95         # geometric mean fusion of perturbed and unperturbed distributions
KL_SCALE = 0.01         # keeps the perturbed distribution near the original
GRAD_LENGTH = 10000     # no window restriction

HEAD_PATH = os.path.join(common.PROJECT_ROOT, "models", "pplm_attribute_head.pt")
TRAIN_CSV = os.path.join(common.PROJECT_ROOT, "data", "processed", "final_train_ready.csv")

# The eight generated classes, matching the lexicon keys and the manifest.
CLASSES = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation"]


# ---------------------------------------------------------------------------
# The PPLM attribute model: a linear head over GPT-2 mean-pooled hidden states
# ---------------------------------------------------------------------------
def train_attribute_head(model, tokenizer, device, hidden_size):
    """
    Train (or load) the discriminator. Cheap: GPT-2 stays frozen, we only fit a single
    linear layer over mean-pooled hidden states, which is the original PPLM design.
    """
    import torch
    import torch.nn as nn
    import pandas as pd

    head = nn.Linear(hidden_size, len(CLASSES)).to(device)
    if os.path.exists(HEAD_PATH):
        state = torch.load(HEAD_PATH, map_location=device)
        if state["hidden_size"] == hidden_size:
            head.load_state_dict(state["state_dict"])
            print(f"  attribute head loaded from {os.path.relpath(HEAD_PATH, common.PROJECT_ROOT)}")
            head.eval()
            return head
        print("  cached head has the wrong hidden size for this backbone, retraining")

    frame = pd.read_csv(TRAIN_CSV)
    frame = frame[frame["emotion_label"].isin(CLASSES)].reset_index(drop=True)
    print(f"  training attribute head on {len(frame)} labelled rows")

    # Precompute frozen hidden states once, then fit the linear layer on them.
    features, labels = [], []
    with torch.no_grad():
        for start in range(0, len(frame), 16):
            batch = frame.iloc[start:start + 16]
            encoded = tokenizer(list(batch["text"].astype(str)), return_tensors="pt",
                                padding=True, truncation=True, max_length=128).to(device)
            hidden = model.transformer(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            features.append(pooled.float().cpu())
            labels.extend(CLASSES.index(e) for e in batch["emotion_label"])
    features = torch.cat(features).to(device)
    labels = torch.tensor(labels, device=device)

    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(60):
        optimizer.zero_grad()
        loss = loss_fn(head(features), labels)
        loss.backward()
        optimizer.step()
    accuracy = (head(features).argmax(-1) == labels).float().mean().item()
    print(f"  attribute head trained: loss {loss.item():.3f}, train accuracy {accuracy*100:.1f}%")

    os.makedirs(os.path.dirname(HEAD_PATH), exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "hidden_size": hidden_size,
                "classes": CLASSES}, HEAD_PATH)
    head.eval()
    return head


# ---------------------------------------------------------------------------
# The PPLM perturbation step
# ---------------------------------------------------------------------------
def perturb_past(past, model, head, last_token, target_class, device, unpert_logits):
    """
    One PPLM update. Runs NUM_ITERATIONS gradient steps that push the LM's cached key and
    value activations in the direction the attribute classifier wants, while a KL term
    holds the perturbed distribution close to the original. This backward pass, repeated
    per generated token, is what makes PPLM expensive and is the likely 6GB failure point.
    """
    import torch
    import torch.nn.functional as F

    grad_accumulator = [torch.zeros_like(layer) for layer in past]
    unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

    for _ in range(NUM_ITERATIONS):
        perturbation = [p.clone().detach().requires_grad_(True) for p in grad_accumulator]
        perturbed_past = [p + d for p, d in zip(past, perturbation)]

        outputs = model(last_token, past_key_values=_to_cache(perturbed_past),
                        output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        pooled = hidden.mean(dim=1)

        class_logits = head(pooled.float())
        target = torch.tensor([target_class], device=device)
        loss = F.cross_entropy(class_logits, target)

        if KL_SCALE > 0:
            pert_probs = F.softmax(outputs.logits[:, -1, :], dim=-1)
            kl = (unpert_probs * ((unpert_probs + 1e-10).log()
                                  - (pert_probs + 1e-10).log())).sum()
            loss = loss + KL_SCALE * kl

        grads = torch.autograd.grad(loss, perturbation, allow_unused=True)
        for index, grad in enumerate(grads):
            if grad is None:
                continue
            norm = grad.norm() + 1e-10
            grad_accumulator[index] = grad_accumulator[index] - STEP_SIZE * (grad / norm ** GAMMA)
        del outputs, grads

    return [(p + d).detach() for p, d in zip(past, grad_accumulator)]


def _to_cache(past_list):
    """
    PPLM's maths operates on a flat list of key and value tensors, but transformers 5.x
    passes caches around as a DynamicCache object. Rebuild one from the perturbed tensors
    so they can be fed back into the model.

    Note on the API: transformers 5.3.0 dropped Cache.from_legacy_cache and made the
    object non-subscriptable. The constructor accepts ddp_cache_data, an iterable of
    (key, value) tuples, which is the supported route in this version.
    """
    from transformers import DynamicCache
    pairs = [(past_list[i], past_list[i + 1]) for i in range(0, len(past_list), 2)]
    return DynamicCache(ddp_cache_data=pairs)


def _from_cache(cache):
    """
    Flatten a DynamicCache into a list of key and value tensors.

    transformers 5.3.0 exposes cache.layers, each a DynamicLayer carrying .keys and
    .values. The older to_legacy_cache() helper no longer exists in this version.
    """
    flat = []
    for layer in cache.layers:
        flat.extend([layer.keys, layer.values])
    return flat


def generate_pplm(model, tokenizer, head, device, row, max_new_tokens):
    """One PPLM-steered continuation, seeded per row exactly like every other config."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(row["seed"])
    target_class = CLASSES.index(row["lexicon_key"])
    prompt = f"Emotion: {row['lexicon_key']} | Story: {row['story_start']}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated = input_ids
    past = None
    for _ in range(max_new_tokens):
        if past is None:
            with torch.no_grad():
                outputs = model(generated[:, :-1], use_cache=True)
            past = _from_cache(outputs.past_key_values)

        last_token = generated[:, -1:]
        with torch.no_grad():
            unpert = model(last_token, past_key_values=_to_cache(past))
        unpert_logits = unpert.logits

        perturbed_past = perturb_past(past, model, head, last_token, target_class,
                                      device, unpert_logits)
        with torch.no_grad():
            pert_out = model(last_token, past_key_values=_to_cache(perturbed_past))

        # Fuse perturbed and original distributions, as in the paper.
        pert_probs = F.softmax(pert_out.logits[:, -1, :], dim=-1)
        unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
        fused = (pert_probs ** GM_SCALE) * (unpert_probs ** (1 - GM_SCALE))
        fused = fused / fused.sum()

        next_token = torch.multinomial(fused, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        past = _from_cache(pert_out.past_key_values)

        if next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text.split("Story:")[1].strip() if "Story:" in text else text


def main():
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    parser = argparse.ArgumentParser(description="PPLM controllable-generation baseline")
    parser.add_argument("--backbone", default="gpt2-large",
                        choices=["gpt2-large", "gpt2-medium", "gpt2"])
    parser.add_argument("--smoke", type=int, default=None,
                        help="run only N rows, report VRAM and latency, write nothing final")
    args = parser.parse_args()

    config_name = "baseline_pplm"
    print(f"=== {config_name} (PPLM, backbone {args.backbone}) ===\n")
    judge.verify_against_source()
    print("EMOTION_MAP verified identical to 07_evaluate_hybrid.py (the 47.50% run).\n")

    rows = common.load_prompt_set()
    if args.smoke:
        rows = rows[:args.smoke]
        print(f"SMOKE TEST: {len(rows)} rows, measuring peak VRAM and latency.\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained(args.backbone)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.backbone).to(device)
    model.eval()
    for parameter in model.parameters():          # frozen LM, PPLM never trains it
        parameter.requires_grad_(False)
    print(f"  {args.backbone} loaded on {device}")

    head = train_attribute_head(model, tokenizer, device, model.config.n_embd)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if not args.smoke:
        exp_dir = common.find_or_create_experiment_dir(config_name, len(rows))
        common.write_provenance(exp_dir, {
            "config_name": config_name,
            "technique": "PPLM gradient-based CTG (Dathathri et al. 2020)",
            "model": f"{args.backbone} (frozen) + PPLM discriminator head",
            "backbone": args.backbone,
            "attribute_model": "linear head over GPT-2 mean-pooled hidden states",
            "attribute_model_note": (
                "RoBERTa cannot serve as PPLM's attribute model: it consumes discrete "
                "tokens, so no gradient path exists to GPT-2's past key/values. This is "
                "the standard PPLM discriminator from the original paper. Attribute model "
                "and evaluation judge therefore stay separate, avoiding circularity."),
            "pplm_num_iterations": NUM_ITERATIONS,
            "pplm_step_size": STEP_SIZE,
            "pplm_gamma": GAMMA,
            "pplm_gm_scale": GM_SCALE,
            "pplm_kl_scale": KL_SCALE,
            "max_new_tokens": common.MAX_NEW_TOKENS,
            "manifest": "data/processed/test_prompts_160.csv",
            "manifest_rows": len(rows),
            "judge": judge.TEACHER_NAME,
        })
        writer = common.ResumableWriter(
            os.path.join(exp_dir, "partial_results.csv"),
            os.path.join(exp_dir, "results.csv"))
        done = writer.completed_prompt_ids()
        if done:
            print(f"  resuming, {len(done)} of {len(rows)} rows already complete")
    else:
        exp_dir, writer, done = None, None, set()

    judge_model, judge_tokenizer, _ = judge.load_judge(device)
    print(f"  judge loaded\n")

    latencies = []
    for index, row in enumerate(rows):
        if row["prompt_id"] in done:
            continue
        started = time.time()
        try:
            story = generate_pplm(model, tokenizer, head, device, row,
                                  common.MAX_NEW_TOKENS)
        except torch.cuda.OutOfMemoryError as error:
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"\n  CUDA OUT OF MEMORY at row {row['prompt_id']}")
            print(f"  backbone      : {args.backbone}")
            print(f"  peak VRAM     : {peak:.2f} GiB")
            print(f"  error         : {str(error)[:200]}")
            print("\n  This is outcome B: gradient-based CTG does not fit the 6GB budget.")
            return
        elapsed = time.time() - started
        latencies.append(elapsed)
        detected = judge.classify(story, judge_model, judge_tokenizer, device)

        if args.smoke:
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if device == "cuda" else 0
            print(f"  --- prompt {row['prompt_id']} target={row['target_label']} ---")
            print(f"      story    : {' '.join(story.split())[:170]}")
            print(f"      detected : {detected}")
            print(f"      latency  : {elapsed:.1f} s/story")
            print(f"      peak VRAM: {peak:.2f} GiB\n")
        else:
            writer.append(row["prompt_id"], row["target_label"], detected, story)
            if (index + 1) % 10 == 0:
                mean = sum(latencies) / len(latencies)
                left = (len(rows) - index - 1) * mean / 60
                print(f"  {index+1}/{len(rows)}  {mean:.1f} s/story  ~{left:.0f} min left")

    peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if device == "cuda" else 0
    mean_latency = sum(latencies) / max(len(latencies), 1)
    print(f"\n  peak VRAM     : {peak:.2f} GiB")
    print(f"  mean latency  : {mean_latency:.1f} s/story")

    if args.smoke:
        projected = mean_latency * 160 / 60
        print(f"  projected full run: {projected:.0f} min for 160 stories")
        print("\nSmoke test done. Report before running the full set.")
        return

    root_copy = os.path.join(common.PROJECT_ROOT, f"{config_name}_results.csv")
    count, _ = writer.finalize(extra_paths=[root_copy])
    final = writer.rows()
    metrics = common.summarize([r["target_label"] for r in final],
                               [r["detected"] for r in final])
    metrics["peak_vram_gib"] = round(peak, 2)
    metrics["mean_seconds_per_story"] = round(mean_latency, 1)
    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    common.print_summary(config_name, metrics)
    common.append_to_summary(config_name, f"{args.backbone} + PPLM",
                             "PPLM gradient-based CTG", metrics,
                             caveat=f"peak {peak:.2f} GiB, {mean_latency:.1f} s/story")
    print(f"\n  wrote {count} rows to {config_name}_results.csv")


if __name__ == "__main__":
    main()
