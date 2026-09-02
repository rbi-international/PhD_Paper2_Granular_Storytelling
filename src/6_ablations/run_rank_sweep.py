"""
run_rank_sweep.py

Task: clean LoRA-rank sweep, reviewer comment 4, the rank angle.

Layman note: a LoRA adapter has a size dial called "rank". A bigger rank means more
trainable knobs bolted onto the frozen model. The reviewer asked whether we checked that
dial. This script trains the SAME recipe four times, changing only that one number, then
scores all four the same way. If the scores barely move, rank is not what makes the method
work, and the steering is.

WHAT IS FROZEN. Everything except r is copied from the production training script,
src/3_training/06_train_gpt2_large_optimized.py: the dataset, epochs, learning rate, batch
size, gradient accumulation, sequence length, target modules and modules_to_save. Those
values are not just copied by hand, they are re-read from that file at startup and checked
(see verify_training_source). If somebody edits the production recipe, this script stops
rather than silently training a different thing and calling it the same.

THE ALPHA DECISION (deliberate, do not "fix" it). lora_alpha is held at 64 for all four
ranks because that is what production used (r=32, alpha=64). The training convention was
alpha = 2*rank, so holding alpha fixed makes the effective scaling alpha/rank swing from
8.0 at rank-8 down to 0.5 at rank-128. That is intentional and gets reported: a flat sweep
despite a 16x swing in scaling is stronger evidence, not weaker.

WHAT THIS SWEEP DOES AND DOES NOT SHOW (read before writing the paper sentence).
modules_to_save trains lm_head and wte in FULL at every rank. For GPT-2 Large that is
about 128.6M parameters, and it dominates the trainable total:

    rank 8    LoRA   5.9M  +  128.6M  =  134.5M trainable   (LoRA is  4.4%)
    rank 32   LoRA  23.6M  +  128.6M  =  152.2M trainable   (LoRA is 15.5%)
    rank 64   LoRA  47.2M  +  128.6M  =  175.8M trainable   (LoRA is 26.8%)
    rank 128  LoRA  94.4M  +  128.6M  =  223.0M trainable   (LoRA is 42.3%)

So a 16x swing in rank is only a 1.66x swing in total trainable capacity. A flat sweep
therefore supports "LoRA rank specifically does not matter under steering". It does NOT
support "adapter capacity is not the operative factor", because capacity is never actually
made small. The paper must use the narrow claim. The capacity-in-general argument is
carried elsewhere, by the model-size comparison, the lexicon-size cliff and decoding.

SEEDING. The production script never set a seed, so it inherited TrainingArguments' default
of 42, and its LoRA weights were initialised from an unseeded RNG. This script sets seed 42
explicitly (the same value production inherited) AND seeds before adapter construction, so
the four ranks differ only in r and not in random draw. Consequence to be honest about:
rank-32 here is a faithful RE-TRAINING of the production recipe, not a bit-identical
reproduction of the production artifact, so the anchor check is "lands near 38.12%", never
"matches exactly".

Usage:
    python src/6_ablations/run_rank_sweep.py --rank 32 --smoke-steps 10   (timing probe)
    python src/6_ablations/run_rank_sweep.py --rank 32                    (train + evaluate)

No em dashes anywhere (project style rule).
"""
import argparse
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

# The one dial under test, and the value held fixed across all four.
RANKS = [8, 32, 64, 128]
LORA_ALPHA = 64
SEED = 42

# Frozen copies of the production training recipe. verify_training_source() checks each of
# these against the real file, so this block cannot drift from the script it claims to copy.
TRAIN_SOURCE = os.path.join("src", "3_training", "06_train_gpt2_large_optimized.py")
FROZEN = {
    "MODEL_ID": "gpt2-large",
    "MAX_LENGTH": 256,
    "BATCH_SIZE": 1,
    "GRAD_ACCUMULATION": 32,
    "LEARNING_RATE": 2e-4,
    "EPOCHS": 10,
    "LORA_ALPHA": 64,
    "LORA_DROPOUT": 0.1,
    "TARGET_MODULES": ["c_attn", "c_proj", "c_fc"],
    "MODULES_TO_SAVE": ["lm_head", "wte"],
    "DATA_FILE": "instruction_train_ready.csv",
}

DATA_PATH = os.path.join(common.PROJECT_ROOT, "data", "processed", FROZEN["DATA_FILE"])
SWEEP_ROOT = os.path.join(common.PROJECT_ROOT, "models", "rank_sweep")


# ---------------------------------------------------------------------------
# Drift guard on the training recipe
# ---------------------------------------------------------------------------
def verify_training_source(verbose=True):
    """
    Re-read the production training script and confirm our frozen copy still matches it.

    Same idea as the SteeringProcessor SHA guard in common.py, applied to hyperparameters.
    The claim "everything except rank is identical to production" is only worth making if
    something actually checks it, so this checks it on every run.

    Rank itself is deliberately NOT checked: production is r=32 and this sweep varies it.
    """
    path = os.path.join(common.PROJECT_ROOT, TRAIN_SOURCE)
    if not os.path.exists(path):
        raise SystemExit(f"Missing training reference: {TRAIN_SOURCE}")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    def scalar(name):
        match = re.search(rf"^{name}\s*=\s*([^\s#]+)", text, re.MULTILINE)
        if not match:
            raise SystemExit(f"Could not find {name} in {TRAIN_SOURCE}")
        return match.group(1).strip().strip('"').strip("'")

    def listing(name):
        match = re.search(rf"{name}\s*=\s*\[(.*?)\]", text, re.DOTALL)
        if not match:
            raise SystemExit(f"Could not find {name} in {TRAIN_SOURCE}")
        return re.findall(r'"([^"]+)"', match.group(1))

    mismatches = []

    def check(label, actual, expected):
        if actual != expected:
            mismatches.append(f"{label}: source has {actual!r}, we froze {expected!r}")

    check("MODEL_ID", scalar("MODEL_ID"), FROZEN["MODEL_ID"])
    check("MAX_LENGTH", int(scalar("MAX_LENGTH")), FROZEN["MAX_LENGTH"])
    check("BATCH_SIZE", int(scalar("BATCH_SIZE")), FROZEN["BATCH_SIZE"])
    check("GRAD_ACCUMULATION", int(scalar("GRAD_ACCUMULATION")), FROZEN["GRAD_ACCUMULATION"])
    check("LEARNING_RATE", float(scalar("LEARNING_RATE")), FROZEN["LEARNING_RATE"])
    check("EPOCHS", int(scalar("EPOCHS")), FROZEN["EPOCHS"])
    check("LORA_ALPHA", int(scalar("LORA_ALPHA")), FROZEN["LORA_ALPHA"])
    check("target_modules", listing("target_modules"), FROZEN["TARGET_MODULES"])
    check("modules_to_save", listing("modules_to_save"), FROZEN["MODULES_TO_SAVE"])
    if FROZEN["DATA_FILE"] not in text:
        mismatches.append(f"data file {FROZEN['DATA_FILE']} no longer referenced in source")
    if "lora_dropout=0.1" not in text.replace(" ", ""):
        mismatches.append("lora_dropout is no longer 0.1 in source")

    if mismatches:
        raise SystemExit(
            "Training recipe has DRIFTED from the production script.\n  "
            + "\n  ".join(mismatches)
            + "\n\nThe sweep claims to hold everything but rank identical to production.\n"
              "Reconcile deliberately, then update FROZEN in this file."
        )
    if verbose:
        print(f"  training recipe verified against {TRAIN_SOURCE}")
    return True


# ---------------------------------------------------------------------------
# Paths and reuse
# ---------------------------------------------------------------------------
def adapter_dir_for(rank):
    return os.path.join(SWEEP_ROOT, f"rank_{rank}")


def checkpoint_dir_for(rank):
    """Per-rank checkpoint directory, so two ranks can never resume from each other."""
    return os.path.join(SWEEP_ROOT, "checkpoints", f"rank_{rank}")


def adapter_is_complete(rank):
    """
    True when this rank has already been trained and saved with the settings we expect.

    Checks the saved adapter_config.json rather than merely the directory existing, so a
    half-written or wrongly-configured adapter is retrained instead of silently reused.
    """
    path = adapter_dir_for(rank)
    config_file = os.path.join(path, "adapter_config.json")
    weights = os.path.join(path, "adapter_model.safetensors")
    if not (os.path.exists(config_file) and os.path.exists(weights)):
        return False
    try:
        with open(config_file, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return config.get("r") == rank and config.get("lora_alpha") == LORA_ALPHA


def has_checkpoint(rank):
    path = checkpoint_dir_for(rank)
    if not os.path.isdir(path):
        return False
    return any(name.startswith("checkpoint-") for name in os.listdir(path))


def expected_optimizer_steps(n_rows):
    """One optimizer step per BATCH_SIZE * GRAD_ACCUMULATION rows, repeated EPOCHS times."""
    per_epoch = math.ceil(n_rows / (FROZEN["BATCH_SIZE"] * FROZEN["GRAD_ACCUMULATION"]))
    return per_epoch * FROZEN["EPOCHS"], per_epoch


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def build_model_and_data(rank):
    """Load GPT-2 Large, attach a LoRA adapter of the requested rank, tokenize the data."""
    import pandas as pd
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, set_seed
    from peft import LoraConfig, TaskType, get_peft_model
    from datasets import Dataset

    # Seed BEFORE adapter construction so lora_A's random init is identical across ranks.
    # Production did not do this, which is why rank-32 is a re-training rather than a
    # bit-identical reproduction. Making it deterministic is the point of a sweep.
    set_seed(SEED)

    tokenizer = GPT2Tokenizer.from_pretrained(FROZEN["MODEL_ID"])
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(FROZEN["MODEL_ID"])
    model.gradient_checkpointing_enable()
    for param in model.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=LORA_ALPHA,
        lora_dropout=FROZEN["LORA_DROPOUT"],
        target_modules=FROZEN["TARGET_MODULES"],
        modules_to_save=FROZEN["MODULES_TO_SAVE"],
    )
    model = get_peft_model(model, peft_config)
    trainable, total = model.get_nb_trainable_parameters()

    frame = pd.read_csv(DATA_PATH)
    dataset = Dataset.from_pandas(frame[["formatted_text"]])

    def tokenize_function(examples):
        return tokenizer(
            examples["formatted_text"],
            truncation=True,
            max_length=FROZEN["MAX_LENGTH"],
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["formatted_text"])
    return model, tokenizer, tokenized, trainable, total


def train_one_rank(rank, smoke_steps=None):
    """
    Train one adapter. Returns a dict of facts for provenance, or raises SystemExit on OOM.

    smoke_steps turns this into a timing probe: it trains that many optimizer steps, saves
    nothing, and reports the measured rate so the full cost is known before committing
    hours to it.
    """
    import torch
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    probing = smoke_steps is not None

    if not probing and adapter_is_complete(rank):
        print(f"  adapter for rank {rank} already trained, reusing "
              f"{os.path.relpath(adapter_dir_for(rank), common.PROJECT_ROOT)}")
        return {"trained_now": False, "train_seconds": None}

    print(f"\n  building GPT-2 Large with LoRA rank {rank}, alpha {LORA_ALPHA} ...")
    setup_start = time.time()
    model, tokenizer, tokenized, trainable, total = build_model_and_data(rank)
    setup_seconds = time.time() - setup_start

    total_steps, per_epoch = expected_optimizer_steps(len(tokenized))
    lora_params = trainable - 128_632_320 if trainable > 128_632_320 else trainable
    print(f"  trainable {trainable:,} of {total:,} "
          f"({100 * trainable / total:.2f}%), LoRA share about {lora_params:,}")
    print(f"  {len(tokenized):,} rows, {per_epoch} optimizer steps per epoch, "
          f"{total_steps} total for {FROZEN['EPOCHS']} epochs")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    args = TrainingArguments(
        output_dir=checkpoint_dir_for(rank),
        per_device_train_batch_size=FROZEN["BATCH_SIZE"],
        gradient_accumulation_steps=FROZEN["GRAD_ACCUMULATION"],
        learning_rate=FROZEN["LEARNING_RATE"],
        num_train_epochs=FROZEN["EPOCHS"],
        logging_dir=os.path.join(common.PROJECT_ROOT, "logs", f"rank_sweep_r{rank}"),
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        seed=SEED,
        report_to="none",
        # Probe mode: cap the steps and write nothing to disk, so a few minutes of
        # measurement does not leave 600MB of checkpoints behind.
        **({"max_steps": smoke_steps, "save_strategy": "no"} if probing else {}),
    )

    resume = (not probing) and has_checkpoint(rank)
    if resume:
        print(f"  resuming rank {rank} from an existing checkpoint")

    print(f"  training{' (PROBE)' if probing else ''} ...")
    train_start = time.time()

    # Trainer construction is INSIDE the guard on purpose. Trainer.__init__ moves the
    # model onto the GPU, which is the first large allocation of the run, so at rank 128
    # it is a likely place to run out of memory. Building it outside the guard would let
    # that OOM escape uncaught, with no experiment folder and no provenance written,
    # which is exactly the failure the guard exists to prevent.
    try:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer.train(resume_from_checkpoint=True if resume else None)
    except torch.cuda.OutOfMemoryError as exc:
        raise _oom(rank, exc, trainable, total)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        raise _oom(rank, exc, trainable, total)
    train_seconds = time.time() - train_start

    peak_gib = (round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
                if torch.cuda.is_available() else None)

    if probing:
        per_step = train_seconds / smoke_steps
        estimate = setup_seconds + per_step * total_steps
        print("\n--- timing probe, rank {} ---".format(rank))
        print(f"  setup (load + tokenize)   {setup_seconds:.1f} s")
        print(f"  measured                  {smoke_steps} optimizer steps in {train_seconds:.1f} s")
        print(f"  rate                      {per_step:.2f} s per optimizer step")
        print(f"  full run needs            {total_steps} steps")
        print(f"  ESTIMATED PER RANK        {estimate / 60:.1f} min ({estimate / 3600:.2f} h)")
        print(f"  peak VRAM during probe    {peak_gib} GiB")
        print("\n  Nothing was saved. This was a measurement only.")
        return {
            "trained_now": False, "probe": True, "seconds_per_step": round(per_step, 3),
            "estimated_seconds": round(estimate, 1), "peak_vram_gib": peak_gib,
        }

    os.makedirs(adapter_dir_for(rank), exist_ok=True)
    model.save_pretrained(adapter_dir_for(rank))
    tokenizer.save_pretrained(adapter_dir_for(rank))
    print(f"\n  trained in {train_seconds / 60:.1f} min, peak VRAM {peak_gib} GiB")
    print(f"  adapter saved to {os.path.relpath(adapter_dir_for(rank), common.PROJECT_ROOT)}")

    return {
        "trained_now": True,
        "train_seconds": round(train_seconds, 1),
        "train_minutes": round(train_seconds / 60, 1),
        "setup_seconds": round(setup_seconds, 1),
        "trainable_params": int(trainable),
        "total_params": int(total),
        "optimizer_steps": total_steps,
        "peak_train_vram_gib": peak_gib,
        "resumed_from_checkpoint": bool(resume),
    }


def _oom(rank, exc, trainable, total):
    """
    Record an out-of-memory failure honestly and stop, without touching the summary.

    rank-128 may not fit in 6GB alongside a fully trained lm_head and wte. That is a
    reportable result for a resource-constrained paper, not a silent failure, so it gets
    its own experiment folder with provenance rather than vanishing into a traceback.
    """
    exp_dir = common.next_experiment_dir(f"rank_{rank}_oom")
    common.write_provenance(exp_dir, {
        "config_name": f"rank_{rank}",
        "outcome": "OUT_OF_MEMORY during training",
        "lora_rank": rank,
        "lora_alpha": LORA_ALPHA,
        "trainable_params": int(trainable),
        "total_params": int(total),
        "note": ("rank exceeded the 6GB VRAM budget during training. Reported as a "
                 "limit of the hardware envelope, which is on-thesis for a "
                 "resource-constrained paper, not as a missing data point."),
        "error": str(exc)[:2000],
    })
    print(f"\n  OUT OF MEMORY at rank {rank}.")
    print(f"  Recorded in {os.path.relpath(exp_dir, common.PROJECT_ROOT)}")
    print("  The summary was NOT touched, so the rest of the sweep stays clean.")
    return SystemExit(2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Clean LoRA rank sweep")
    parser.add_argument("--rank", required=True, type=int, choices=RANKS)
    parser.add_argument("--smoke-steps", type=int, default=None,
                        help="timing probe only: train this many optimizer steps, save "
                             "nothing, report the extrapolated cost of a full run")
    args = parser.parse_args()
    rank = args.rank

    print(f"=== rank sweep, r={rank}, alpha={LORA_ALPHA} "
          f"(effective scaling {LORA_ALPHA / rank:g}) ===\n")
    verify_training_source()

    training = train_one_rank(rank, smoke_steps=args.smoke_steps)
    if args.smoke_steps is not None:
        return

    caveat = (
        f"rank swept at fixed alpha=64 matching production, so effective scaling "
        f"alpha/rank = {LORA_ALPHA / rank:g}. lm_head and wte are trained in full at "
        f"every rank (about 128.6M params), so total trainable capacity varies only "
        f"1.66x across the sweep while rank varies 16x. Supports a claim about LoRA "
        f"rank specifically, not about adapter capacity in general."
    )

    common.execute_run(
        config_name=f"rank_{rank}",
        output_csv=f"rank_{rank}_results.csv",
        technique=f"LoRA + lexical steering (b=5.0), LoRA rank {rank}, alpha 64",
        tier="15",
        boost_factor=5.0,
        decoding="topp",
        model_label=f"GPT-2 Large + LoRA r{rank}",
        caveat=caveat,
        adapter_path=adapter_dir_for(rank),
        extra_provenance={
            "lora_rank": rank,
            "lora_alpha": LORA_ALPHA,
            "effective_scaling_alpha_over_rank": LORA_ALPHA / rank,
            "training_seed": SEED,
            "training_seed_note": (
                "42 is the value production inherited from the TrainingArguments default, "
                "made explicit here rather than chosen arbitrarily. This script also seeds "
                "before adapter construction, which production did not, so LoRA init is "
                "identical across the four ranks. rank-32 is therefore a faithful "
                "re-training of the production recipe, not a bit-identical reproduction "
                "of the production artifact."
            ),
            "training_recipe_source": TRAIN_SOURCE.replace("\\", "/"),
            "training_recipe_frozen": FROZEN,
            "training": training,
            "capacity_note": (
                "modules_to_save trains lm_head and wte in full at every rank, about "
                "128.6M params, which dominates the trainable total. Rank varies 16x but "
                "total trainable capacity varies only 1.66x (134.5M at r8 to 223.0M at "
                "r128). This sweep isolates LoRA rank, not adapter capacity in general."
            ),
        },
    )


if __name__ == "__main__":
    main()
