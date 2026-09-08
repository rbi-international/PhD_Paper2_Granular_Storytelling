"""
common.py

Layman note: this is the shared toolbox every experiment uses. It knows how to load
the model, how to nudge it towards emotion words, how to write results down safely
so a crash costs nothing, and how to record exactly what was running at the time.
Nothing here decides an experiment. The runners do that.

Technical note, and the important idea in this file:
We do NOT reimplement the steering logic. We load the actual SteeringProcessor class
object out of src/5_evaluation/07_evaluate_hybrid.py, the file that produced the
47.50% headline result. There is therefore no second implementation that could drift
from the first. A pinned hash of the class body forms a second layer, so if anyone
edits that file the runs stop rather than silently measuring different steering.

Guard independence (verified by self_test() at every startup):
The lexicon ablation swaps the reference module's EMOTION_LEXICON global. The drift
guards read the FILE on disk, and lexicon_tiers.py imports only os and re, so it has
no handle on the module object at all. The swapped global is physically unreachable
from the guard's code path. The two channels cannot interfere.

No em dashes anywhere (project style rule).
"""
import contextlib
import csv
import hashlib
import importlib.util
import os
import platform
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import judge
import lexicon_tiers

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF_RELPATH = os.path.join("src", "5_evaluation", "07_evaluate_hybrid.py")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test_prompts_160.csv")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
SUMMARY_PATH = os.path.join(PROJECT_ROOT, "results", "all_configs_summary.csv")

# Model under test: GPT-2 Large plus the LoRA adapter behind the 47.50% result.
BASE_MODEL_ID = "gpt2-large"
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models", "gpt2_large_optimized")

# SHA256 of the normalized SteeringProcessor class body in the reference file.
# Re-pin this ONLY when deliberately accepting a change to the steering algorithm.
REF_STEERING_SHA256 = "e864537764c260ed616a7950a8290a72222d5bf702a15e407a890e7163f3d714"

# Scored label set. Disgust is generated but merged into Anger, so it is absent here.
EMOTIONS = ["Joy", "Fear", "Sadness", "Anger", "Trust", "Surprise", "Anticipation"]

# Generation settings held constant across the decoding ablation. Only the sampling
# rule varies, so the ablation measures decoding and nothing else. The topp preset is
# the configuration that produced 47.50%.
MAX_NEW_TOKENS = 60
REPETITION_PENALTY = 1.2
DECODING_PRESETS = {
    "topp": {"do_sample": True, "temperature": 0.8, "top_p": 0.92},
    "topk": {"do_sample": True, "temperature": 0.8, "top_k": 50},
    "greedy": {"do_sample": False},
}

_REF_MODULE = None
_LEXICON_SWAP_DEPTH = 0


# ---------------------------------------------------------------------------
# Reference binding and drift guards
# ---------------------------------------------------------------------------
def _load_reference_module():
    """
    Import 07_evaluate_hybrid.py by file path. Its directory name starts with a digit,
    so a normal import statement cannot reach it. Importing has no side effects: the
    module defines constants, a class and functions, and guards main() behind __main__.
    Registering in sys.modules lets inspect locate the source file later.
    """
    global _REF_MODULE
    if _REF_MODULE is None:
        source = os.path.join(PROJECT_ROOT, REF_RELPATH)
        spec = importlib.util.spec_from_file_location("hybrid_ref", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules["hybrid_ref"] = module
        spec.loader.exec_module(module)
        _REF_MODULE = module
    return _REF_MODULE


def _normalize(source_text):
    """
    Strip blank lines and whole-line comments before hashing, so reformatting or
    re-commenting does not halt a run. This deliberately trades comment sensitivity for
    practicality: comments cannot change behaviour in this class. Do NOT extend this to
    strip anything semantic, or the guard would start hiding real changes.
    """
    return "\n".join(
        line.rstrip() for line in source_text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def verify_steering_source(project_root=None):
    """
    Second layer of the steering guard. Layer one is that we use the real class, so no
    copy exists to diverge. This layer pins the class body, so an EDIT to the reference
    file stops the run instead of silently changing what is being measured.

    Reads the FILE, exactly like lexicon_tiers.verify_against_source(). It never inspects
    the module global that reference_lexicon() swaps, so an ablation in progress cannot
    influence this result.
    """
    if project_root is None:
        project_root = PROJECT_ROOT
    source = os.path.join(project_root, REF_RELPATH)
    raw = open(source, encoding="utf-8").read()
    block = re.search(r"^class SteeringProcessor\(LogitsProcessor\):.*?(?=^\S)", raw, re.S | re.M)
    if not block:
        raise RuntimeError(f"Could not locate class SteeringProcessor in {source}")
    digest = hashlib.sha256(_normalize(block.group(0)).encode()).hexdigest()
    if digest != REF_STEERING_SHA256:
        raise RuntimeError(
            "SteeringProcessor has changed in 07_evaluate_hybrid.py.\n"
            f"  expected {REF_STEERING_SHA256}\n"
            f"  found    {digest}\n"
            "The ablation would no longer run the steering that produced 47.50%. "
            "Reconcile the source, or re-pin REF_STEERING_SHA256 deliberately."
        )
    return True


def get_steering_processor_class():
    """The real class from the 47.50% source. Not a copy, so it cannot drift."""
    verify_steering_source()
    return _load_reference_module().SteeringProcessor


@contextlib.contextmanager
def reference_lexicon(tier_lexicon):
    """
    Temporarily point the reference module's EMOTION_LEXICON at an ablation tier.

    Why this is needed: SteeringProcessor reads EMOTION_LEXICON from its own module
    globals (07_evaluate_hybrid.py line 36), so a tier cannot be passed as an argument.
    Swapping the global keeps the ALGORITHM byte identical to the headline run and varies
    only its data input, which is exactly what the lexicon ablation is meant to vary.

    The real hazard is re-entrancy. If two swaps nest, the inner one captures the OUTER
    swap's tier as its "original" and restores to that, leaving the module permanently
    holding a tier lexicon with no error raised. We refuse to nest instead. A post
    assignment identity check would not catch this, and in a single threaded interpreter
    it can never fire at all, so it is not used here.
    """
    global _LEXICON_SWAP_DEPTH
    if _LEXICON_SWAP_DEPTH != 0:
        raise RuntimeError(
            "reference_lexicon() is already active. Nesting would restore the wrong "
            "lexicon and silently corrupt every later tier in this process. "
            "Build each steering processor in its own swap."
        )
    module = _load_reference_module()
    original = module.EMOTION_LEXICON
    _LEXICON_SWAP_DEPTH += 1
    module.EMOTION_LEXICON = tier_lexicon
    try:
        yield module
    finally:
        module.EMOTION_LEXICON = original
        _LEXICON_SWAP_DEPTH -= 1


def build_steering_processor(tokenizer, target_emotion, tier_lexicon, boost_factor=5.0):
    """
    Construct the real SteeringProcessor against a tier lexicon.

    The swap window spans only this constructor. SteeringProcessor resolves token_ids in
    __init__, so once the object exists it never consults the module global again and the
    original is already restored. During all 160 generations the steering code running is
    provably the unmodified 47.50% implementation, and only the constructor ever saw the
    tier. The ablation therefore varies exactly one thing: the token set baked into one
    object.
    """
    steering_class = get_steering_processor_class()
    with reference_lexicon(tier_lexicon):
        processor = steering_class(tokenizer, target_emotion, boost_factor=boost_factor)
    return processor


def self_test(verbose=True):
    """
    Assert that the lexicon swap and the drift guards are independent. Proven by hand
    once; this makes it a permanent invariant, so a future edit that breaks the
    independence is caught automatically instead of relying on someone re-testing.

    Checks the three non-trivial cases:
      1. guards pass before any swap,
      2. guards still pass DURING a swap (they read the file, not the global),
      3. the module global is restored to the SAME object afterwards.
    Case 3 compares against an identity captured OUTSIDE the context manager, so it
    genuinely tests the manager's behaviour rather than restating its own assignment.
    """
    lexicon_tiers.verify_against_source()
    judge.verify_against_source()
    verify_steering_source()

    module = _load_reference_module()
    pristine = module.EMOTION_LEXICON

    with reference_lexicon(lexicon_tiers.get_lexicon("5")) as swapped:
        if len(swapped.EMOTION_LEXICON["Joy"]) != 5:
            raise RuntimeError("Swap did not take effect inside reference_lexicon().")
        lexicon_tiers.verify_against_source()
        verify_steering_source()

    if module.EMOTION_LEXICON is not pristine:
        raise RuntimeError("reference_lexicon() did not restore the original EMOTION_LEXICON.")

    try:
        with reference_lexicon(lexicon_tiers.get_lexicon("5")):
            with reference_lexicon(lexicon_tiers.get_lexicon("10")):
                pass
        raise RuntimeError("Re-entrancy guard failed to refuse a nested swap.")
    except RuntimeError as error:
        if "already active" not in str(error):
            raise

    if module.EMOTION_LEXICON is not pristine:
        raise RuntimeError("Module global left corrupted after the re-entrancy test.")

    if verbose:
        print("Self test passed:")
        print("  lexicon, EMOTION_MAP and SteeringProcessor all match the 47.50% source")
        print("  guards pass during an active lexicon swap (they read the file, not the global)")
        print("  module global restored to the same object, nesting refused")
    return True


# ---------------------------------------------------------------------------
# Models and data
# ---------------------------------------------------------------------------
def load_hybrid_model(device=None, adapter_path=None):
    """
    GPT-2 Large plus a LoRA adapter.

    adapter_path defaults to ADAPTER_PATH, the adapter behind the 47.50% result, so every
    caller written before the rank sweep keeps its exact previous behaviour. The parameter
    exists only so the rank sweep can point at a freshly trained adapter per rank. Nothing
    else about the evaluation changes: same base model, same tokenizer, same eval mode.
    """
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from peft import PeftModel

    if adapter_path is None:
        adapter_path = ADAPTER_PATH
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    base = GPT2LMHeadModel.from_pretrained(BASE_MODEL_ID)
    model = PeftModel.from_pretrained(base, adapter_path).to(device)
    model.eval()
    return model, tokenizer, device


def load_prompt_set(path=None):
    """Read the frozen 160-row manifest. Rows are dicts with prompt_id and seed as ints."""
    if path is None:
        path = MANIFEST_PATH
    if not os.path.exists(path):
        raise SystemExit(
            f"Missing {os.path.relpath(path, PROJECT_ROOT)}\n"
            "Run:  python src/6_ablations/filter_neutral_stems.py\n"
            "then: python src/6_ablations/make_test_prompts.py"
        )
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["prompt_id"] = int(row["prompt_id"])
            row["seed"] = int(row["seed"])
            row["stem_id"] = int(row["stem_id"])
            rows.append(row)
    return rows


def generate_one(model, tokenizer, device, row, decoding, logits_processor=None):
    """
    Generate one story.

    The seed is applied per row, not per run, so row 84 produces identical text whether
    it is the 84th generation of a clean run or the first after a crash resume. Order
    cannot affect the result.

    Story extraction matches the reference exactly, so the stem stays inside the judged
    text just as the word "The" did in the 47.50% run.
    """
    import torch
    from transformers import LogitsProcessorList

    torch.manual_seed(row["seed"])
    inputs = tokenizer(row["prompt_text"], return_tensors="pt").to(device)
    processors = LogitsProcessorList([logits_processor]) if logits_processor else None

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            repetition_penalty=REPETITION_PENALTY,
            logits_processor=processors,
            pad_token_id=tokenizer.eos_token_id,
            **DECODING_PRESETS[decoding],
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full.split("Story:")[1].strip() if "Story:" in full else full


# ---------------------------------------------------------------------------
# Crash safe output
# ---------------------------------------------------------------------------
class ResumableWriter:
    """
    Append one row at a time, flushed and fsynced, so a crash never loses completed work.

    Resume is keyed on the manifest's frozen prompt_id, never on row index. The manifest
    is frozen, so this is belt and braces, but index based resume would silently pair the
    wrong story with the wrong target if the manifest were ever reordered, and that is a
    bug you would never notice in the output.
    """
    PARTIAL_FIELDS = ["prompt_id", "target_label", "detected", "story"]

    def __init__(self, partial_path, final_path):
        self.partial_path = partial_path
        self.final_path = final_path
        os.makedirs(os.path.dirname(partial_path), exist_ok=True)
        if not os.path.exists(partial_path):
            with open(partial_path, "w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.PARTIAL_FIELDS).writeheader()

    def completed_prompt_ids(self):
        done = set()
        with open(self.partial_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("prompt_id", "").strip():
                    done.add(int(row["prompt_id"]))
        return done

    def append(self, prompt_id, target_label, detected, story):
        with open(self.partial_path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.PARTIAL_FIELDS).writerow({
                "prompt_id": prompt_id,
                "target_label": target_label,
                "detected": detected,
                "story": story,
            })
            handle.flush()
            os.fsync(handle.fileno())

    def rows(self):
        with open(self.partial_path, newline="", encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle) if r.get("prompt_id", "").strip()]
        rows.sort(key=lambda r: int(r["prompt_id"]))
        return rows

    def finalize(self, extra_paths=()):
        """
        Write the final CSV in the EXACT project schema: Target, Detected, Story.
        Ordered by prompt_id so the output is stable regardless of completion order.
        """
        rows = self.rows()
        for path in [self.final_path, *extra_paths]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Target", "Detected", "Story"])
                writer.writeheader()
                for row in rows:
                    writer.writerow({
                        "Target": row["target_label"],
                        "Detected": row["detected"],
                        "Story": row["story"],
                    })
        return len(rows), self.final_path


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def git_state():
    """
    Commit hash plus whether the tree was clean. A commit hash on a dirty tree is a lie:
    it says "this code" while uncommitted changes actually ran. The dirty file list is
    recorded too, because once the flag is false the next question is always "dirty how".
    """
    def run(args):
        try:
            result = subprocess.run(["git", *args], cwd=PROJECT_ROOT,
                                    capture_output=True, text=True, timeout=30)
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    porcelain = run(["status", "--porcelain"])
    dirty = [line[3:] for line in porcelain.splitlines() if line.strip()]
    return {
        "git_commit": run(["rev-parse", "HEAD"]) or "unknown",
        "git_branch": run(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown",
        "git_tree_clean": not dirty,
        "git_dirty_files": dirty,
    }


def collect_environment():
    """Everything needed to reproduce the numeric result on another machine."""
    import torch
    import transformers
    import peft

    # str() on every version: torch.__version__ is a TorchVersion object, not a plain
    # string, and yaml.safe_dump refuses to serialise it.
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "peft": str(peft.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["cuda"] = str(torch.version.cuda)
        info["cudnn"] = int(torch.backends.cudnn.version())
        info["gpu"] = str(torch.cuda.get_device_name(0))
        info["vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
    return info


def next_experiment_dir(config_name):
    """experiments/experiment_XXX_<config_name>/, numbered in creation order."""
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    existing = [d for d in os.listdir(EXPERIMENTS_DIR) if d.startswith("experiment_")]
    numbers = []
    for name in existing:
        parts = name.split("_")
        if len(parts) > 1 and parts[1].isdigit():
            numbers.append(int(parts[1]))
    index = max(numbers) + 1 if numbers else 1
    path = os.path.join(EXPERIMENTS_DIR, f"experiment_{index:03d}_{config_name}")
    os.makedirs(path, exist_ok=True)
    return path


def find_or_create_experiment_dir(config_name, expected_rows):
    """
    Reuse an INCOMPLETE directory for this config, otherwise start a new one.

    Without this, execute_run() always created a fresh directory, so ResumableWriter's
    resume path was unreachable in normal operation: a re-run after a crash started from
    row 0 in a new folder rather than continuing. The writer itself worked, but the
    workflow claim "resumable from a crash" was false. This makes it true.

    A directory counts as incomplete when its partial file has between 1 and
    expected_rows - 1 rows. A finished config is never reused, so deliberately re-running
    a completed experiment still produces a clean new directory rather than silently
    doing nothing.
    """
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    suffix = f"_{config_name}"
    for name in sorted(os.listdir(EXPERIMENTS_DIR)):
        if not (name.startswith("experiment_") and name.endswith(suffix)):
            continue
        path = os.path.join(EXPERIMENTS_DIR, name)
        partial = os.path.join(path, "partial_results.csv")
        if not os.path.exists(partial):
            continue
        with open(partial, newline="", encoding="utf-8") as handle:
            done = sum(1 for row in csv.DictReader(handle) if row.get("prompt_id", "").strip())
        if 0 < done < expected_rows:
            print(f"  reusing {name}, {done} of {expected_rows} rows already complete")
            return path
    return next_experiment_dir(config_name)


def write_provenance(exp_dir, config):
    """Write config.yaml. Warns loudly when the tree is dirty, so it is seen during the run."""
    import yaml

    payload = dict(config)
    payload["git"] = git_state()
    payload["environment"] = collect_environment()
    payload["run_command"] = " ".join([os.path.basename(sys.executable), *sys.argv])
    payload["reference"] = {
        "source_file": REF_RELPATH.replace("\\", "/"),
        "steering_sha256": REF_STEERING_SHA256,
        "steering_loaded_live": True,
    }
    path = os.path.join(exp_dir, "config.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)

    if not payload["git"]["git_tree_clean"]:
        print("\n  WARNING: git tree was NOT clean at run time.")
        print(f"  The recorded commit {payload['git']['git_commit'][:7]} does not fully describe this run.")
        for name in payload["git"]["git_dirty_files"][:10]:
            print(f"    dirty: {name}")
        print()
    return path


# ---------------------------------------------------------------------------
# Scoring and summary
# ---------------------------------------------------------------------------
def summarize(targets, detected):
    """N, top-1 accuracy and per-emotion F1, using the same metrics as the figure code."""
    from sklearn.metrics import accuracy_score, f1_score

    metrics = {
        "N": len(targets),
        "top1_accuracy": round(accuracy_score(targets, detected) * 100, 2),
    }
    for emotion in EMOTIONS:
        metrics[f"f1_{emotion}"] = round(
            f1_score(targets, detected, labels=[emotion], average="micro", zero_division=0), 4
        )
    return metrics


def print_summary(config_name, metrics):
    """Final summary so numbers can be sanity checked before they reach the paper."""
    print(f"\n--- {config_name} ---")
    print(f"  N              {metrics['N']}")
    print(f"  Top-1 accuracy {metrics['top1_accuracy']:.2f}%")
    print("  Per-emotion F1:")
    for emotion in EMOTIONS:
        print(f"    {emotion:<14} {metrics[f'f1_{emotion}']:.4f}")
    if "mean_seconds_per_story" in metrics:
        print(f"  Latency        {metrics['mean_seconds_per_story']:.2f} s per story "
              f"(generation only, judge excluded, over {metrics['rows_timed']} rows)")
    if "peak_vram_gib" in metrics:
        print(f"  Peak VRAM      {metrics['peak_vram_gib']:.2f} GiB")


def append_to_summary(config_name, model_name, technique, metrics, caveat=""):
    """Append one row to results/all_configs_summary.csv, replacing any earlier row."""
    fields = ["config_name", "model", "technique", "N", "top1_accuracy"] + \
             [f"f1_{e}" for e in EMOTIONS] + ["caveat"]
    row = {
        "config_name": config_name,
        "model": model_name,
        "technique": technique,
        "N": metrics["N"],
        "top1_accuracy": metrics["top1_accuracy"],
        "caveat": caveat,
    }
    for emotion in EMOTIONS:
        row[f"f1_{emotion}"] = metrics[f"f1_{emotion}"]

    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    rows = []
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, newline="", encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle) if r.get("config_name") != config_name]
    rows.append(row)
    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for existing in rows:
            writer.writerow({key: existing.get(key, "") for key in fields})
    return SUMMARY_PATH


# ---------------------------------------------------------------------------
# The shared run loop, used by all three runners
# ---------------------------------------------------------------------------
def execute_run(config_name, output_csv, technique, tier, boost_factor, decoding,
                model_label="GPT-2 Large + LoRA r32", caveat="",
                adapter_path=None, extra_provenance=None):
    """
    One complete experiment: 160 generations, scored, saved and summarised.

    Reproducible from a single command, resumable after a crash, and every run leaves a
    config.yaml recording exactly what produced it.

    adapter_path and extra_provenance were added for the rank sweep and are both optional.
    Omitting them reproduces the pre-sweep behaviour exactly, which is what keeps the eight
    completed Task A runs valid without re-running them. extra_provenance is merged into
    config.yaml, so a caller can record the facts only it knows (rank, alpha, training
    time) without this function needing to know about them.
    """
    import json

    print(f"=== {config_name} ===\n")
    self_test()

    if adapter_path is None:
        adapter_path = ADAPTER_PATH

    rows = load_prompt_set()
    tier_lexicon = lexicon_tiers.get_lexicon(tier)
    exp_dir = find_or_create_experiment_dir(config_name, len(rows))

    provenance = {
        "config_name": config_name,
        "technique": technique,
        "model": model_label,
        "base_model": BASE_MODEL_ID,
        "adapter": os.path.relpath(adapter_path, PROJECT_ROOT).replace("\\", "/"),
        "lexicon_tier": str(tier),
        "lexicon_word_counts": lexicon_tiers.tier_word_counts(tier),
        "lexicon_words": tier_lexicon,
        "boost_factor": boost_factor,
        "decoding": decoding,
        "decoding_params": DECODING_PRESETS[decoding],
        "max_new_tokens": MAX_NEW_TOKENS,
        "repetition_penalty": REPETITION_PENALTY,
        "manifest": os.path.relpath(MANIFEST_PATH, PROJECT_ROOT).replace("\\", "/"),
        "manifest_rows": len(rows),
        "judge": judge.TEACHER_NAME,
        "caveat": caveat,
    }
    if extra_provenance:
        provenance.update(extra_provenance)
    write_provenance(exp_dir, provenance)

    writer = ResumableWriter(
        os.path.join(exp_dir, "partial_results.csv"),
        os.path.join(exp_dir, "results.csv"),
    )
    done = writer.completed_prompt_ids()
    if done:
        print(f"  resuming, {len(done)} of {len(rows)} rows already complete\n")

    model, tokenizer, device = load_hybrid_model(adapter_path=adapter_path)
    judge_model, judge_tokenizer, _ = judge.load_judge(device)
    print(f"  model and judge loaded on {device}\n")

    # One steering processor per emotion, each built in its own swap window.
    processors = {
        key: build_steering_processor(tokenizer, key, tier_lexicon, boost_factor)
        for key in tier_lexicon
    }

    # Latency instrumentation. The timer wraps generate_one ONLY, deliberately excluding
    # judge.classify, because the number this produces gets compared against baselines
    # that time generation alone (PPLM records mean_seconds_per_story the same way).
    # Folding judging time into the denominator would silently understate any ratio built
    # from it. Resumed rows are skipped by the counter too, so a run finished across two
    # sessions reports the mean over rows THIS process actually generated rather than a
    # figure diluted by work it did not do.
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    generation_seconds = 0.0
    generated_here = 0

    for index, row in enumerate(rows):
        if row["prompt_id"] in done:
            continue
        started = time.time()
        story = generate_one(model, tokenizer, device, row, decoding,
                             processors[row["lexicon_key"]])
        generation_seconds += time.time() - started
        generated_here += 1
        detected = judge.classify(story, judge_model, judge_tokenizer, device)
        writer.append(row["prompt_id"], row["target_label"], detected, story)
        if (index + 1) % 20 == 0:
            print(f"  {index + 1}/{len(rows)} generated")

    root_copy = os.path.join(PROJECT_ROOT, output_csv)
    count, _ = writer.finalize(extra_paths=[root_copy])

    final = writer.rows()
    metrics = summarize([r["target_label"] for r in final], [r["detected"] for r in final])

    # Recorded for every run from now on, so the repo never again holds a latency claim
    # with no measured denominator. mean_seconds_per_story counts only rows generated in
    # this process; rows_timed says how many that was, so a resumed run cannot be mistaken
    # for a full-length timing measurement.
    if generated_here:
        metrics["mean_seconds_per_story"] = round(generation_seconds / generated_here, 2)
        metrics["rows_timed"] = generated_here
    if torch.cuda.is_available():
        metrics["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)

    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print_summary(config_name, metrics)
    append_to_summary(config_name, model_label, technique, metrics, caveat)
    print(f"\n  wrote {count} rows to {output_csv}")
    print(f"  provenance: {os.path.relpath(exp_dir, PROJECT_ROOT)}")
    return metrics


if __name__ == "__main__":
    self_test()
