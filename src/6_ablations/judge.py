"""
judge.py

Layman note: this is the automatic marker. It reads a piece of text and says which
emotion it carries. We reuse the exact same marker everywhere, so a score from one
experiment means the same thing as a score from another.

Technical note: RoBERTa fine-tuned on GoEmotions (28 labels), mapped down to the
Plutchik-8 scheme used throughout the paper. EMOTION_MAP and the classify() logic
are copied verbatim from src/5_evaluation/07_evaluate_hybrid.py, the run that
produced 47.50% top-1. They are copied rather than imported because directory names
starting with a digit (5_evaluation) are not importable as Python modules.
verify_against_source() guards the copy against drift and every caller runs it.

No em dashes anywhere (project style rule).
"""
import os
import re

TEACHER_NAME = "SamLowe/roberta-base-go_emotions"

# --- GoEmotions (28 labels) collapsed to Plutchik-8, verbatim from 07_evaluate_hybrid.py ---
EMOTION_MAP = {
    "joy": "Joy", "amusement": "Joy", "excitement": "Joy", "love": "Joy", "optimism": "Joy", "pride": "Joy",
    "admiration": "Trust", "approval": "Trust", "caring": "Trust", "gratitude": "Trust", "relief": "Trust",
    "fear": "Fear", "nervousness": "Fear",
    "surprise": "Surprise", "confusion": "Surprise", "curiosity": "Surprise",
    "sadness": "Sadness", "disappointment": "Sadness", "embarrassment": "Sadness", "grief": "Sadness", "remorse": "Sadness",
    "disgust": "Disgust", "anger": "Anger", "annoyance": "Anger", "disapproval": "Anger",
    "realization": "Anticipation", "desire": "Anticipation",
    "neutral": "Neutral",
}

# Labels that count as carrying a target emotion. Anything mapping outside this set
# (including unmapped GoEmotions labels) is treated as Neutral by classify().
PLUTCHIK_LABELS = {"Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation"}


def verify_against_source(project_root=None):
    """
    Guard against drift: re-parse EMOTION_MAP out of 07_evaluate_hybrid.py and confirm
    our copy still matches. Raises if they diverge, so a silent edit to either file can
    never quietly change what a score means.
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source = os.path.join(project_root, "src", "5_evaluation", "07_evaluate_hybrid.py")
    text = open(source, encoding="utf-8").read()
    body = re.search(r"EMOTION_MAP = \{(.*?)\n\}", text, re.S)
    if not body:
        raise RuntimeError(f"Could not locate EMOTION_MAP in {source}")
    namespace = {}
    exec("SOURCE_MAP = {" + body.group(1) + "\n}", namespace)
    if namespace["SOURCE_MAP"] != EMOTION_MAP:
        raise RuntimeError(
            "EMOTION_MAP has drifted from 07_evaluate_hybrid.py. Scores would no longer "
            "mean what they meant in the 47.50% run. Reconcile the two before running."
        )
    return True


def load_judge(device=None):
    """Load the RoBERTa teacher. Returns (model, tokenizer, device)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    verify_against_source()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(TEACHER_NAME).to(device)
    model.eval()
    return model, tokenizer, device


def classify(text, model, tokenizer, device):
    """
    Top-1 Plutchik label for a piece of text. Identical logic to 07_evaluate_hybrid.py:
    sigmoid over the 28 GoEmotions logits, take the argmax, map it down to Plutchik-8,
    and fall back to Neutral for anything unmapped.
    """
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)
    top_idx = torch.argmax(probs, dim=1).item()
    return EMOTION_MAP.get(model.config.id2label[top_idx], "Neutral")


def emotion_profile(text, model, tokenizer, device):
    """
    Richer view used by the stem neutrality filter. Returns the top-1 Plutchik label
    plus the highest probability assigned to ANY emotion-carrying label. The second
    number catches the case where Neutral wins narrowly (Neutral 0.31, Fear 0.29),
    which a top-1 check alone would wave through.
    """
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)[0]
    top_idx = torch.argmax(probs).item()
    top_label = EMOTION_MAP.get(model.config.id2label[top_idx], "Neutral")

    max_emotion_prob = 0.0
    max_emotion_label = "None"
    for idx in range(probs.shape[0]):
        mapped = EMOTION_MAP.get(model.config.id2label[idx], "Neutral")
        if mapped in PLUTCHIK_LABELS:
            value = float(probs[idx])
            if value > max_emotion_prob:
                max_emotion_prob = value
                max_emotion_label = mapped
    return {
        "top_label": top_label,
        "neutral_prob": float(probs[top_idx]) if top_label == "Neutral" else 0.0,
        "max_emotion_label": max_emotion_label,
        "max_emotion_prob": max_emotion_prob,
    }


if __name__ == "__main__":
    verify_against_source()
    print("EMOTION_MAP verified identical to 07_evaluate_hybrid.py (the 47.50% run).")
    print(f"Teacher: {TEACHER_NAME}")
    print(f"GoEmotions labels mapped: {len(EMOTION_MAP)}")
