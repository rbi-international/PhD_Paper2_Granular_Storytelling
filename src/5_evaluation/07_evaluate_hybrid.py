import os
import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, LogitsProcessor, LogitsProcessorList, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# POINTING TO THE LARGE OPTIMIZED MODEL
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models", "gpt2_large_optimized")
BASE_MODEL_ID = "gpt2-large"
TEACHER_NAME = "SamLowe/roberta-base-go_emotions"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLES_PER_EMOTION = 20
PROMPT_STORY_START = "The" # We give a very neutral start token

# --- THE LEXICON (Symbolic Knowledge) ---
EMOTION_LEXICON = {
    "Joy": ["smile", "laugh", "happy", "delight", "wonderful", "bright", "cheer", "excited", "love", "joy", "grin", "warmth", "beautiful", "sun", "light"],
    "Trust": ["friend", "trust", "believe", "safe", "rely", "honor", "truth", "accept", "help", "support", "agree", "faith", "promise", "steady"],
    "Fear": ["dark", "scared", "afraid", "shiver", "scream", "run", "terror", "shadow", "cold", "panic", "threat", "nervous", "hide", "creep", "blood"],
    "Surprise": ["shock", "gasp", "sudden", "unexpected", "stunned", "wow", "flash", "blink", "amazed", "jolt", "startle", "wide", "abrupt"],
    "Sadness": ["cry", "tear", "loss", "grief", "sob", "sad", "alone", "miss", "pain", "hurt", "empty", "sorry", "dark", "weep", "broken"],
    "Disgust": ["gross", "nasty", "sick", "vile", "foul", "ugly", "hate", "repulse", "dirty", "rotten", "trash", "slime", "smell", "puke"],
    "Anger": ["rage", "furious", "hate", "yell", "shout", "mad", "angry", "fist", "hit", "burn", "fight", "stupid", "annoy", "kill", "slam"],
    "Anticipation": ["wait", "hope", "ready", "soon", "plan", "wish", "dream", "expect", "watch", "look", "forward", "prepare", "future"]
}

# --- Steering Logic ---
class SteeringProcessor(LogitsProcessor):
    def __init__(self, tokenizer, target_emotion, boost_factor=5.0): # Soft Boost (5.0) to avoid breaking grammar
        self.tokenizer = tokenizer
        self.boost_factor = boost_factor
        self.target_words = EMOTION_LEXICON.get(target_emotion, [])
        self.token_ids = []
        for word in self.target_words:
            ids = [
                tokenizer.encode(word, add_special_tokens=False),
                tokenizer.encode(" " + word, add_special_tokens=False),
                tokenizer.encode(word.capitalize(), add_special_tokens=False)
            ]
            for i in ids: self.token_ids.extend(i)
        self.token_ids = list(set(self.token_ids))

    def __call__(self, input_ids, scores):
        if self.token_ids:
            for tid in self.token_ids:
                if tid < scores.shape[1]:
                    scores[:, tid] += self.boost_factor
        return scores

# --- Evaluation Setup ---
EMOTION_MAP = {
    "joy": "Joy", "amusement": "Joy", "excitement": "Joy", "love": "Joy", "optimism": "Joy", "pride": "Joy",
    "admiration": "Trust", "approval": "Trust", "caring": "Trust", "gratitude": "Trust", "relief": "Trust",
    "fear": "Fear", "nervousness": "Fear",
    "surprise": "Surprise", "confusion": "Surprise", "curiosity": "Surprise",
    "sadness": "Sadness", "disappointment": "Sadness", "embarrassment": "Sadness", "grief": "Sadness", "remorse": "Sadness",
    "disgust": "Disgust", "anger": "Anger", "annoyance": "Anger", "disapproval": "Anger",
    "realization": "Anticipation", "desire": "Anticipation",
    "neutral": "Neutral"
}

def load_large_model():
    print(f"Loading GPT-2 Large from {ADAPTER_PATH}...")
    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_ID)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

def load_teacher():
    print("Loading Teacher...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(TEACHER_NAME).to(DEVICE)
    model.eval()
    return model, tokenizer

def classify(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)
    top_idx = torch.argmax(probs, dim=1).item()
    return EMOTION_MAP.get(model.config.id2label[top_idx], "Neutral")

def main():
    student, student_tok = load_large_model()
    teacher, teacher_tok = load_teacher()
    true_labels, pred_labels, texts = [], [], []

    print("\n--- HYBRID EVALUATION (Large + Steering) ---")
    
    for emotion in EMOTION_LEXICON.keys():
        print(f"Testing: {emotion}...")
        expected = "Anger" if emotion == "Disgust" else emotion
        
        # Apply Steering
        steering = SteeringProcessor(student_tok, emotion, boost_factor=5.0)
        logits_processor = LogitsProcessorList([steering])
        
        for _ in range(SAMPLES_PER_EMOTION):
            # Format: "Emotion: Joy | Story: The"
            input_text = f"Emotion: {emotion} | Story: {PROMPT_STORY_START}"
            inputs = student_tok(input_text, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = student.generate(
                    **inputs,
                    max_new_tokens=60,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.92,
                    repetition_penalty=1.2,
                    logits_processor=logits_processor, # HYBRID MAGIC
                    pad_token_id=student_tok.eos_token_id
                )
            
            full = student_tok.decode(outputs[0], skip_special_tokens=True)
            try:
                story = full.split("Story:")[1].strip()
            except:
                story = full
            
            detected = classify(story, teacher, teacher_tok)
            true_labels.append(expected)
            pred_labels.append(detected)
            texts.append(story)

    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n--- HYBRID RESULTS ---")
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print(classification_report(true_labels, pred_labels, zero_division=0))
    
    pd.DataFrame({"Target": true_labels, "Detected": pred_labels, "Story": texts}).to_csv("hybrid_results.csv", index=False)

if __name__ == "__main__":
    main()