import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models", "final_adapter")
BASE_MODEL_ID = "gpt2-medium"
TEACHER_NAME = "SamLowe/roberta-base-go_emotions"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMOTIONS = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Anger", "Anticipation"] # Removed Neutral for strict testing
SAMPLES_PER_EMOTION = 10  # Low for quick testing, increase to 50 for final paper
PROMPT = "The door opened slowly,"

# --- Mapping (Same as Annotation) ---
EMOTION_MAP = {
    "joy": "Joy", "amusement": "Joy", "excitement": "Joy", "love": "Joy", "optimism": "Joy", "pride": "Joy",
    "admiration": "Trust", "approval": "Trust", "caring": "Trust", "gratitude": "Trust", "relief": "Trust",
    "fear": "Fear", "nervousness": "Fear",
    "surprise": "Surprise", "confusion": "Surprise", "curiosity": "Surprise",
    "sadness": "Sadness", "disappointment": "Sadness", "embarrassment": "Sadness", "grief": "Sadness", "remorse": "Sadness",
    "disgust": "Anger", "anger": "Anger", "annoyance": "Anger", "disapproval": "Anger",
    "realization": "Anticipation", "desire": "Anticipation",
    "neutral": "Neutral"
}

def load_student():
    print("Loading Student Model...")
    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_ID)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

def load_teacher():
    print("Loading Teacher Model...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(TEACHER_NAME).to(DEVICE)
    model.eval()
    return model, tokenizer

def classify_text(text, teacher_model, teacher_tokenizer):
    inputs = teacher_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = teacher_model(**inputs)
    
    probs = torch.sigmoid(outputs.logits)
    top_idx = torch.argmax(probs, dim=1).item()
    raw_label = teacher_model.config.id2label[top_idx]
    return EMOTION_MAP.get(raw_label, "Neutral")

def main():
    # 1. Load Models
    student_model, student_tokenizer = load_student()
    teacher_model, teacher_tokenizer = load_teacher()

    true_labels = []
    pred_labels = []
    generated_texts = []

    print(f"\n--- Starting Evaluation ({SAMPLES_PER_EMOTION} samples per emotion) ---")

    # 2. Generation Loop
    for target_emotion in EMOTIONS:
        print(f"Testing: {target_emotion}...")
        for _ in range(SAMPLES_PER_EMOTION):
            # Prompt Student
            input_text = f"<|startoftext|> [{target_emotion}] {PROMPT}"
            inputs = student_tokenizer(input_text, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = student_model.generate(
                    **inputs, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.9
                )
            
            gen_text = student_tokenizer.decode(outputs[0], skip_special_tokens=True)
            clean_text = gen_text.replace(f"[{target_emotion}]", "").replace("<|startoftext|>", "").strip()
            
            # Grade with Teacher
            detected_emotion = classify_text(clean_text, teacher_model, teacher_tokenizer)
            
            true_labels.append(target_emotion)
            pred_labels.append(detected_emotion)
            generated_texts.append(clean_text)

    # 3. Calculate Metrics
    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n--- RESULTS ---")
    print(f"Overall Arc Adherence Accuracy: {acc * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, zero_division=0))
    
    # Save detailed logs for analysis
    df_results = pd.DataFrame({
        "Target": true_labels,
        "Detected": pred_labels,
        "Text": generated_texts
    })
    df_results.to_csv("evaluation_results_baseline.csv", index=False)
    print("Saved detailed results to 'evaluation_results_baseline.csv'")

if __name__ == "__main__":
    main()