import os
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# POINTING TO THE NEW OPTIMIZED FOLDER
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models", "gpt2_large_optimized")
BASE_MODEL_ID = "gpt2-large" 
TEACHER_NAME = "SamLowe/roberta-base-go_emotions"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMOTIONS = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation"]
SAMPLES_PER_EMOTION = 20

# --- Mapping ---
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
    print(f"Loading Optimized GPT-2 Large from {ADAPTER_PATH}...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(ADAPTER_PATH)
        tokenizer.pad_token = tokenizer.eos_token
    except:
        print("Could not load tokenizer from adapter, using base...")
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
    student_model, student_tokenizer = load_large_model()
    teacher_model, teacher_tokenizer = load_teacher()

    true_labels = []
    pred_labels = []
    generated_texts = []

    print(f"\n--- Starting GPT-2 LARGE OPTIMIZED Evaluation ---")

    for target_emotion in EMOTIONS:
        expected_label = "Anger" if target_emotion == "Disgust" else target_emotion
        print(f"Testing: {target_emotion}...")
        
        for _ in range(SAMPLES_PER_EMOTION):
            # Prompt-Free Generation (The model must invent the story)
            input_text = f"Emotion: {target_emotion} | Story:"
            
            inputs = student_tokenizer(input_text, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = student_model.generate(
                    **inputs, 
                    max_new_tokens=80, 
                    do_sample=True, 
                    temperature=0.85, # Creativity allowed
                    top_p=0.92, 
                    repetition_penalty=1.2,
                    pad_token_id=student_tokenizer.eos_token_id
                )
            
            full_text = student_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract story part
            try:
                generated_story = full_text.split("Story:")[1].strip()
            except IndexError:
                generated_story = full_text
            
            detected_emotion = classify_text(generated_story, teacher_model, teacher_tokenizer)
            
            true_labels.append(expected_label)
            pred_labels.append(detected_emotion)
            generated_texts.append(generated_story)

    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n--- GPT-2 LARGE OPTIMIZED RESULTS ---")
    print(f"Overall Arc Adherence Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, zero_division=0))
    
    pd.DataFrame({
        "Target": true_labels, 
        "Detected": pred_labels, 
        "Generated_Text": generated_texts
    }).to_csv("evaluation_results_large_optimized.csv", index=False)

if __name__ == "__main__":
    main()