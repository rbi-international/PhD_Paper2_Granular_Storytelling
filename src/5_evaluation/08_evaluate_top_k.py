import os
import torch
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models", "gpt2_large_optimized")
BASE_MODEL_ID = "gpt2-large"
TEACHER_NAME = "SamLowe/roberta-base-go_emotions"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLES_PER_EMOTION = 20
TOP_K = 3  # CHECK IF TARGET IS IN TOP 3

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

def main():
    print(f"--- Loading Models for Top-{TOP_K} Evaluation ---")
    
    # Load Student
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(ADAPTER_PATH)
    except:
        tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_ID)
    student = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(DEVICE)
    student.eval()

    # Load Teacher
    t_tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_NAME).to(DEVICE)
    teacher.eval()

    correct_count = 0
    total_count = 0
    
    print("\n--- Starting Generation ---")
    
    # Target Emotions
    targets = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation"]
    
    for target in targets:
        print(f"Testing {target}...")
        
        for _ in range(SAMPLES_PER_EMOTION):
            # Generate
            input_text = f"Emotion: {target} | Story:"
            inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = student.generate(
                    **inputs, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.92, repetition_penalty=1.2
                )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            try: story = text.split("Story:")[1].strip()
            except: story = text

            # Evaluate (Top K)
            inputs = t_tokenizer(story, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            with torch.no_grad():
                outputs = teacher(**inputs)
            
            # Get Top K indices
            probs = torch.sigmoid(outputs.logits)
            top_k_indices = torch.topk(probs, k=5, dim=1).indices[0].cpu().numpy() # Check top 5 raw
            
            # Map them to Plutchik
            detected_emotions = []
            for idx in top_k_indices:
                raw_lbl = teacher.config.id2label[idx]
                mapped = EMOTION_MAP.get(raw_lbl, "Neutral")
                detected_emotions.append(mapped)
            
            # Check overlap
            # Disgust -> Anger merge logic
            expected = "Anger" if target == "Disgust" else target
            
            # If the expected emotion is ANYWHERE in the top K detected, it's a pass
            if expected in detected_emotions[:TOP_K]:
                correct_count += 1
            
            total_count += 1

    acc = correct_count / total_count
    print(f"\n--- FINAL VERDICT ---")
    print(f"Top-{TOP_K} Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()