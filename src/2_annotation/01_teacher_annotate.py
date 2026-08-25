import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "writing_prompts_subset.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "annotated", "annotated_train.csv")

# Teacher Model (GoEmotions)
TEACHER_NAME = "SamLowe/roberta-base-go_emotions"
BATCH_SIZE = 16  # Safe for 6GB VRAM (RoBERTa Base)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- The "Nuance" Mapping: 27 GoEmotions -> 8 Plutchik ---
# This is the core intellectual contribution of the data processing phase.
EMOTION_MAP = {
    # Plutchik: Joy
    "joy": "Joy", "amusement": "Joy", "excitement": "Joy", "love": "Joy", "optimism": "Joy", "pride": "Joy",
    # Plutchik: Trust
    "admiration": "Trust", "approval": "Trust", "caring": "Trust", "gratitude": "Trust", "relief": "Trust",
    # Plutchik: Fear
    "fear": "Fear", "nervousness": "Fear",
    # Plutchik: Surprise
    "surprise": "Surprise", "confusion": "Surprise", "curiosity": "Surprise",
    # Plutchik: Sadness
    "sadness": "Sadness", "disappointment": "Sadness", "embarrassment": "Sadness", "grief": "Sadness", "remorse": "Sadness",
    # Plutchik: Disgust
    "disgust": "Disgust",
    # Plutchik: Anger
    "anger": "Anger", "annoyance": "Anger", "disapproval": "Anger",
    # Plutchik: Anticipation
    "realization": "Anticipation", "desire": "Anticipation",
    # Neutral
    "neutral": "Neutral"
}

def load_data():
    df = pd.read_csv(INPUT_FILE)
    # Ensure text is string and not empty
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str)
    return df

def get_device():
    print(f"--- Running on: {DEVICE} ---")
    if DEVICE == 'cuda':
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return torch.device(DEVICE)

def main():
    device = get_device()
    df = load_data()
    print(f"Loaded {len(df)} stories for annotation.")

    # 1. Load Teacher
    print("Loading Teacher Model (RoBERTa)...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(TEACHER_NAME).to(device)
    model.eval() # Set to inference mode

    # 2. Inference Loop
    predicted_emotions = []
    
    # We process in batches to speed up
    texts = df['text'].tolist()
    
    print(f"Starting Annotation of {len(texts)} stories...")
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i : i + BATCH_SIZE]
        
        # Tokenize (Truncate to 512 for RoBERTa)
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get raw probabilities
        probs = torch.sigmoid(outputs.logits)
        
        # Get top predicted label index
        top_indices = torch.argmax(probs, dim=1).cpu().numpy()
        
        # Map ID to Label -> Map to Plutchik
        for idx in top_indices:
            raw_label = model.config.id2label[idx]
            plutchik_label = EMOTION_MAP.get(raw_label, "Neutral") # Default to Neutral if mapping fails
            predicted_emotions.append(plutchik_label)

    # 3. Save Results
    df['emotion_label'] = predicted_emotions
    
    # Filter out Neutral if we want purely emotional stories (Optional, but good for strong training)
    # df = df[df['emotion_label'] != "Neutral"] 
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Annotation Complete. Saved to {OUTPUT_FILE}")
    print("\n--- Emotion Distribution ---")
    print(df['emotion_label'].value_counts())

if __name__ == "__main__":
    main()