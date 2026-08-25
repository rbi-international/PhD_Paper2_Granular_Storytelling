import os
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from sklearn.metrics import accuracy_score, classification_report

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models", "final_special_adapter")
BASE_MODEL_ID = "gpt2-medium"
TEACHER_NAME = "SamLowe/roberta-base-go_emotions"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMOTIONS = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation"]
SAMPLES_PER_EMOTION = 20 # 20 * 8 = 160 stories total
PROMPT = "The door opened slowly,"

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

# Fix Disgust mapping for evaluation (since we mapped Disgust->Anger in training, the model might output Anger for Disgust prompts)
# We will treat "Anger" as a correct response for "Disgust" prompt for fairness, or keep it strict. 
# Let's keep it strict but acknowledge Disgust is rare.

def load_special_student():
    print(f"Loading Special Student Model from {ADAPTER_PATH}...")
    
    # 1. Load the ADAPTER's tokenizer (which has [Joy], [Fear] etc.)
    tokenizer = GPT2Tokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Load Base Model
    base_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_ID)
    base_model.resize_token_embeddings(len(tokenizer)) # Resize to fit new tokens
    
    # 3. Load LoRA
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
    student_model, student_tokenizer = load_special_student()
    teacher_model, teacher_tokenizer = load_teacher()

    true_labels = []
    pred_labels = []
    
    print(f"\n--- Starting Special Token Evaluation ({SAMPLES_PER_EMOTION} samples/emotion) ---")

    for target_emotion in EMOTIONS:
        # Note: Disgust was merged into Anger during training, so we expect [Disgust] to generate Anger
        expected_label = "Anger" if target_emotion == "Disgust" else target_emotion
        
        print(f"Testing: [{target_emotion}] -> Expecting: {expected_label}")
        
        for _ in range(SAMPLES_PER_EMOTION):
            # Input is just the special token + prompt
            # Since [Joy] is now a single token, we don't need spaces inside brackets if tokenizer handles it,
            # but for safety in string construction we use the format we trained on.
            input_text = f"<|startoftext|> [{target_emotion}] {PROMPT}"
            
            inputs = student_tokenizer(input_text, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = student_model.generate(
                    **inputs, 
                    max_new_tokens=80, 
                    do_sample=True, 
                    temperature=0.85, # Slightly higher temp for creativity
                    top_p=0.9, 
                    repetition_penalty=1.2
                )
            
            gen_text = student_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean artifacts
            clean_text = gen_text.replace(f"[{target_emotion}]", "").replace("<|startoftext|>", "").strip()
            
            detected_emotion = classify_text(clean_text, teacher_model, teacher_tokenizer)
            
            true_labels.append(expected_label)
            pred_labels.append(detected_emotion)

    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n--- SPECIAL TOKEN RESULTS ---")
    print(f"Overall Arc Adherence Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, zero_division=0))
    
    # Save results
    pd.DataFrame({"Target": true_labels, "Detected": pred_labels}).to_csv("evaluation_results_special.csv", index=False)

if __name__ == "__main__":
    main()