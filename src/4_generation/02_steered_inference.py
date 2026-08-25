import os
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessor, LogitsProcessorList, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models", "final_gpt2_instruction")
BASE_MODEL_ID = "gpt2-medium"
TEACHER_NAME = "SamLowe/roberta-base-go_emotions"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_STORY_START = "The door opened slowly,"

# --- 1. The Neuro-Symbolic Dictionary ---
# We define "Anchor Words" for each emotion. 
# The model is nudged towards these if the instruction matches.
EMOTION_LEXICON = {
    "Joy": ["smile", "laugh", "happy", "delight", "wonderful", "bright", "cheer", "excited", "love", "joy", "grin", "warmth", "beautiful"],
    "Trust": ["friend", "trust", "believe", "safe", "rely", "honor", "truth", "accept", "help", "support", "agree"],
    "Fear": ["dark", "scared", "afraid", "shiver", "scream", "run", "terror", "shadow", "cold", "panic", "threat", "nervous"],
    "Surprise": ["shock", "gasp", "sudden", "unexpected", "stunned", "wow", "flash", "blink", "amazed", "jolt", "startle"],
    "Sadness": ["cry", "tear", "loss", "grief", "sob", "sad", "alone", "miss", "pain", "hurt", "empty", "sorry", "dark"],
    "Disgust": ["gross", "nasty", "sick", "vile", "foul", "ugly", "hate", "repulse", "dirty", "rotten", "trash"],
    "Anger": ["rage", "furious", "hate", "yell", "shout", "mad", "angry", "fist", "hit", "burn", "fight", "stupid", "annoy"],
    "Anticipation": ["wait", "hope", "ready", "soon", "plan", "wish", "dream", "expect", "watch", "look", "forward"]
}

# --- 2. The Steering Logic ---
class LexicalSteeringProcessor(LogitsProcessor):
    def __init__(self, tokenizer, target_emotion, boost_factor=5.0):
        self.tokenizer = tokenizer
        self.boost_factor = boost_factor
        self.target_words = EMOTION_LEXICON.get(target_emotion, [])
        
        # Convert words to token IDs
        self.token_ids = []
        for word in self.target_words:
            # We add a leading space because GPT-2 treats " word" and "word" differently
            ids1 = tokenizer.encode(word, add_special_tokens=False)
            ids2 = tokenizer.encode(" " + word, add_special_tokens=False)
            self.token_ids.extend(ids1 + ids2)
        
        self.token_ids = list(set(self.token_ids)) # Unique IDs

    def __call__(self, input_ids, scores):
        # Boost the score of target tokens
        if self.token_ids:
            for tid in self.token_ids:
                if tid < scores.shape[1]: # Safety check
                    scores[:, tid] += self.boost_factor
        return scores

# --- Evaluation Logic (Same as before, but with Steering) ---
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

def load_models():
    print(f"Loading Models on {DEVICE}...")
    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_ID)
    student_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(DEVICE)
    student_model.eval()

    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(TEACHER_NAME).to(DEVICE)
    teacher_model.eval()
    
    return student_model, tokenizer, teacher_model, teacher_tokenizer

def classify(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)
    top_idx = torch.argmax(probs, dim=1).item()
    return EMOTION_MAP.get(model.config.id2label[top_idx], "Neutral")

def main():
    student, student_tok, teacher, teacher_tok = load_models()
    
    true_labels, pred_labels, texts = [], [], []
    
    print("\n--- Evaluation with Lexical Steering ---")
    
    # Reduced samples for quick verification (increase to 20 for final)
    SAMPLES = 10 
    
    for emotion in EMOTION_LEXICON.keys():
        print(f"Testing: {emotion}...")
        expected = "Anger" if emotion == "Disgust" else emotion
        
        # Define the Steering Processor for this specific emotion
        steering = LexicalSteeringProcessor(student_tok, emotion, boost_factor=3.0)
        logits_processor = LogitsProcessorList([steering])
        
        for _ in range(SAMPLES):
            input_text = f"Emotion: {emotion} | Story: {PROMPT_STORY_START}"
            inputs = student_tok(input_text, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = student.generate(
                    **inputs,
                    max_new_tokens=60,
                    do_sample=True,
                    temperature=0.75,
                    top_p=0.9,
                    logits_processor=logits_processor, # Injecting the bias
                    pad_token_id=student_tok.eos_token_id
                )
            
            full = student_tok.decode(outputs[0], skip_special_tokens=True)
            try:
                story = full.split("| Story:")[1].strip()
            except:
                story = full
            
            detected = classify(story, teacher, teacher_tok)
            
            true_labels.append(expected)
            pred_labels.append(detected)
            texts.append(story)

    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n--- STEERED RESULTS ---")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(classification_report(true_labels, pred_labels, zero_division=0))
    
    # Save
    pd.DataFrame({"Target": true_labels, "Detected": pred_labels, "Story": texts}).to_csv("steered_results.csv", index=False)

if __name__ == "__main__":
    main()