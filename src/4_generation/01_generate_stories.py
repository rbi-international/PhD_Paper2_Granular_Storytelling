import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel, PeftConfig

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models", "final_adapter")
BASE_MODEL_ID = "gpt2-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Defines the 8 Plutchik Emotions we want to test
EMOTIONS = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation"]
TEST_PROMPT = "The old clock tower struck midnight,"

def generate_stories():
    print(f"--- Loading Models on {DEVICE} ---")
    
    # 1. Load Base Model & Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_ID)
    
    # 2. Load the LoRA Adapter (Your Fine-Tuning)
    print(f"Loading LoRA Adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.to(DEVICE)
    model.eval() # Inference Mode

    print("\n--- Generating Granular Emotional Arcs ---")
    print(f"Prompt: {TEST_PROMPT}\n")

    results = []

    for emotion in EMOTIONS:
        # Construct the input with the special tag
        input_text = f"<|startoftext|> [{emotion}] {TEST_PROMPT}"
        
        inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100,      # Generate ~100 words
                do_sample=True,          # Creative generation
                temperature=0.7,         # Balance creativity/coherence
                top_p=0.9,               # Nucleus sampling
                repetition_penalty=1.2,  # Prevent getting stuck
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the artifacting (remove the tag from the output view)
        clean_text = generated_text.replace(f"[{emotion}]", "").strip()
        
        print(f"[{emotion.upper()}]:\n{clean_text}\n{'-'*40}")
        results.append((emotion, clean_text))

    # Optional: Save to file for inspection
    with open("generated_samples.txt", "w", encoding="utf-8") as f:
        for emo, text in results:
            f.write(f"EMOTION: {emo}\nTEXT: {text}\n\n")

if __name__ == "__main__":
    generate_stories()