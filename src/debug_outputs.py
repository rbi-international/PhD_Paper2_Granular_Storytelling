import os
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models", "optimized_adapter")
BASE_MODEL_ID = "gpt2-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def debug_generation():
    print(f"--- Debugging Model Output on {DEVICE} ---")
    
    # Load
    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_ID)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.to(DEVICE)
    model.eval()

    # Test Prompts
    emotions = ["Joy", "Anger", "Fear"]
    prompt = "The phone rang loudly,"

    for emo in emotions:
        input_text = f"<|startoftext|> [{emo}] {prompt}"
        inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
        
        print(f"\nINPUT: {input_text}")
        
        # Generate with low temperature to see what the model 'really' wants to say
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, 
            do_sample=True, 
            temperature=0.6, 
            top_p=0.9
        )
        
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"OUTPUT: {text}")

if __name__ == "__main__":
    debug_generation()