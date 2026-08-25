import os
import pandas as pd

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "final_train_ready.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "instruction_train_ready.csv")

def reformat_data():
    print("--- Reformatting Data for GPT-2 Instruction Tuning ---")
    df = pd.read_csv(INPUT_FILE)
    
    # We strip the old format and build the new Instruction format
    # Format: "Emotion: [EMOTION] | Story: [TEXT] <|endoftext|>"
    # The "|" acts as a strong separator (delimiter) for the model.
    
    new_formatted = []
    
    for index, row in df.iterrows():
        # Clean up the text (remove the old [Joy] tag if it exists in the raw text, though it shouldn't)
        text = str(row['text']).strip()
        emotion = row['emotion_label']
        
        # The New Structure
        fmt = f"Emotion: {emotion} | Story: {text} <|endoftext|>"
        new_formatted.append(fmt)

    df['formatted_text'] = new_formatted
    
    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved Instruction-Formatted Data to: {OUTPUT_FILE}")
    print(f"Sample: {new_formatted[0][:100]}...")

if __name__ == "__main__":
    reformat_data()