import os
import pandas as pd
from sklearn.utils import resample

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "annotated", "annotated_train.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "final_train_ready.csv")

TARGET_COUNT = 600 # Target for Neutral (close to Sadness count)

def balance_and_format():
    print("--- Balancing Dataset Distribution ---")
    df = pd.read_csv(INPUT_FILE)
    
    # 1. Merge Disgust (4) into Anger (Similar quadrant in Plutchik)
    df.loc[df['emotion_label'] == 'Disgust', 'emotion_label'] = 'Anger'
    print("Merged 'Disgust' into 'Anger' due to low sample count.")

    # 2. Separate Classes
    df_neutral = df[df['emotion_label'] == 'Neutral']
    df_others = df[df['emotion_label'] != 'Neutral']

    # 3. Undersample Neutral
    # We downsample Neutral to prevent it from overwhelming the emotional signal
    df_neutral_downsampled = resample(df_neutral, 
                                      replace=False, 
                                      n_samples=TARGET_COUNT, 
                                      random_state=42)
    
    # 4. Recombine
    df_balanced = pd.concat([df_neutral_downsampled, df_others])
    
    # 5. Format for GPT-2 Training
    # Format: <|startoftext|> [EMOTION] Story... <|endoftext|>
    # Note: We add a special bracketed tag for the emotion.
    df_balanced['formatted_text'] = (
        "<|startoftext|> [" + df_balanced['emotion_label'] + "] " + 
        df_balanced['text'] + " <|endoftext|>"
    )

    # 6. Save
    # We save just the formatted text column for training, but keep label for reference
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle
    df_balanced.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Balanced Dataset Saved: {OUTPUT_FILE}")
    print("\n--- New Distribution ---")
    print(df_balanced['emotion_label'].value_counts())
    print("\n--- Sample Input Format ---")
    print(df_balanced['formatted_text'].iloc[0][:100] + "...")

if __name__ == "__main__":
    balance_and_format()