import os
import logging
import pandas as pd
from datasets import load_dataset
from transformers import GPT2TokenizerFast

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "writing_prompts_subset.csv")
DATASET_NAME = "euclaise/writingprompts"
SAMPLE_SIZE = 5000  # Pull 5k, we will filter down
MIN_WORDS = 100     # Need enough text for an arc
MAX_WORDS = 800     # Keep within GPT-2 (1024 token) limits roughly

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ingest_data():
    logger.info(f"--- Starting Data Ingestion from {DATASET_NAME} ---")
    
    # 1. Download Dataset (Streaming mode to save RAM, or split slice)
    try:
        # Taking a slice of train split
        logger.info(f"Downloading first {SAMPLE_SIZE} examples...")
        ds = load_dataset(DATASET_NAME, split=f"train[:{SAMPLE_SIZE}]")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return

    # 2. Convert to Pandas for easier cleaning
    df = pd.DataFrame(ds)
    logger.info(f"Initial count: {len(df)} stories")

    # 3. Pre-processing & Cleaning
    # The dataset has 'prompt' and 'story'. We focus on 'story'.
    
    # Initialize tokenizer for length checks (fast approximation)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")
    
    clean_stories = []
    
    logger.info("Filtering data (Length Constraints: 100-800 words)...")
    
    for story in df['story']:
        # Basic text cleaning
        text = story.strip()
        word_count = len(text.split())
        
        # Filter by word count first (faster)
        if MIN_WORDS <= word_count <= MAX_WORDS:
            # Optional: Check token count to be safe
            token_count = len(tokenizer.encode(text, truncation=False, max_length=1024))
            if token_count < 1024:
                clean_stories.append(text)

    # 4. Create Clean Dataframe
    df_clean = pd.DataFrame(clean_stories, columns=["text"])
    logger.info(f"Filtered count: {len(df_clean)} high-quality stories")
    
    # 5. Save to Disk
    df_clean.to_csv(RAW_DATA_PATH, index=False)
    logger.info(f"Saved raw data to: {RAW_DATA_PATH}")

if __name__ == "__main__":
    # Ensure dependencies are installed
    # pip install datasets pandas transformers
    ingest_data()