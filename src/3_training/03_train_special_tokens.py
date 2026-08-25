import os
import logging
import pandas as pd
import torch
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "final_train_ready.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "student_special_tokens")
FINAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "final_special_adapter")

MODEL_ID = "gpt2-medium"
MAX_LENGTH = 256 

# --- New Tokens ---
# We define these explicitly so the tokenizer treats them as unique atomic units
SPECIAL_EMOTIONS = [
    "[Joy]", "[Trust]", "[Fear]", "[Surprise]", 
    "[Sadness]", "[Disgust]", "[Anger]", "[Anticipation]"
]

# Hyperparameters
BATCH_SIZE = 2
GRAD_ACCUMULATION = 16
LEARNING_RATE = 2e-4 # Slightly lower LR for embeddings stability
EPOCHS = 5           # 5 Epochs is usually enough for embeddings to converge
LORA_RANK = 32       # Rank 32 is a good balance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_tokenize_data(tokenizer):
    df = pd.read_csv(DATA_PATH)
    dataset = Dataset.from_pandas(df[["formatted_text"]])

    def tokenize_function(examples):
        outputs = tokenizer(
            examples["formatted_text"], 
            truncation=True, 
            max_length=MAX_LENGTH, 
            padding="max_length"
        )
        return outputs

    logger.info("Tokenizing dataset with Special Tokens...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["formatted_text"])
    return tokenized_datasets

def train():
    logger.info(f"--- Starting Special Token Training ---")
    
    # 1. Setup Tokenizer & Add Tokens
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token 
    
    # CRITICAL: Add the emotions as special tokens
    num_added_toks = tokenizer.add_tokens(SPECIAL_EMOTIONS)
    logger.info(f"Added {num_added_toks} special emotion tokens to vocabulary.")

    # 2. Load Model & Resize Embeddings
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    model.resize_token_embeddings(len(tokenizer)) # Resize input layer to fit new tokens
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False

    # 3. LoRA Configuration with Embedding Training
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=LORA_RANK,
        lora_alpha=64, 
        lora_dropout=0.1,
        # THIS IS THE KEY: We tell LoRA to train the Word Token Embeddings (wte)
        # and the Output Head (lm_head) so our new tokens actually learn meaning.
        modules_to_save=["wte", "lm_head"] 
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() 

    # 4. Prepare Data
    train_data = load_and_tokenize_data(tokenizer)
    
    # 5. Training Args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_dir=f"{PROJECT_ROOT}/logs/training_logs_special",
        logging_steps=20,
        save_steps=100,
        fp16=True, 
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    logger.info("Starting training loop...")
    trainer.train()
    
    # 6. Save (We must save the tokenizer too because it changed!)
    logger.info(f"Saving Special Token Adapter to {FINAL_MODEL_DIR}")
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

if __name__ == "__main__":
    train()