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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "student_checkpoints")
FINAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "final_adapter")

MODEL_ID = "gpt2-medium"
MAX_LENGTH = 300 # Truncate strictly to save VRAM (Story + Prompt usually < 300)

# Hyperparameters for RTX 3060 (6GB)
BATCH_SIZE = 4       # Small batch size to fit memory
GRAD_ACCUMULATION = 8 # Accumulate to simulate Batch Size = 32
LEARNING_RATE = 3e-4 # LoRA usually needs higher LR than full finetuning
EPOCHS = 3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_tokenize_data(tokenizer):
    """Loads CSV and converts to HuggingFace Dataset."""
    df = pd.read_csv(DATA_PATH)
    
    # Create Dataset object
    dataset = Dataset.from_pandas(df[["formatted_text"]])

    def tokenize_function(examples):
        outputs = tokenizer(
            examples["formatted_text"], 
            truncation=True, 
            max_length=MAX_LENGTH, 
            padding="max_length"
        )
        return outputs

    logger.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["formatted_text"])
    return tokenized_datasets

def train():
    logger.info(f"--- Starting LoRA Training for {MODEL_ID} ---")
    
    # 1. Setup Tokenizer
    # GPT-2 doesn't have a pad token by default, so we use EOS
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token 

    # 2. Load Base Model
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    # Freeze base model to save memory (Safety check, LoRA does this too usually)
    for param in model.parameters():
        param.requires_grad = False

    # 3. Apply LoRA
    # r=Rank (8 is standard), alpha=Scaling, dropout=Regularization
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() # This will show you the massive efficiency gain

    # 4. Prepare Data
    train_data = load_and_tokenize_data(tokenizer)
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_dir=f"{PROJECT_ROOT}/logs/training_logs",
        logging_steps=10,
        save_steps=100,
        fp16=True, # Mixed Precision is CRITICAL for 6GB VRAM
        save_total_limit=2,
        report_to="none" # Disable wandb for local run
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 7. Execute
    logger.info("Starting training loop...")
    trainer.train()
    
    # 8. Save Final Model
    logger.info(f"Saving Final Adapter to {FINAL_MODEL_DIR}")
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

if __name__ == "__main__":
    train()