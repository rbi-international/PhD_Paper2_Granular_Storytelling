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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "student_checkpoints_opt")
FINAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "optimized_adapter")

MODEL_ID = "gpt2-medium"
MAX_LENGTH = 256 # Reduced slightly to allow for higher Rank/Batch

# --- OPTIMIZED HYPERPARAMETERS ---
BATCH_SIZE = 2       # Reduced to 2 to save VRAM for the larger LoRA Rank
GRAD_ACCUMULATION = 16 # Increased to maintain effective batch size of 32
LEARNING_RATE = 5e-4 # More aggressive learning rate
EPOCHS = 10          # Train longer to force convergence
LORA_RANK = 64       # 8x more capacity than before
LORA_ALPHA = 128     # Alpha is usually 2x Rank

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

    logger.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["formatted_text"])
    return tokenized_datasets

def train():
    logger.info(f"--- Starting OPTIMIZED LoRA Training (Rank {LORA_RANK}) ---")
    
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token 

    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    for param in model.parameters():
        param.requires_grad = False

    # OPTIMIZED CONFIG
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=LORA_RANK,      # The big change
        lora_alpha=LORA_ALPHA, 
        lora_dropout=0.05
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() 

    train_data = load_and_tokenize_data(tokenizer)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_dir=f"{PROJECT_ROOT}/logs/training_logs_opt",
        logging_steps=50,
        save_steps=200,
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
    
    logger.info(f"Saving Optimized Adapter to {FINAL_MODEL_DIR}")
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

if __name__ == "__main__":
    train()