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
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "instruction_train_ready.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "gpt2_large_checkpoints")
FINAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "gpt2_large_optimized")

# THE UPGRADE: GPT-2 Large
MODEL_ID = "gpt2-large"
MAX_LENGTH = 256

# Hyperparameters for "Max Juice" on 6GB VRAM
BATCH_SIZE = 1           # Strict limit for Large
GRAD_ACCUMULATION = 32   # Higher accumulation to simulate Batch Size 32 (Stable gradients)
LEARNING_RATE = 2e-4     # Standard LoRA rate
EPOCHS = 10              # Large models converge faster
LORA_RANK = 32           # Rank 32 is powerful enough for Large
LORA_ALPHA = 64          # 2x Rank

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
    logger.info(f"--- Starting GPT-2 LARGE Optimized Training ---")
    
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token 

    logger.info("Loading GPT-2 Large Model...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    
    # CRITICAL: Gradient Checkpointing allows Large model training on 6GB
    model.gradient_checkpointing_enable()
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False

    # TARGET ALL LINEAR MODULES for Maximum "Juice"
    # c_attn = Attention, c_proj = Output Projection, c_fc = Fully Connected
    target_modules = ["c_attn", "c_proj", "c_fc"]

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA, 
        lora_dropout=0.1,
        target_modules=target_modules,
        # We also train the Head and Embeddings to ensure the "Instruction" structure is learned perfectly
        modules_to_save=["lm_head", "wte"] 
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
        logging_dir=f"{PROJECT_ROOT}/logs/gpt2_large_logs",
        logging_steps=10,
        save_steps=200,
        fp16=True,             # FP16 is mandatory
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
    
    logger.info(f"Saving Optimized GPT-2 Large Adapter to {FINAL_MODEL_DIR}")
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

if __name__ == "__main__":
    train()