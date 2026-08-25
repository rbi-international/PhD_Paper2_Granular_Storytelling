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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "gpt2_instruction_checkpoints")
FINAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "final_gpt2_instruction")

MODEL_ID = "gpt2-medium"
MAX_LENGTH = 256 

# High-Capacity Hyperparameters
BATCH_SIZE = 2
GRAD_ACCUMULATION = 16
LEARNING_RATE = 3e-4 
EPOCHS = 15
LORA_RANK = 128     # VERY HIGH RANK -> Fits 6GB VRAM but learns much more
LORA_ALPHA = 256    # 2x Rank

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
    logger.info(f"--- Starting GPT-2 High-Rank Instruction Training ---")
    
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token 

    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    # Freeze base
    for param in model.parameters():
        param.requires_grad = False

    # TARGET ALL LINEAR LAYERS for maximum impact
    # GPT-2 layer names: c_attn, c_proj, c_fc
    target_modules = ["c_attn", "c_proj", "c_fc"]

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA, 
        lora_dropout=0.1,
        target_modules=target_modules, 
        modules_to_save=["wte", "lm_head"] # Keep training embeddings too
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
        logging_dir=f"{PROJECT_ROOT}/logs/training_logs_instruction",
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
    
    logger.info(f"Saving Instruction Adapter to {FINAL_MODEL_DIR}")
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

if __name__ == "__main__":
    train()