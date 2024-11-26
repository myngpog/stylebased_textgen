# modified code from ChatGPT (4o, paid version, November 26, 2024)
import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

MODEL_NAME = "gpt2"
OUTPUT_DIR = "model_output"

# Load the data (with my writing)
file_paths = ["writing_samples.jsonl", "sample_text.jsonl"]
dataset = load_dataset("json", data_files=file_paths)

# Loads pre-trained model of GPT2
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenize the dataset (reformat into a format the model can use)
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=False, padding=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Data collator to handle batching for more efficient parallel processing
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling
)

# Training function
def train_model():
    print("Loading GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )

    # training the model
    trainer.train()

    # Save the model and tokenizer into the directory
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train_model()
