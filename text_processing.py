# modified code from ChatGPT (4o, paid version, November 26, 2024, December 9th 2024, and December 11th 2024)
import torch, json
import nltk
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

nltk.download("punkt_tab")

MODEL_NAME_NEO = "EleutherAI/gpt-neo-125m"
MODEL_NAME_GPT2 = "gpt2"
OUTPUT_DIR = "model_output_GPT2"

# Loads pre-trained model of GPT2
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME_GPT2)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

"""
This function splits the object into sentences so it can be properly processed by the model.
"""
def split_story_into_sentences(story_text):
    sentences = nltk.sent_tokenize(story_text)  # Split the text into sentences
    return sentences  # Return a list of sentences

# Read the JSONL file
file_paths = ["sample_text.jsonl", "writing_samples.jsonl"]
output_file = "split_samples.jsonl"
raw_dataset = load_dataset("json", data_files=file_paths)

# Process each story in the dataset
with open(output_file, "w", encoding="utf-8") as out_f:
    for item in raw_dataset["train"]:  # Hugging Face datasets use splits like "train"
        story_text = item["text"] 
        sentences = split_story_into_sentences(story_text)  # Split the text into sentences
        for sentence in sentences:
            out_f.write(json.dumps({"text": sentence}) + "\n")  # Save each sentence as a new JSON object

# Tokenize the dataset with truncation and padding
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True,  # Truncate sequences exceeding the max length
        padding="max_length",  # Pad shorter sequences to the max length
        max_length=1024  # Set the model's max context length
    )

processed_dataset = load_dataset("json", data_files=output_file)
tokenized_dataset = processed_dataset.map(tokenize_function, batched=True)

# Data collator to handle batching for more efficient parallel processing
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling
)

# Training function
def train_model():
    print("Loading GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME_GPT2)

    # Resize token embeddings to match tokenizer's vocabulary size
    model.resize_token_embeddings(len(tokenizer))
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=4,
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