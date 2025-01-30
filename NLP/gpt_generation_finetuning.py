# Install necessary libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk
import torch
import os
import sys

# Function to load IMDb subset or create one if not found
def get_imdb_subset(subset_path="imdb_subset", seed=42):
    if os.path.exists(subset_path):
        return load_from_disk(subset_path)

    dataset = load_dataset("imdb")  # Load full IMDb dataset
    subset = dataset["train"].shuffle(seed=seed).select(range(500))  # Create subset
    subset.save_to_disk(subset_path)  # Save for future use
    return subset

# Parse command-line arguments
imdb_subset_path = sys.argv[1]
output_file = sys.argv[2]
models_dir = sys.argv[3]

dataset = get_imdb_subset(imdb_subset_path)

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Ensure padding token is set
tokenizer.pad_token = tokenizer.eos_token

# Create subsets for positive and negative reviews
positive_reviews = dataset.filter(lambda x: x['label'] == 1).select(range(100))
negative_reviews = dataset.filter(lambda x: x['label'] == 0).select(range(100))

# Function to tokenize reviews
def tokenize_reviews(dataset, tokenizer, max_length=150):
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=max_length)

    return dataset.map(tokenize_function, batched=True).rename_column("label", "labels")

# Tokenize datasets
tokenized_positive_reviews = tokenize_reviews(positive_reviews, tokenizer)
tokenized_negative_reviews = tokenize_reviews(negative_reviews, tokenizer)

# Save tokenizer
tokenizer.save_pretrained(os.path.join(models_dir, "gpt2_positive"))
tokenizer.save_pretrained(os.path.join(models_dir, "gpt2_negative"))

# Training arguments
training_args = TrainingArguments(
    output_dir=models_dir,
    learning_rate=6e-5,
    per_device_train_batch_size=8,
    num_train_epochs=15,
    logging_dir=os.path.join(models_dir, "gpt2_logs"),
    logging_steps=100,
    report_to="none"
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Train Positive Model
model_positive = GPT2LMHeadModel.from_pretrained("gpt2")
trainer_positive = Trainer(
    model=model_positive,
    args=training_args,
    train_dataset=tokenized_positive_reviews,
    data_collator=data_collator
)
trainer_positive.train()
trainer_positive.save_model(os.path.join(models_dir, "gpt2_positive"))

# Train Negative Model
model_negative = GPT2LMHeadModel.from_pretrained("gpt2")
trainer_negative = Trainer(
    model=model_negative,
    args=training_args,
    train_dataset=tokenized_negative_reviews,
    data_collator=data_collator
)
trainer_negative.train()
trainer_negative.save_model(os.path.join(models_dir, "gpt2_negative"))

# Load trained models for generation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_positive = GPT2LMHeadModel.from_pretrained(os.path.join(models_dir, "gpt2_positive")).to(device)
model_negative = GPT2LMHeadModel.from_pretrained(os.path.join(models_dir, "gpt2_negative")).to(device)

# Define prompt for text generation
prompt = "The movie was"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generation settings
max_length = 100
temperature = 0.7
top_k = 40
top_p = 0.9
repetition_penalty = 1.2
attention_mask = input_ids.ne(tokenizer.pad_token_id)

# Generate Positive Reviews
positive_reviews = []
with torch.no_grad():
    for _ in range(5):
        output = model_positive.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True
        )
        positive_reviews.append(tokenizer.decode(output[0], skip_special_tokens=True))

# Generate Negative Reviews
negative_reviews = []
with torch.no_grad():
    for _ in range(5):
        output = model_negative.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True
        )
        negative_reviews.append(tokenizer.decode(output[0], skip_special_tokens=True))

# Save generated reviews to a file
with open(output_file, "w") as f:
    f.write("Reviews generated by positive model:\n")
    for i, review in enumerate(positive_reviews, 1):
        f.write(f"{i}. {review}\n")

    f.write("\nReviews generated by negative model:\n")
    for i, review in enumerate(negative_reviews, 1):
        f.write(f"{i}. {review}\n")