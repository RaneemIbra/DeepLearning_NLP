import os
from datasets import load_dataset, load_from_disk
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score
import sys

def get_imdb_subset(subset_path="imdb_subset", seed=42):
    # Check if the subset already exists
    if os.path.exists(subset_path):
        subset = load_from_disk(subset_path)  # Load from disk
    else:
        dataset = load_dataset("imdb")  # Load full IMDb dataset
        subset = dataset["train"].shuffle(seed=seed).select(range(500))  # Create subset
        subset.save_to_disk(subset_path)  # Save subset to disk
    return subset

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python bert_classification_finetuning.py <subset_path>")
    subset_path = sys.argv[1] 
    output_dir = "imdb_finetuned_results"

    if(os.path.exists(subset_path)):
        dataset = load_from_disk(subset_path)
    else:
        dataset = get_imdb_subset(subset_path)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    # Split dataset into training and evaluation sets
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=4,
        report_to="none"
    )

    # Use data collator to handle padding dynamically
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    results = trainer.predict(eval_dataset)
    predictions = results.predictions.argmax(axis=-1)
    labels = results.label_ids

    # Compute accuracy
    accuracy = accuracy_score(labels, predictions)
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")