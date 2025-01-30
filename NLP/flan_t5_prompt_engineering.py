import os
import sys
import json
import re
import pandas as pd
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import accuracy_score

def get_imdb_subset(subset_path="imdb_subset", seed=42):
    # Check if the subset already exists
    if os.path.exists(subset_path):
        subset = load_from_disk(subset_path)  # Load from disk
    else:
        dataset = load_dataset("imdb")  # Load full IMDb dataset
        subset = dataset["train"].shuffle(seed=seed).select(range(500))  # Create subset
        subset.save_to_disk(subset_path)  # Save subset to disk
    return subset

# Function to classify a review using a specified prompt
def classify_review(model, tokenizer, review, prompt):
    formatted_prompt = prompt.format(review=review)
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python flan_t5_prompt_engineering.py <path/to/imdb_subset> <path/to/flan_t5_imdb_results.txt>")

    subset_path = sys.argv[1]
    output_file_path = sys.argv[2]

    dataset = get_imdb_subset(subset_path)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    # Convert the dataset to a pandas DataFrame for easier manipulation
    df = pd.DataFrame(dataset['text'], columns=["text"])
    df['label'] = dataset['label']

    # Separate positive and negative reviews
    positive_reviews = df[df['label'] == 1]
    negative_reviews = df[df['label'] == 0]

    # Sample 25 reviews from each class to ensure stratification
    sampled_positive_reviews = positive_reviews.sample(n=25, random_state=42)
    sampled_negative_reviews = negative_reviews.sample(n=25, random_state=42)

    # Combine the two samples back into a single DataFrame
    sampled_reviews = pd.concat([sampled_positive_reviews, sampled_negative_reviews])

    # Shuffle the combined dataset to mix positive and negative reviews
    sampled_reviews = sampled_reviews.sample(frac=1, random_state=42).reset_index(drop=True)

    # Zero-shot Prompt: No example provided
    zeroShot = "Classify the sentiment of the following movie review as positive or negative: {review}"

    # Few-shot Prompt: Provide two examples before the actual task
    fewShot = """
    Classify the sentiment of the following movie reviews as positive or negative:
    Review 1 (Positive): 'I absolutely loved this movie! It was heartwarming and thrilling.'
    Review 2 (Negative): 'This was a terrible movie. It was boring and too long.'
    Now classify the sentiment of this new review: {review}
    """

    # Instruction-based Prompt: Detailed instructions for the task
    instructionBased = """
    You are a sentiment analysis model. Your task is to determine whether a given movie review expresses a positive or negative sentiment.
    Please read the following review and classify its sentiment as either 'positive' or 'negative': {review}
    """

    # Iterate over sampled reviews and classify each with all prompt types
    results = []
    for index, row in sampled_reviews.iterrows():
        review_text = row['text']
        true_label = 'Positive' if row['label'] == 1 else 'Negative'

        # Classify with zero-shot prompt
        zero_shot_result = classify_review(model, tokenizer, review_text, zeroShot)

        # Classify with few-shot prompt
        few_shot_result = classify_review(model, tokenizer, review_text, fewShot)

        # Classify with instruction-based prompt
        instruction_based_result = classify_review(model, tokenizer, review_text, instructionBased)

        # Collect results for this review
        results.append({
            'Review': review_text,
            'True Label': true_label,
            'Zero-Shot': zero_shot_result,
            'Few-Shot': few_shot_result,
            'Instruction-Based': instruction_based_result
        })

    # Convert results to DataFrame for easier handling
    results_df = pd.DataFrame(results)

    # Save the results to a text file
    with open(output_file_path, 'w') as f:
        for index, result in results_df.iterrows():
            f.write(f"Review {index + 1}: {result['Review']}\n")
            f.write(f"Review {index + 1} True Label: {result['True Label']}\n")
            f.write(f"Review {index + 1} Zero-Shot: {result['Zero-Shot']}\n")
            f.write(f"Review {index + 1} Few-Shot: {result['Few-Shot']}\n")
            f.write(f"Review {index + 1} Instruction-Based: {result['Instruction-Based']}\n\n")

    print(f"Classification results saved to '{output_file_path}'")

    # Extract true labels and predictions for each prompting strategy
    true_labels = results_df['True Label'].tolist()
    zero_shot_predictions = results_df['Zero-Shot'].apply(lambda x: 'Positive' if 'positive' in x.lower() else 'Negative').tolist()
    few_shot_predictions = results_df['Few-Shot'].apply(lambda x: 'Positive' if 'positive' in x.lower() else 'Negative').tolist()
    instruction_based_predictions = results_df['Instruction-Based'].apply(lambda x: 'Positive' if 'positive' in x.lower() else 'Negative').tolist()

    # Calculate accuracy for each prompt type
    zero_shot_accuracy = accuracy_score(true_labels, zero_shot_predictions)
    few_shot_accuracy = accuracy_score(true_labels, few_shot_predictions)
    instruction_based_accuracy = accuracy_score(true_labels, instruction_based_predictions)

    # Print the accuracies
    print(f"Zero-Shot Accuracy: {zero_shot_accuracy:.2f}")
    print(f"Few-Shot Accuracy: {few_shot_accuracy:.2f}")
    print(f"Instruction-Based Accuracy: {instruction_based_accuracy:.2f}")

    # Save the analysis to a file or display it as needed
    analysis_results = {
        "Zero-Shot": zero_shot_accuracy,
        "Few-Shot": few_shot_accuracy,
        "Instruction-Based": instruction_based_accuracy
    }

    # Optional: Save the results to a CSV or text file for reporting
    results_analysis_df = pd.DataFrame([analysis_results])
    results_analysis_df.to_csv('prompting_strategy_accuracy_analysis.csv', index=False)

    print("Detailed accuracy analysis saved to 'prompting_strategy_accuracy_analysis.csv'.")
