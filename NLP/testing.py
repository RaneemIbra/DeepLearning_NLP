import numpy as np
import pandas as pd
import sys
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import re

random.seed(42)
np.random.seed(42)

class other_speakers:
    def __init__(self, name):
        self.name = name
        self.objects = []

class first_speaker:
    def __init__(self, name):
        self.name = name
        self.objects = []

class second_speaker:
    def __init__(self, name):
        self.name = name
        self.objects = []

map_speakers_to_aliases = {
    "ר' ריבלין": "ראובן ריבלין",
    "ראובן ריבלין": "ראובן ריבלין",
    "רובי ריבלין": "ראובן ריבלין",
    "אברהם בורג": "א' בורג",
    "א' בורג": "א' בורג"
}

def get_speaker_name_from_alias(name: str) -> str:
    return map_speakers_to_aliases.get(name, name)

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Retain alphanumeric and Hebrew characters, remove others
    text = re.sub(r'[^a-zA-Z\u0590-\u05FF\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text



def create_feature_vector(df):
    vectorizer = TfidfVectorizer(
        max_features=3000,
        min_df=1,
        ngram_range=(1, 2)  # Include unigrams and bigrams
    )
    df['processed_text'] = df['sentence_text'].apply(preprocess_text)
    
    # Debugging step: Check for empty rows
    if df['processed_text'].str.strip().eq('').any():
        empty_rows = df[df['processed_text'].str.strip().eq('')]
        print(f"Warning: Found {len(empty_rows)} empty rows in processed_text.")
        print(empty_rows)
        raise ValueError("Preprocessing resulted in empty rows. Please check the input data or preprocessing logic.")

    features = vectorizer.fit_transform(df['processed_text'])
    labels = df['class']
    return features, labels, vectorizer

def create_custom_feature_vector(df):
    df['sentence_length'] = df['sentence_text'].str.split().str.len()
    df['commas'] = df['sentence_text'].str.count(r',')
    df['periods'] = df['sentence_text'].str.count(r'\.')
    df['unique_words'] = df['sentence_text'].apply(lambda x: len(set(x.split())))
    df['avg_word_length'] = df['sentence_text'].apply(lambda x: np.mean([len(word) for word in x.split() if len(word) > 0]))
    custom_features = df[[  # Updated features
        'sentence_length',
        'commas',
        'periods',
        'unique_words',
        'avg_word_length',
    ]].values
    return custom_features

def eval_classifier(features, labels, classifier_name):
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    }

    results = {}
    print("\n")
    print(classifier_name)
    for name, classifier in classifiers.items():
        predictions = cross_val_predict(classifier, features, labels, cv=5)
        scores = cross_val_score(classifier, features, labels, cv=5)
        report = classification_report(labels, predictions, target_names=labels.unique(), zero_division=0)
        cm = confusion_matrix(labels, predictions)
        results[name] = {
            "mean_accuracy": np.mean(scores),
            "classification_report": report,
            "confusion_matrix": cm
        }
        print(f"Model: {name}")
        print(f"Mean accuracy: {np.mean(scores):.4f}")
        print(f"Classification report:\n{report}\n")
        print(f"Confusion Matrix:\n{cm}\n")
    return results

def tune_hyperparameters(features, labels):
    # Logistic Regression
    logistic_params = {'C': [0.1, 1, 10], 'max_iter': [500, 1000]}
    logistic_grid = GridSearchCV(LogisticRegression(random_state=42), logistic_params, cv=5)
    logistic_grid.fit(features, labels)

    # KNN
    knn_params = {'n_neighbors': [3, 5, 7, 9]}
    knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
    knn_grid.fit(features, labels)

    print("Best Logistic Regression Parameters:", logistic_grid.best_params_)
    print("Best KNN Parameters:", knn_grid.best_params_)
    return logistic_grid.best_estimator_, knn_grid.best_estimator_

def classify_sentences(input_file, output_file, model, vectorizer, label_encoder):
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    features = vectorizer.transform(sentences)
    predictions = model.predict(features)
    classified_labels = label_encoder.inverse_transform(predictions)
    with open(output_file, 'w', encoding='utf-8') as f:
        for label in classified_labels:
            f.write(f"{label}\n")

if __name__ == '__main__': 
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_jsonl_file> <output_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    first_speaker_instance = first_speaker("\u05e8\u05d0\u05d5\u05d1\u05df \u05e8\u05d9\u05d1\u05dc\u05d9\u05df")
    second_speaker_instance = second_speaker("\u05d0' \u05d1\u05d5\u05e8\u05d2")
    other_speakers_instance = other_speakers("other")

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            objects = json.loads(line.strip())
            raw_name = objects['speaker_name']
            name_under_the_alias = get_speaker_name_from_alias(raw_name)

            if name_under_the_alias == "\u05e8\u05d0\u05d5\u05d1\u05df \u05e8\u05d9\u05d1\u05dc\u05d9\u05df":
                first_speaker_instance.objects.append(objects)
            elif name_under_the_alias == "\u05d0' \u05d1\u05d5\u05e8\u05d2":
                second_speaker_instance.objects.append(objects)
            else:
                other_speakers_instance.objects.append(objects)

    speaker_data = {
        first_speaker_instance.name: first_speaker_instance,
        second_speaker_instance.name: second_speaker_instance,
        other_speakers_instance.name: other_speakers_instance
    }

    speaker_counts = {
        speaker_name: len(speaker_object.objects) 
        for speaker_name, speaker_object in speaker_data.items()
    }

    top_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    data = []
    for speaker_name, speaker_object in speaker_data.items():
        if speaker_name == top_speakers[1][0]:
            label = 'first'
        elif speaker_name == top_speakers[2][0]:
            label = 'second'
        else:
            label = 'other'
        for objects in speaker_object.objects:
            row = {
                "protocol_name": objects["protocol_name"],
                "knesset_number": objects["knesset_number"],
                "protocol_type": objects["protocol_type"],
                "protocol_number": objects["protocol_number"],
                "speaker_name": objects["speaker_name"],
                "sentence_text": objects["sentence_text"],
                "class": label
            }
            data.append(row)

    df = pd.DataFrame(data)

    minimum_counter = df['class'].value_counts().min()
    df_downsampled = (
        df.groupby('class', group_keys=False)
        .apply(lambda x: x.sample(n=minimum_counter, random_state=42))
    )
    df_downsampled.to_csv(output_file, index=False, encoding='utf-8-sig')

    features, labels, vectorizer = create_feature_vector(df_downsampled)
    custom_features = create_custom_feature_vector(df_downsampled)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(custom_features)

    # Hyperparameter Tuning
    best_logistic, best_knn = tune_hyperparameters(features, labels)

    # Evaluation
    print("\nEvaluating optimized models...")
    eval_classifier(features, labels, 'TFIDF Vector')
    eval_classifier(scaled_features, labels, 'Custom Vector')

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    best_logistic.fit(features, encoded_labels)

    knesset_sentences = "knesset_sentences.txt"
    classification_output = "classification_results.txt"
    classify_sentences(knesset_sentences, classification_output, best_logistic, vectorizer, label_encoder)
