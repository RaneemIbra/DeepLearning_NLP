#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import json
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

random.seed(42)
np.random.seed(42)


################################################################################
# Helper classes (optional, as in your original code)
################################################################################

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

################################################################################
# Map some known aliases to unify speaker names
################################################################################

map_speakers_to_aliases = {
    "ר' ריבלין": "ראובן ריבלין",
    "ראובן ריבלין": "ראובן ריבלין",
    "רובי ריבלין": "ראובן ריבלין",
    "אברהם בורג": "א' בורג",
    "א' בורג": "א' בורג"
}

def get_speaker_name_from_alias(name: str) -> str:
    return map_speakers_to_aliases.get(name, name)


################################################################################
# Create TF-IDF features
################################################################################

def create_tfidf_vector(df):
    """
    Create TF-IDF features for 'sentence_text'.
    Adjust ngram_range and other hyperparams to improve accuracy.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        min_df=2,
        max_features=20000    # can increase if you have enough memory
    )
    features = vectorizer.fit_transform(df['sentence_text'])
    return features, vectorizer


################################################################################
# Classification of new sentences
################################################################################

def classify_sentences(input_file, output_file, model, vectorizer, label_encoder):
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    X_test = vectorizer.transform(sentences)
    y_pred = model.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    with open(output_file, 'w', encoding='utf-8') as f:
        for label in y_pred_labels:
            f.write(f"{label}\n")


################################################################################
# Main script
################################################################################

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_jsonl_file> <output_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]   # e.g. knesset_corpus.jsonl
    output_file = sys.argv[2]  # e.g. output.csv (ensure you have permission!)

    # Create speaker instances
    first_speaker_instance = first_speaker("ראובן ריבלין")
    second_speaker_instance = second_speaker("א' בורג")
    other_speakers_instance = other_speakers("other")

    # Read JSONL
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_obj = json.loads(line.strip())
            raw_name = line_obj['speaker_name']
            name_alias = get_speaker_name_from_alias(raw_name)

            if name_alias == "ראובן ריבלין":
                first_speaker_instance.objects.append(line_obj)
            elif name_alias == "א' בורג":
                second_speaker_instance.objects.append(line_obj)
            else:
                other_speakers_instance.objects.append(line_obj)

    # Put them in a dictionary
    speaker_data = {
        first_speaker_instance.name: first_speaker_instance,
        second_speaker_instance.name: second_speaker_instance,
        other_speakers_instance.name: other_speakers_instance
    }

    # Count sentences
    speaker_counts = {
        sname: len(sobj.objects)
        for sname, sobj in speaker_data.items()
    }
    # Sort by number of sentences
    top_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    # Prepare data
    # We'll label the top speaker as 'first', second as 'second', the rest 'other'
    # or adapt logic depending on your assignment's exact instructions
    data_rows = []
    for sname, sobj in speaker_data.items():
        if sname == top_speakers[0][0]:
            label = 'first'
        elif sname == top_speakers[1][0]:
            label = 'second'
        else:
            label = 'other'
        for obj in sobj.objects:
            row = {
                "protocol_name": obj["protocol_name"],
                "knesset_number": obj["knesset_number"],
                "protocol_type": obj["protocol_type"],
                "protocol_number": obj["protocol_number"],
                "speaker_name": obj["speaker_name"],
                "sentence_text": obj["sentence_text"],
                "class": label
            }
            data_rows.append(row)

    df = pd.DataFrame(data_rows)

    # Balance the classes by down-sampling
    # Avoid the groupby warning by resetting index:
    min_count = df['class'].value_counts().min()
    df_downsampled = (
        df.groupby('class', group_keys=False, as_index=False)
          .apply(lambda x: x.sample(n=min_count, random_state=42))
          .reset_index(drop=True)
    )

    # Try saving to a different path if you still have the permission error
    df_downsampled.to_csv(
        output_file,
        index=False,
        encoding='utf-8-sig'
    )

    # Label-encode the classes before model training
    label_encoder = LabelEncoder()
    df_downsampled['class_num'] = label_encoder.fit_transform(df_downsampled['class'])

    # Build TF-IDF features
    X, vectorizer = create_tfidf_vector(df_downsampled)
    y = df_downsampled['class_num']

    # Let's do grid search for LogisticRegression
    from sklearn.model_selection import GridSearchCV

    param_grid_lr = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear', 'saga'],  # 'saga' can handle l1 or l2
        'max_iter': [1000]
    }
    lr = LogisticRegression(random_state=42)
    grid_search_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search_lr.fit(X, y)
    best_lr = grid_search_lr.best_estimator_

    print("\nBest Logistic Regression params:", grid_search_lr.best_params_)
    print("Best Logistic Regression CV accuracy:", grid_search_lr.best_score_)

    # Grid search for KNN
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    knn = KNeighborsClassifier()
    grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search_knn.fit(X, y)
    best_knn = grid_search_knn.best_estimator_

    print("\nBest KNN params:", grid_search_knn.best_params_)
    print("Best KNN CV accuracy:", grid_search_knn.best_score_)

    # Evaluate both with cross_val_predict
    from sklearn.model_selection import cross_val_predict

    # Logistic Regression
    lr_preds = cross_val_predict(best_lr, X, y, cv=5)
    print("\nLogistic Regression classification report:")
    print(classification_report(
        y, lr_preds,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    # KNN
    knn_preds = cross_val_predict(best_knn, X, y, cv=5)
    print("\nKNN classification report:")
    print(classification_report(
        y, knn_preds,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    # Choose best model
    if grid_search_lr.best_score_ >= grid_search_knn.best_score_:
        final_model = best_lr
        print("\nChosen final model: Logistic Regression")
    else:
        final_model = best_knn
        print("\nChosen final model: KNN")

    # Fit final model on entire dataset
    final_model.fit(X, y)

    # Classify new sentences
    knesset_sentences_file = "knesset_sentences.txt"
    classification_output_file = "classification_results.txt"

    classify_sentences(
        knesset_sentences_file,
        classification_output_file,
        final_model,
        vectorizer,
        label_encoder
    )

    print(f"\nClassification results saved to: {classification_output_file}")
    print("Done.")
