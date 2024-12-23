import numpy as np
import pandas as pd
import sys
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import clone

# Fix random seed for reproducibility
random.seed(42)
np.random.seed(42)

############################################
#  Classes for speakers
############################################

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

############################################
#  Aliases
############################################
map_speakers_to_aliases = {
    "ר' ריבלין": "ראובן ריבלין",
    "ראובן ריבלין": "ראובן ריבלין",
    "רובי ריבלין": "ראובן ריבלין",
    "אברהם בורג": "א' בורג",
    "א' בורג": "א' בורג"
}

def get_speaker_name_from_alias(name: str) -> str:
    return map_speakers_to_aliases.get(name, name)

############################################
#  Hebrew Stop Words (sample)
############################################
hebrew_stopwords = [
    "על", "את", "אם", "של", "זה", "אני", "או",
    "הוא", "היא", "הם", "הן", "אבל", "כי", "מה",
    "כמו", "עם", "עוד", "היה", "הייתה", "היינו",
    # add or remove as relevant for your corpus
]

############################################
#  Create TF–IDF Feature Vector
############################################

def create_feature_vector(df):
    """
    Creates a TF–IDF vector with bigger n-grams, adjusted frequency thresholds, and sublinear TF.
    """
    vectorizer = TfidfVectorizer(
        stop_words=hebrew_stopwords,
        ngram_range=(1, 3),       # now capturing up to 3-grams
        min_df=1,                # keep even very rare terms
        max_df=0.9,              # ignore terms in >90% docs
        max_features=30000,      # expand the vocabulary
        sublinear_tf=True        # log-scale the term frequency
    )
    features = vectorizer.fit_transform(df['sentence_text'])
    return features, vectorizer

############################################
#  Create Custom Feature Vector
############################################
def create_custom_feature_vector(df):
    """
    Creates a custom numeric feature vector that does NOT rely on BoW:
      - sentence length (# tokens)
      - count of commas, periods, quotations, dashes, question marks
    """
    # We'll create new columns in df temporarily
    df['sentence_length'] = df['sentence_text'].str.split().str.len()
    df['commas'] = df['sentence_text'].str.count(r',')
    df['period'] = df['sentence_text'].str.count(r'\.')
    df['quotations'] = df['sentence_text'].str.count(r'"') + df['sentence_text'].str.count(r"'")
    df['dashes'] = df['sentence_text'].str.count(r'-')
    df['question'] = df['sentence_text'].str.count(r'\?')

    custom_features = df[[
        'sentence_length',
        'commas',
        'period',
        'quotations',
        'dashes',
        'question'
    ]].values
    return custom_features

############################################
#  Evaluate Classifier Function (5-fold)
############################################
def eval_classifier(features, encoded_labels, label_encoder, classifier_name):
    """
    Quick function to do cross_val_predict/cross_val_score
    with default KNN & LR (WITHOUT GridSearch).
    We pass in numeric labels and label_encoder for the
    classification_report's 'target_names'.
    """
    print(f"\n=== Evaluating: {classifier_name} ===")

    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'LogisticRegression': LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    }

    results = {}
    for name, clf in classifiers.items():
        # cross_val_predict/cross_val_score should use numeric labels
        preds_num = cross_val_predict(clf, features, encoded_labels, cv=5)
        scores = cross_val_score(clf, features, encoded_labels, cv=5)

        # decode numeric predictions to strings for the classification report
        preds_str = label_encoder.inverse_transform(preds_num)
        true_str  = label_encoder.inverse_transform(encoded_labels)

        # Now we can show them as "first","other","second"
        report = classification_report(true_str, preds_str, target_names=label_encoder.classes_, zero_division=0)

        mean_acc = np.mean(scores)
        print(f"\n[{name}] mean accuracy: {mean_acc:.4f}")
        print(f"{name} classification report:\n{report}")
        results[name] = {"accuracy": mean_acc, "report": report}
    return results

############################################
#  GridSearch for Fine-Tuning
############################################
def tune_knn(features, encoded_labels):
    """
    Tune KNN using GridSearchCV, searching over n_neighbors and p,
    but with numeric labels only.
    """
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'p': [1, 2]
    }
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(features, encoded_labels)
    print("\nBest KNN params:", grid.best_params_)
    print("Best KNN CV Accuracy: %.4f" % grid.best_score_)
    return grid.best_estimator_, grid.best_score_

def tune_logreg(features, encoded_labels):
    """
    Larger search over C, solver, and max_iter, plus possibility of 'penalty' variations
    (only valid combos).
    """
    param_grid = {
        'penalty': ['l2', 'l1'],      # but 'l1' requires 'liblinear'
        'solver': ['liblinear'],      # can also try 'lbfgs' but then penalty='l1' won't work
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'max_iter': [200, 500, 1000]
    }
    lr = LogisticRegression(random_state=42)
    grid = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(features, encoded_labels)
    print("\nBest LogisticRegression params:", grid.best_params_)
    print("Best LogisticRegression CV Accuracy: %.4f" % grid.best_score_)
    return grid.best_estimator_, grid.best_score_

############################################
#  Classification of New Sentences
############################################
def classify_sentences(input_file, output_file, model, vectorizer, label_encoder):
    """
    Reads new lines from `input_file`,
    predicts using the trained `model` with the `vectorizer`,
    writes the predicted labels to `output_file`.
    Expects model to produce numeric labels, so we decode via label_encoder.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    features = vectorizer.transform(sentences)
    preds_num = model.predict(features)
    preds_str = label_encoder.inverse_transform(preds_num)

    with open(output_file, 'w', encoding='utf-8') as f:
        for label in preds_str:
            f.write(f"{label}\n")

############################################
#  Main Script
############################################
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_jsonl_file> <output_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Initialize speaker objects
    first_speaker_instance = first_speaker("ראובן ריבלין")
    second_speaker_instance = second_speaker("א' בורג")
    other_speakers_instance = other_speakers("other")

    # Read input .jsonl
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            objects = json.loads(line.strip())
            raw_name = objects['speaker_name']
            name_under_the_alias = get_speaker_name_from_alias(raw_name)

            if name_under_the_alias == "ראובן ריבלין":
                first_speaker_instance.objects.append(objects)
            elif name_under_the_alias == "א' בורג":
                second_speaker_instance.objects.append(objects)
            else:
                other_speakers_instance.objects.append(objects)

    speaker_data = {
        first_speaker_instance.name: first_speaker_instance,
        second_speaker_instance.name: second_speaker_instance,
        other_speakers_instance.name: other_speakers_instance
    }

    speaker_counts = {
        sname: len(sobj.objects)
        for sname, sobj in speaker_data.items()
    }
    # Sort by descending count
    top_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    # Build data: top_speakers[1]-> 'first', top_speakers[2]-> 'second', else 'other'
    data = []
    for speaker_name, speaker_object in speaker_data.items():
        if speaker_name == top_speakers[1][0]:
            label = 'first'
        elif speaker_name == top_speakers[2][0]:
            label = 'second'
        else:
            label = 'other'
        for obj in speaker_object.objects:
            row = {
                "protocol_name": obj["protocol_name"],
                "knesset_number": obj["knesset_number"],
                "protocol_type": obj["protocol_type"],
                "protocol_number": obj["protocol_number"],
                "speaker_name": obj["speaker_name"],
                "sentence_text": obj["sentence_text"],
                "class": label
            }
            data.append(row)

    df = pd.DataFrame(data)

    # Down-sample each class
    min_count = df['class'].value_counts().min()
    df_downsampled = df.groupby('class', group_keys=False).apply(
        lambda x: x.sample(n=min_count, random_state=42)
    )
    df_downsampled.to_csv(output_file, index=False, encoding='utf-8-sig')

    # ----------------------------------------------------------------
    #  1) Convert your string labels ("first","other","second") to numeric
    # ----------------------------------------------------------------
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df_downsampled["class"])
    # e.g. "first" -> 0, "other" -> 1, "second" -> 2 (the actual order can vary)

    # ============ TF–IDF FEATURES ============
    features_tfidf, vectorizer_tfidf = create_feature_vector(df_downsampled)

    # ============ CUSTOM FEATURES ============
    custom_raw = create_custom_feature_vector(df_downsampled)
    scaler = StandardScaler()
    scaled_custom = scaler.fit_transform(custom_raw)

    # ============ Quick 5-fold Evals (no param tuning) ============
    tfidf_results  = eval_classifier(features_tfidf, encoded_labels, label_encoder, 'TF–IDF vector')
    custom_results = eval_classifier(scaled_custom,   encoded_labels, label_encoder, 'Custom vector')

    # ============ GRID SEARCH TUNING (TF–IDF) ============
    print("\n\n--- GridSearch Tuning on TF–IDF features ---")
    best_knn_tfidf, knn_tfidf_score = tune_knn(features_tfidf, encoded_labels)
    best_lr_tfidf,  lr_tfidf_score  = tune_logreg(features_tfidf, encoded_labels)

    # Evaluate final best models with cross_val_predict for classification report
    best_knn_preds = cross_val_predict(clone(best_knn_tfidf), features_tfidf, encoded_labels, cv=5)
    best_knn_preds_str = label_encoder.inverse_transform(best_knn_preds)
    true_str = label_encoder.inverse_transform(encoded_labels)
    best_knn_report = classification_report(true_str, best_knn_preds_str, target_names=label_encoder.classes_, zero_division=0)

    best_lr_preds = cross_val_predict(clone(best_lr_tfidf), features_tfidf, encoded_labels, cv=5)
    best_lr_preds_str = label_encoder.inverse_transform(best_lr_preds)
    best_lr_report = classification_report(true_str, best_lr_preds_str, target_names=label_encoder.classes_, zero_division=0)

    print("\nBest KNN on TF–IDF => CV Accuracy: %.4f" % knn_tfidf_score)
    print("Classification Report:\n", best_knn_report)

    print("\nBest LR on TF–IDF => CV Accuracy: %.4f" % lr_tfidf_score)
    print("Classification Report:\n", best_lr_report)

    # ============ GRID SEARCH TUNING (CUSTOM) ============
    print("\n\n--- GridSearch Tuning on CUSTOM features ---")
    best_knn_custom, knn_custom_score = tune_knn(scaled_custom, encoded_labels)
    best_lr_custom,  lr_custom_score  = tune_logreg(scaled_custom, encoded_labels)

    best_knn_preds2 = cross_val_predict(clone(best_knn_custom), scaled_custom, encoded_labels, cv=5)
    best_knn_preds2_str = label_encoder.inverse_transform(best_knn_preds2)
    best_knn_report2 = classification_report(true_str, best_knn_preds2_str, target_names=label_encoder.classes_, zero_division=0)

    best_lr_preds2 = cross_val_predict(clone(best_lr_custom), scaled_custom, encoded_labels, cv=5)
    best_lr_preds2_str = label_encoder.inverse_transform(best_lr_preds2)
    best_lr_report2 = classification_report(true_str, best_lr_preds2_str, target_names=label_encoder.classes_, zero_division=0)

    print("\nBest KNN on CUSTOM => CV Accuracy: %.4f" % knn_custom_score)
    print("Classification Report:\n", best_knn_report2)

    print("\nBest LR on CUSTOM => CV Accuracy: %.4f" % lr_custom_score)
    print("Classification Report:\n", best_lr_report2)

    # ============================================
    # Final pick: e.g. best logistic regression on TF–IDF
    # ============================================
    print("\n\n--- Final Model: Best LR (TF–IDF) ---")
    final_model = best_lr_tfidf
    final_model.fit(features_tfidf, encoded_labels)

    # Classify new sentences
    knesset_sentences = "knesset_sentences.txt"
    classification_output = "classification_results.txt"
    classify_sentences(knesset_sentences, classification_output, final_model, vectorizer_tfidf, label_encoder)

    print("\nDone.")
