import numpy as np
import pandas as pd
import sys
import json
import random
import re
from collections import Counter

# Sklearn imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

################################################################################
# Optional: Custom Hebrew stopwords (placeholder)
################################################################################
hebrew_stopwords = {
    'של', 'על', 'עם', 'הוא', 'היא', 'זה', 'לא', 'כן',
    'אני', 'מה', 'האם', 'אתה', 'את', 'אנחנו', 'יש',
    'אם', 'גם', 'כל', 'כי', 'או', 'שהוא', 'שהיא'
}
# You may wish to expand or replace this list with a more comprehensive set.

################################################################################
# Speaker Classes
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
# Speaker name mapping
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
# Basic text cleaning
################################################################################
def preprocess_text(text: str) -> str:
    """
    Basic text preprocessing for Hebrew:
    1. Lowercase (if appropriate for Hebrew).
    2. Remove digits.
    3. Remove extra punctuation.
    4. Remove stopwords (optional).
    """
    # Lowercase
    text = text.lower()

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove characters outside Hebrew letters, basic punctuation
    text = re.sub(r'[^א-ת\s\'\".,\-?!]', '', text)

    # Remove Hebrew stopwords (optional)
    words = text.split()
    filtered_words = [w for w in words if w not in hebrew_stopwords]
    text = ' '.join(filtered_words)

    return text

################################################################################
# TF-IDF feature creation
################################################################################
def create_feature_vector(df,
                          ngram_range=(1, 2),
                          max_features=10000,
                          min_df=2,
                          stop_words=None):
    """
    Create TF-IDF features from the `cleaned_text` column of df.
    """
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        max_df=0.7,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        stop_words=stop_words
    )
    features = vectorizer.fit_transform(df['cleaned_text'])
    # we do not return 'labels' here since we now handle numeric labels separately
    return features, vectorizer

################################################################################
# Custom numeric features
################################################################################
def create_custom_feature_vector(df):
    """
    Create custom numeric features from `cleaned_text`, e.g. punctuation counts.
    """
    df['sentence_length'] = df['cleaned_text'].str.split().str.len()
    df['commas'] = df['cleaned_text'].str.count(r',')
    df['period'] = df['cleaned_text'].str.count(r'\.')
    df['quotations'] = df['cleaned_text'].str.count(r'"') + df['cleaned_text'].str.count(r"'")
    df['dashes'] = df['cleaned_text'].str.count(r'-')
    df['question'] = df['cleaned_text'].str.count(r'\?')

    features = df[[
        'sentence_length',
        'commas',
        'period',
        'quotations',
        'dashes',
        'question'
    ]].values
    return features

################################################################################
# Evaluation function: uses numeric labels, decodes for reporting
################################################################################
def eval_classifier(features, numeric_labels, classifier, label_encoder, cv=5):
    """
    Evaluate a classifier with cross_val_predict on numeric labels,
    then decode them for the classification report and confusion matrix.
    """
    preds_numeric = cross_val_predict(classifier, features, numeric_labels, cv=cv)
    scores = cross_val_score(classifier, features, numeric_labels, cv=cv)

    # Decode integer labels -> original string labels
    y_true_str = label_encoder.inverse_transform(numeric_labels)
    y_pred_str = label_encoder.inverse_transform(preds_numeric)

    report = classification_report(
        y_true_str, y_pred_str, 
        target_names=label_encoder.classes_, 
        zero_division=0
    )
    conf_mat = confusion_matrix(
        y_true_str, y_pred_str, 
        labels=label_encoder.classes_
    )
    return preds_numeric, scores, report, conf_mat

################################################################################
# Simple matplotlib confusion matrix
################################################################################
def plot_confusion_matrix(cm, label_names, title="Confusion Matrix"):
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

################################################################################
# Classify new sentences
################################################################################
def classify_sentences(input_file, output_file, model, vectorizer, label_encoder):
    """
    Uses the final trained model to classify new sentences.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    cleaned_sentences = [preprocess_text(s) for s in sentences]
    features = vectorizer.transform(cleaned_sentences)
    pred_numeric = model.predict(features)
    pred_str = label_encoder.inverse_transform(pred_numeric)

    with open(output_file, 'w', encoding='utf-8') as f:
        for label in pred_str:
            f.write(f"{label}\n")

################################################################################
# Main script
################################################################################
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_jsonl_file> <output_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Speaker objects
    first_speaker_instance = first_speaker("ראובן ריבלין")
    second_speaker_instance = second_speaker("א' בורג")
    other_speakers_instance = other_speakers("other")

    # Read JSONL lines, parse, and store
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            raw_name = entry['speaker_name']
            alias_name = get_speaker_name_from_alias(raw_name)

            if alias_name == "ראובן ריבלין":
                first_speaker_instance.objects.append(entry)
            elif alias_name == "א' בורג":
                second_speaker_instance.objects.append(entry)
            else:
                other_speakers_instance.objects.append(entry)

    speaker_data = {
        first_speaker_instance.name: first_speaker_instance,
        second_speaker_instance.name: second_speaker_instance,
        other_speakers_instance.name: other_speakers_instance
    }

    # Check distribution
    speaker_counts = {
        k: len(v.objects) for k, v in speaker_data.items()
    }
    # Sort descending by count
    top_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    # Build a DataFrame
    data_rows = []
    for speaker_name, speaker_object in speaker_data.items():
        # Assign class label based on top_speakers ranking
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
            data_rows.append(row)

    df = pd.DataFrame(data_rows)

    # Downsample so each class has the same number of examples
    min_count = df['class'].value_counts().min()
    df_downsampled = (
        df.groupby('class', group_keys=False)
          .apply(lambda x: x.sample(n=min_count, random_state=42), include_groups = False)
    )

    # Save the downsampled dataset
    df_downsampled.to_csv(output_file, index=False, encoding='utf-8-sig')

    # Preprocess text
    df_downsampled['cleaned_text'] = df_downsampled['sentence_text'].apply(preprocess_text)

    ############################################################################
    # 1) Label Encode the classes => numeric labels
    ############################################################################
    label_encoder = LabelEncoder()
    df_downsampled['numeric_label'] = label_encoder.fit_transform(df_downsampled['class'])
    # e.g. if label_encoder.classes_ => ['first','other','second'] => 0,1,2
    labels_numeric = df_downsampled['numeric_label'].values
    label_names = label_encoder.classes_  # array of string labels

    ############################################################################
    # 2) Create TF-IDF features
    ############################################################################
    tfidf_features, tfidf_vectorizer = create_feature_vector(
        df_downsampled,
        ngram_range=(1,3),
        max_features=10000,
        min_df=2,
        stop_words=None
    )

    ############################################################################
    # 3) Create custom features, scale them
    ############################################################################
    custom_features = create_custom_feature_vector(df_downsampled)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(custom_features)

    ############################################################################
    # Hyperparameter Tuning
    # Expanded param grids to hopefully improve accuracy
    ############################################################################
    lr_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']  # saga can handle l1/l2
    }
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1=Manhattan, 2=Euclidean
    }

    # 3-A) LogisticRegression (TF-IDF)
    lr = LogisticRegression(max_iter=5000, random_state=42)
    lr_gridsearch = GridSearchCV(lr, lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    lr_gridsearch.fit(tfidf_features, labels_numeric)
    best_lr_tfidf = lr_gridsearch.best_estimator_

    print("Best LogisticRegression params (TF-IDF):", lr_gridsearch.best_params_)
    print("Best LogisticRegression accuracy (TF-IDF):", lr_gridsearch.best_score_)

    # Evaluate
    _, lr_scores_tfidf, lr_report_tfidf, lr_confmat_tfidf = eval_classifier(
        tfidf_features, labels_numeric, best_lr_tfidf, label_encoder, cv=5
    )
    print("\nLogisticRegression (TF-IDF) CV Accuracy:", np.mean(lr_scores_tfidf))
    print("Classification Report (TF-IDF):")
    print(lr_report_tfidf)
    print("Confusion Matrix (TF-IDF):")
    print(lr_confmat_tfidf)
    plot_confusion_matrix(lr_confmat_tfidf, label_names, title="LR (TF-IDF) Confusion Matrix")

    # 3-B) KNN (TF-IDF)
    knn = KNeighborsClassifier()
    knn_gridsearch = GridSearchCV(knn, knn_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    knn_gridsearch.fit(tfidf_features, labels_numeric)
    best_knn_tfidf = knn_gridsearch.best_estimator_

    print("\nBest KNN params (TF-IDF):", knn_gridsearch.best_params_)
    print("Best KNN accuracy (TF-IDF):", knn_gridsearch.best_score_)

    _, knn_scores_tfidf, knn_report_tfidf, knn_confmat_tfidf = eval_classifier(
        tfidf_features, labels_numeric, best_knn_tfidf, label_encoder, cv=5
    )
    print("\nKNN (TF-IDF) CV Accuracy:", np.mean(knn_scores_tfidf))
    print("Classification Report (TF-IDF):")
    print(knn_report_tfidf)
    print("Confusion Matrix (TF-IDF):")
    print(knn_confmat_tfidf)
    plot_confusion_matrix(knn_confmat_tfidf, label_names, title="KNN (TF-IDF) Confusion Matrix")

    ############################################################################
    # 3-C) LogisticRegression (Custom features)
    ############################################################################
    lr_gridsearch_custom = GridSearchCV(lr, lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    lr_gridsearch_custom.fit(scaled_features, labels_numeric)
    best_lr_custom = lr_gridsearch_custom.best_estimator_

    print("\nBest LogisticRegression params (CUSTOM):", lr_gridsearch_custom.best_params_)
    print("Best LogisticRegression accuracy (CUSTOM):", lr_gridsearch_custom.best_score_)

    _, lr_scores_custom, lr_report_custom, lr_confmat_custom = eval_classifier(
        scaled_features, labels_numeric, best_lr_custom, label_encoder, cv=5
    )
    print("\nLogisticRegression (Custom) CV Accuracy:", np.mean(lr_scores_custom))
    print("Classification Report (Custom):")
    print(lr_report_custom)
    print("Confusion Matrix (Custom):")
    print(lr_confmat_custom)
    plot_confusion_matrix(lr_confmat_custom, label_names, title="LR (Custom) Confusion Matrix")

    ############################################################################
    # 3-D) KNN (Custom features)
    ############################################################################
    knn_gridsearch_custom = GridSearchCV(knn, knn_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    knn_gridsearch_custom.fit(scaled_features, labels_numeric)
    best_knn_custom = knn_gridsearch_custom.best_estimator_

    print("\nBest KNN params (CUSTOM):", knn_gridsearch_custom.best_params_)
    print("Best KNN accuracy (CUSTOM):", knn_gridsearch_custom.best_score_)

    _, knn_scores_custom, knn_report_custom, knn_confmat_custom = eval_classifier(
        scaled_features, labels_numeric, best_knn_custom, label_encoder, cv=5
    )
    print("\nKNN (Custom) CV Accuracy:", np.mean(knn_scores_custom))
    print("Classification Report (Custom):")
    print(knn_report_custom)
    print("Confusion Matrix (Custom):")
    print(knn_confmat_custom)
    plot_confusion_matrix(knn_confmat_custom, label_names, title="KNN (Custom) Confusion Matrix")

    ############################################################################
    # 4) Final Model Training & Classification
    #    We pick the best LR on TF-IDF, as an example, but you can pick any.
    ############################################################################
    final_model = best_lr_tfidf
    final_model.fit(tfidf_features, labels_numeric)

    # Classify new sentences
    knesset_sentences = "knesset_sentences.txt"
    classification_output = "classification_results.txt"
    classify_sentences(knesset_sentences, classification_output, final_model, tfidf_vectorizer, label_encoder)

    ############################################################################
    # 5) Distribution of words & class overlap
    ############################################################################
    vocab = tfidf_vectorizer.get_feature_names_out()
    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(df_downsampled['cleaned_text'])
    word2idx = count_vect.vocabulary_
    idx2word = {v: k for k, v in word2idx.items()}
    total_counts = np.asarray(counts.sum(axis=0)).ravel().tolist()

    # Sort by descending frequency
    sorted_word_counts = sorted(zip(total_counts, idx2word.values()), key=lambda x: x[0], reverse=True)
    print("\nTop 20 Words by Raw Frequency:")
    for freq, word in sorted_word_counts[:20]:
        print(f"{word}: {freq}")

    # Class overlap
    class_counts = Counter(df_downsampled['class'])
    print("\nClass Distribution (After Downsampling):")
    for cls_name, cnt in class_counts.items():
        print(f"{cls_name}: {cnt}")

    print("\nScript completed.")
