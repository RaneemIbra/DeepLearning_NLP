import numpy as np
import pandas as pd
import sys
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
random.seed(42)
np.random.seed(42)

# Why not combine them into one class? is there an actual reason?
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

def create_feature_vector(df):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(df['sentence_text'])
    labels = df['class']
    return features, labels, vectorizer

def create_custom_feature_vector(df):
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
        'question',
    ]].values
    return custom_features

def eval_classifier(features, labels, classifier_name):
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'LogisticRegression': LogisticRegression(max_iter= 1000, random_state=42)
    }

    results = {}
    print("\n")
    print(classifier_name)
    for name, classifier in classifiers.items():
        predictions = cross_val_predict(classifier, features, labels, cv=5)
        scores = cross_val_score(classifier, features, labels, cv=5)
        report = classification_report(labels, predictions, target_names=labels.unique(), zero_division=0)
        results[name] = {
            "mean_accuracy": np.mean(scores),
            "classification_report": report
        }
        print(f"mean accuracy: {np.mean(scores):.4f}")
        print(f"classification report for {name}:\n{report}\n")
    return results

if __name__ == '__main__': 
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_jsonl_file> <output_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    first_speaker_instance = first_speaker("ראובן ריבלין")
    second_speaker_instance = second_speaker("א' בורג")
    other_speakers_instance = other_speakers("other")

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
        speaker_name: len(speaker_object.objects) 
        for speaker_name, speaker_object in speaker_data.items()
    }

    top_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    data = []
    for speaker_name, speaker_object in speaker_data.items():
        if speaker_name == top_speakers[1][0]:
            label = 'first speaker'
        elif speaker_name == top_speakers[2][0]:
            label = 'second speaker'
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

    df = pd.DataFrame(data) # break then we will be back :)

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
    tfidf_results = eval_classifier(features, df_downsampled['class'], 'TFIDF vector')
    custom_results = eval_classifier(scaled_features, df_downsampled['class'], 'custom vector')
    print("evaluation completed")