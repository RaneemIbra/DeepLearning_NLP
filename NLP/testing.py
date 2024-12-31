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
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
random.seed(42)
np.random.seed(42)

# a class for the other speakers
class other_speakers:
    def __init__(self, name):
        self.name = name
        self.objects = []

# a class for the speaker that appeared the most
class first_speaker:
    def __init__(self, name):
        self.name = name
        self.objects = []

# a class for the speaker that appeared the second most
class second_speaker:
    def __init__(self, name):
        self.name = name
        self.objects = []

# map the names that belong to one person to a single key in order to gather as many sentences
map_speakers_to_aliases = {
    "ר' ריבלין": "ראובן ריבלין",
    "ראובן ריבלין": "ראובן ריבלין",
    "רובי ריבלין": "ראובן ריבלין",
    "אברהם בורג": "א' בורג",
    "א' בורג": "א' בורג"
}

# a function that will retrive the value associated with the many aliasis the speaker goes by
def get_speaker_name_from_alias(name: str) -> str:
    return map_speakers_to_aliases.get(name, name)

# a function that creates the feature vector
def create_feature_vector(df):
    # we chose the tfidf vectorizer because it works better with our dataset
    # we also accounted for scaling and we took into consideration unigrams and bigrams
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        max_df=0.4,
        ngram_range=(1,2),
        max_features=5000,
        min_df=2
    )
    features = vectorizer.fit_transform(df['sentence_text'])
    labels = df['class']
    selector = SelectKBest(chi2, k=1000)
    selected_features = selector.fit_transform(features, labels)
    
    return selected_features, labels, vectorizer, selector

# a function to create a custom feature vector, we take into count the length of the sentence, and punctuation
def create_custom_feature_vector(df):
    df = df.copy()
    df['sentence_length'] = df['sentence_text'].str.split().str.len()
    df['commas'] = df['sentence_text'].str.count(r',')
    df['period'] = df['sentence_text'].str.count(r'\.')
    df['quotations'] = df['sentence_text'].str.count(r'"') + df['sentence_text'].str.count(r"'")
    df['dashes'] = df['sentence_text'].str.count(r'-')
    df['question'] = df['sentence_text'].str.count(r'\?')
    df['protocol_number'] = df['protocol_number'].astype(float)
    df['knesset_number'] = df['knesset_number'].astype(float)
    
    label_encoder = LabelEncoder()
    df['protocol_type'] = label_encoder.fit_transform(df['protocol_type'])
    
    custom_features = df[[
        'sentence_length',
        'commas',
        'period',
        'quotations',
        'dashes',
        'question',
        'protocol_number',
        'knesset_number',
        'protocol_type'
    ]].values
    return custom_features

# a function to evaluate the classifiers
def eval_classifier(features, labels, neighbors_count, weights, metric, C, solver):
    # define the classifiers with the best parameters
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=neighbors_count, weights=weights, metric=metric),
        'LogisticRegression': LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42)
    }

    results = {}
    # encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # iterate over the classifiers and use cross validation with 5 folds to evaluate them
    for name, classifier in classifiers.items():
        predictions = cross_val_predict(classifier, features, encoded_labels, cv=5)
        scores = cross_val_score(classifier, features, encoded_labels, cv=5)
        report = classification_report(encoded_labels, predictions, target_names=label_encoder.classes_, zero_division=0)
        results[name] = {
            "mean_accuracy": np.mean(scores),
            "classification_report": report
        }
        print(f"mean accuracy: {np.mean(scores):.4f}")
        print(f"classification report for {name}:\n{report}\n")
    return results

# a function to filter the dataframe for binary classification
def filter_binary_classes(df):
    return df[df['class'].isin(['first', 'second'])]

# a function to classify the sentences given in the knesset_sentences.txt file
# we open the file then for each line we try to classify it using the model passed
def classify_sentences(input_file, output_file, model, vectorizer, label_encoder, selector):
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = f.readlines()

    features = vectorizer.transform(sentences)
    selected_features = selector.transform(features)
    predictions = model.predict(selected_features)
    decoded_predictions = label_encoder.inverse_transform(predictions)

    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence, prediction in zip(sentences, decoded_predictions):
            f.write(f"{prediction}\n")

if __name__ == '__main__': 
    if len(sys.argv) != 4:
        # print("Usage: python script.py <input_jsonl_file> <input_sentences_file> <output_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    knesset_sentences = sys.argv[2]
    output_file = sys.argv[3]

    # create instances of the first and second speakers that we already found by running a small script
    first_speaker_instance = first_speaker("ראובן ריבלין")
    second_speaker_instance = second_speaker("א' בורג")
    other_speakers_instance = other_speakers("other")

    # gather all the sentences for the speakers and put them in lists
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

    # group the data
    speaker_data = {
        first_speaker_instance.name: first_speaker_instance,
        second_speaker_instance.name: second_speaker_instance,
        other_speakers_instance.name: other_speakers_instance
    }

    speaker_counts = {
        speaker_name: len(speaker_object.objects) 
        for speaker_name, speaker_object in speaker_data.items()
    }

    # sort the list of speakers from the one that appeared the most to the least
    top_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    # here we want to create the dataframe, so we iterate over the speakers that we have
    # and we give them the labels corresponding to their position, and then we start by
    # giving each row the data for each sentence, and then we append them to the list
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

    # create the data frame from the list
    df = pd.DataFrame(data)

    # find the class with the least amount of sentences to downsample the other classes
    minimum_counter = df['class'].value_counts().min()
    # perform the down-sampling by choosing randomly the sentences
    df_downsampled = (
        df.groupby('class', group_keys=False, as_index=False)
        .apply(lambda x: x.sample(n=minimum_counter, random_state=42))
    )

    # create the feature vector and the custom vector by calling the functions that we built
    features, labels, vectorizer, selector = create_feature_vector(df_downsampled)
    custom_features = create_custom_feature_vector(df_downsampled)

    # scale the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(custom_features)

    # binary classification
    df_binary = filter_binary_classes(df_downsampled)
    binary_features, binary_labels, _, _ = create_feature_vector(df_binary)
    binary_custom_features = create_custom_feature_vector(df_binary)
    scaled_binary_features = scaler.fit_transform(binary_custom_features)

    print("Binary Classification Results:")
    # evaluate classifiers on binary classification
    binary_tfidf_results = eval_classifier(binary_features, binary_labels, 7, 'distance', 'euclidean', 10.0, 'liblinear')
    binary_custom_results = eval_classifier(scaled_binary_features, binary_labels, 11, 'distance', 'manhattan', 0.1, 'lbfgs')

    print("Multiclass Classification Results:")
    # evaluate classifiers on multiclass classification
    tfidf_results = eval_classifier(features, df_downsampled['class'], 7, 'distance', 'euclidean', 10.0, 'lbfgs')
    custom_results = eval_classifier(scaled_features, df_downsampled['class'], 11, 'distance', 'manhattan', 0.1, 'lbfgs')

    # encode the labels to be able to pass valid value
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # use the best parameters for Logistic Regression model
    logistic_regression_model = LogisticRegression(C=10.0, solver='liblinear', max_iter=1000, random_state=42)
    logistic_regression_model.fit(features, encoded_labels)

    # after training the model we run it on the unseen sentences, and save the classification result
    classification_output = "classification_results.txt"
    classify_sentences(knesset_sentences, classification_output, logistic_regression_model, vectorizer, label_encoder, selector)