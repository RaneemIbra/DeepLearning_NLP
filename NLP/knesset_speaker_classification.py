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
from sklearn.preprocessing import LabelEncoder
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
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df= 0.4, ngram_range=(1,2))
    features = vectorizer.fit_transform(df['sentence_text'])
    labels = df['class']
    return features, labels, vectorizer

# a function to create a custom feature vector, we take into count the length of the sentence, and punctuation
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

# a function to evaluate the classifiers
def eval_classifier(features, labels, classifier_name):
    # define the classifiers and map them to a key, the parameters choice is explained in the pdf
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=9),
        'LogisticRegression': LogisticRegression(max_iter= 1000, random_state=42)
    }

    results = {}
    print("\n")
    print(classifier_name)
    # iterate over the classifiers and use cross validation with 5 folds to evaluate them
    # then report the classification and return the results
    for name, classifier in classifiers.items():
        predictions = cross_val_predict(classifier, features, labels, cv=5)
        scores = cross_val_score(classifier, features, labels, cv=5)
        report = classification_report(labels, predictions, target_names=labels.unique(), zero_division=0)
        results[name] = {
            "mean_accuracy": np.mean(scores),
            "classification_report": report
        }
        # print(f"mean accuracy: {np.mean(scores):.4f}")
        # print(f"classification report for {name}:\n{report}\n")
    return results

# a function to classify the sentences given in the knesset_sentences.txt file
# we open the file then for each line we try to classify it using the model passed
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
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_jsonl_file> <input_sentences_file> <output_csv_file>")
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
        df.groupby('class', group_keys=False)
        .apply(lambda x: x.sample(n=minimum_counter, random_state=42))
    )
    # df_downsampled.to_csv(output_file, index=False, encoding='utf-8-sig')

    # create the feature vector and the custom vector by calling the functions that we built
    features, labels, vectorizer = create_feature_vector(df_downsampled)
    custom_features = create_custom_feature_vector(df_downsampled)

    # scale the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(custom_features)

    # then evaluate the classifiers on the vectors that we created
    tfidf_results = eval_classifier(features, df_downsampled['class'], 'TFIDF vector')
    custom_results = eval_classifier(scaled_features, df_downsampled['class'], 'custom vector')
    
    # encode the labels to be able to pass valid value
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # we chose the logistic regression model because it classified the data better than KNN
    logistic_regression_model = LogisticRegression(max_iter=1000, random_state=42)
    logistic_regression_model.fit(features, encoded_labels)

    # after training the model we run it on the unseen sentences, and save the classification result
    classification_output = "classification_results.txt"
    classify_sentences(knesset_sentences, classification_output, logistic_regression_model, vectorizer, label_encoder)