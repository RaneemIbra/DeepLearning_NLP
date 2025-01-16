import os
import re
import json
import numpy as np
import pandas as pd
import sys
from gensim.models import Word2Vec
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# a function to preprocess the corpus and return a list of tokenized sentences
def preprocess_corpus(filepath):
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            sentence = record.get('sentence_text', '')
            # remove non-Hebrew characters and tokenize the sentence
            tokens = [re.sub(r'[^\u0590-\u05FF]', '', token) for token in sentence.split()]
            tokens = [token for token in tokens if token and len(token) > 1]
            if tokens:
                sentences.append(tokens)
    return sentences

# a function to compute the embeddings of the sentences using the Word2Vec model
def compute_embeddings(sentences, model_path):
    model = Word2Vec.load(model_path)
    embeddings = []
    for sentence in sentences:
        # compute the sentence embedding as the average of the word embeddings
        word_vectors = [model.wv[word] for word in sentence if word in model.wv]
        if word_vectors:
            embeddings.append(np.mean(word_vectors, axis=0))
        else:
            embeddings.append(np.zeros(model.vector_size))
    return embeddings

# map the names that belong to one person to a single key to gather as many sentences
map_speakers_to_aliases = {
    "ר' ריבלין": "ראובן ריבלין",
    "ראובן ריבלין": "ראובן ריבלין",
    "רובי ריבלין": "ראובן ריבלין",
    "אברהם בורג": "א' בורג",
    "א' בורג": "א' בורג"
}

# function to retrieve the value associated with the many aliases the speaker goes by
def get_speaker_name_from_alias(name: str) -> str:
    return map_speakers_to_aliases.get(name, name)

# function to extract the speaker data from the corpus
def extract_speaker_data(corpus_df, model_path):
    sentences, embeddings, speakers = [], [], []
    model = Word2Vec.load(model_path)
    for _, row in corpus_df.iterrows():
        sentence = row['sentence_text']
        raw_speaker = row['speaker_name']
        speaker = get_speaker_name_from_alias(raw_speaker)
        if speaker in ["ראובן ריבלין", "א' בורג"]:
            sentences.append(sentence)
            speakers.append(speaker)
            tokens = tokenize_sentence(sentence)
            embeddings.append(compute_mean_embedding(tokens, model))
    return sentences, embeddings, speakers

# function to tokenize a sentence
def tokenize_sentence(sentence):
    tokens = [re.sub(r'[^\u0590-\u05FF]', '', token) for token in sentence.split()]
    return [token for token in tokens if token and len(token) > 1]

# function to compute the mean embedding of a list of tokens
def compute_mean_embedding(tokens, model):
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

# function to balance the classes by undersampling the majority class
def downsample_classes(sentences, embeddings, speakers):
    speaker1_count = speakers.count("ראובן ריבלין")
    speaker2_count = speakers.count("א' בורג")
    majority_class, minority_class = ("ראובן ריבלין", "א' בורג") if speaker1_count > speaker2_count else ("א' בורג", "ראובן ריבלין")
    majority_indices = [i for i, s in enumerate(speakers) if s == majority_class]
    minority_indices = [i for i, s in enumerate(speakers) if s == minority_class]
    np.random.seed(42)
    np.random.shuffle(majority_indices)
    majority_indices = majority_indices[:len(minority_indices)]
    balanced_indices = majority_indices + minority_indices
    return [sentences[i] for i in balanced_indices], [embeddings[i] for i in balanced_indices], [speakers[i] for i in balanced_indices]

# function to evaluate the classifier using 5-fold cross-validation
def evaluate_classifier(embeddings, labels):
    features = np.array(embeddings)
    encoded_labels = LabelEncoder().fit_transform(labels)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    predictions = cross_val_predict(knn, features, encoded_labels, cv=skf)
    report = classification_report(encoded_labels, predictions, target_names=LabelEncoder().fit(labels).classes_, zero_division=0)
    print(report)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)
    corpus_path, model_path = sys.argv[1], sys.argv[2]
    try:
        tokenized_sentences = preprocess_corpus(corpus_path)
        model = Word2Vec.load(model_path)
    except Exception as e:
        sys.exit(f"Error loading the model or processing the input file: {e}")
    corpus_df = pd.read_json(corpus_path, lines=True)
    sentences, embeddings, speakers = extract_speaker_data(corpus_df, model_path)
    balanced_sentences, balanced_embeddings, balanced_speakers = downsample_classes(sentences, embeddings, speakers)
    evaluate_classifier(balanced_embeddings, balanced_speakers)
