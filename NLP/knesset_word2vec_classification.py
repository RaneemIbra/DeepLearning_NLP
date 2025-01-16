import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import random
import json
import sys
from gensim.models import Word2Vec
import re
random.seed(42)
np.random.seed(42)

map_speakers_to_aliases = {
    "ר' ריבלין": "ראובן ריבלין",
    "ראובן ריבלין": "ראובן ריבלין",
    "רובי ריבלין": "ראובן ריבלין",
    "אברהם בורג": "א' בורג",
    "א' בורג": "א' בורג"
}

def get_speaker_name_from_alias(name: str) -> str:
    return map_speakers_to_aliases.get(name, name)

def load_and_prepare_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    df = pd.DataFrame(data)
    df['speaker_name'] = df['speaker_name'].apply(get_speaker_name_from_alias)
    speaker_counts = df['speaker_name'].value_counts()
    top_speakers = speaker_counts.index[:2]
    print(top_speakers)

    df['class'] = df['speaker_name'].apply(lambda x: 'first' if x == top_speakers[0] else ('second' if x == top_speakers[1] else None))
    df = df[df['class'].notna()]

    min_samples = df['class'].value_counts().min()
    df = df.groupby('class').apply(lambda x: x.sample(n=min_samples, random_state=42)).reset_index(drop=True)

    return df

def eval_knn(features, labels, n_neighbors=9, metric='cosine'):

    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    # Encoding labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # 5-fold cross-validation
    predictions = cross_val_predict(classifier, features, encoded_labels, cv=5)
    # scores = cross_val_score(classifier, features, encoded_labels, cv=5)
    report = classification_report(encoded_labels, predictions, target_names=label_encoder.classes_, zero_division=0)

    # print(f"Mean Accuracy: {np.mean(scores):.4f}")
    print(f"{report}")

def load_corpus(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sentences.append(data['sentence_text'])
    return sentences

def preprocess_sentences(sentences):
    tokenized_sentences = []
    pattern = (
        r'\"[^\"]+\"|'          
        r'[א-ת]+\"[א-ת]+|'      
        r'[א-ת]+\'[א-ת]+|'    
        r'[א-ת]+'               
    )

    for sentence in sentences:
        tokens = re.findall(pattern, sentence)
        clean_tokens = [token.strip('"') for token in tokens]
        filtered_tokens = [token for token in clean_tokens if len(token) > 1]
        tokenized_sentences.append(filtered_tokens)
    return tokenized_sentences

def compute_sentence_embeddings(sentences, model):
    sentence_embeddings = []
    # compute the sentence embedding as the average of the word embeddings
    for sentence in sentences:
        valid_tokens = [word for word in sentence if word in model.wv]
        # if there are no valid tokens, the sentence embedding is a zero vector
        if valid_tokens:
            # compute the sentence embedding as the average of the word embeddings
            word_vectors = np.array([model.wv[word] for word in valid_tokens])
            sentence_vector = np.mean(word_vectors, axis=0)
        else:
            sentence_vector = np.zeros(model.vector_size)
        sentence_embeddings.append(sentence_vector)
    return sentence_embeddings

if __name__ == '__main__':
    if(len(sys.argv) != 3):
        print("Usage: knesset_word2vec_classification.py <corpus_file_path> <model_path>")
        sys.exit(1)
    knesset_corpus_path = sys.argv[1]
    model_path = sys.argv[2]

    model = Word2Vec.load(model_path)
    raw_sentences = load_corpus(knesset_corpus_path)
    tokenized_sentences = preprocess_sentences(raw_sentences)
    print("First 5 tokenized sentences:")
    for i in range(5):
        print(tokenized_sentences[i])
    
    sentence_embeddings = compute_sentence_embeddings(tokenized_sentences, model)
    df = load_and_prepare_data(knesset_corpus_path)
    print("DataFrame head:")
    print(df.head())
    print("\nDataFrame info:")
    print(df.info())
    print("\nFirst 5 sentence embeddings:")
    print(sentence_embeddings[:5])

    df = df.reset_index(drop=True)

    filtered_embeddings = np.array(sentence_embeddings)[df.index.values]

    eval_knn(filtered_embeddings, df['class'])