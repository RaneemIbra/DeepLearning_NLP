import json
from collections import defaultdict, Counter
import math
import pandas as pd
import time


class Trigram_LM:
    def __init__(self, vocab_size):
        self.models = {"committee": defaultdict(int), "plenary": defaultdict(int)}
        self.bigrams = {"committee": defaultdict(int), "plenary": defaultdict(int)}
        self.unigrams = {"committee": defaultdict(int), "plenary": defaultdict(int)}
        self.total = {"committee": 0, "plenary": 0}
        self.default_type = "committee"
        self.laplace_constant = 1
        self.vocab_size = vocab_size
        self.vocabulary = set()

    def calculate_prob_of_sentence(self, space_separated_tokens):
        tokens = space_separated_tokens.split()
        tokens = ["<s>", "<s>"] + tokens + ["</s>"]
        log_prob = 0.0
        protocol_type = self.default_type
        for i in range(2, len(tokens)):
            trigrams = (tokens[i - 2], tokens[i - 1], tokens[i])
            bigrams = (tokens[i - 1], tokens[i])
            unigrams = tokens[i]

            unigrams_count = self.unigrams[protocol_type][unigrams]
            bigrams_count = self.bigrams[protocol_type][bigrams]
            trigrams_count = self.models[protocol_type][trigrams]
            total_count = self.total[protocol_type]

            if total_count > 0:
                unigram_prob = (unigrams_count + self.laplace_constant) / (total_count + self.vocab_size)
            else:
                unigram_prob = 0

            if self.unigrams[protocol_type][tokens[i - 1]] > 0:
                bigrams_prob = (bigrams_count + self.laplace_constant) / (
                    self.unigrams[protocol_type][tokens[i - 1]] + self.vocab_size
                )
            else:
                bigrams_prob = 0

            if self.bigrams[protocol_type][(tokens[i - 2], tokens[i - 1])] > 0:
                trigrams_prob = (trigrams_count + self.laplace_constant) / (
                    self.bigrams[protocol_type][(tokens[i - 2], tokens[i - 1])] + self.vocab_size
                )
            else:
                trigrams_prob = 0

            probability = trigrams_prob * 0.1 + bigrams_prob * 0.3 + unigram_prob * 0.6
            if probability > 0:
                log_prob += math.log(probability)
            else:
                log_prob += float("-inf")
        return log_prob

    def generate_next_token(self, space_separated_tokens):
        tokens = space_separated_tokens.split()
        if len(tokens) < 2:
            tokens = ["<s>"] * (2 - len(tokens)) + tokens
        last_token = tokens[-1]
        second_last_token = tokens[-2]
        generated_token = None
        max_probability = float("-inf")
        for i in self.vocabulary:
            trigram = (second_last_token, last_token, i)
            bigram = (last_token, i)
            unigram = i

            unigrams_count = self.unigrams[self.default_type][unigram]
            bigrams_count = self.bigrams[self.default_type][bigram]
            trigrams_count = self.models[self.default_type][trigram]
            total_count = self.total[self.default_type]

            if total_count > 0:
                unigram_prob = (unigrams_count + self.laplace_constant) / (total_count + self.vocab_size)
            else:
                unigram_prob = 0

            if self.unigrams[self.default_type][last_token] > 0:
                bigrams_prob = (bigrams_count + self.laplace_constant) / (
                    self.unigrams[self.default_type][last_token] + self.vocab_size
                )
            else:
                bigrams_prob = 0

            if self.bigrams[self.default_type][(second_last_token, last_token)] > 0:
                trigrams_prob = (trigrams_count + self.laplace_constant) / (
                    self.bigrams[self.default_type][(second_last_token, last_token)] + self.vocab_size
                )
            else:
                trigrams_prob = 0

            probability = trigrams_prob * 0.1 + bigrams_prob * 0.3 + unigram_prob * 0.6
            if probability > max_probability:
                max_probability = probability
                generated_token = i
        return generated_token, max_probability


def compute_idf(documents):
    """Precompute IDF values for all terms in the documents."""
    total_docs = len(documents)
    doc_frequency = defaultdict(int)
    for doc in documents:
        unique_terms = set(doc.split())
        for term in unique_terms:
            doc_frequency[term] += 1

    return {term: math.log(total_docs / (1 + freq)) for term, freq in doc_frequency.items()}


def extract_ngrams(sentences, n):
    """Generate n-grams from a list of sentences."""
    ngrams = Counter()
    for sentence in sentences:
        words = sentence.split()
        if len(words) < n:
            continue
        ngrams.update(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return ngrams


def get_k_n_t_collocations(k, n, t, corpus, type, idf_cache):
    results = {"committee": {}, "plenary": {}}

    for protocol_type in ["committee", "plenary"]:
        sentences = corpus[corpus["protocol_type"] == protocol_type]["sentence_text"].tolist()
        ngrams = extract_ngrams(sentences, n)

        if type == "frequency":
            filtered_collocations = {coll: freq for coll, freq in ngrams.items() if freq >= t}
            sorted_collocations = sorted(filtered_collocations.items(), key=lambda x: x[1], reverse=True)
            results[protocol_type] = dict(sorted_collocations[:k])
        elif type == "tfidf":
            tf_idf_scores = {}
            total_terms = sum(ngrams.values())
            for ngram, freq in ngrams.items():
                term = " ".join(ngram)
                tf = freq / total_terms
                tf_idf_scores[ngram] = tf * idf_cache.get(term, 0)

            sorted_tfidf = sorted(tf_idf_scores.items(), key=lambda x: x[1], reverse=True)
            results[protocol_type] = dict(sorted_tfidf[:k])

    return results


def save_collocation_to_file(results, n, type):
    """Save collocations to a file."""
    with open("knesset_collocations.txt", "a", encoding="utf-8") as file:
        header = f"{n}-gram collocations:\n{type.capitalize()}:\n"
        file.write(header)
        for protocol_type, collocations in results.items():
            file.write(f"{protocol_type.capitalize()} corpus:\n")
            for collocation, value in collocations.items():
                collocation_text = " ".join(collocation)
                file.write(f"{collocation_text}: {value:.4f}\n")
            file.write("\n")


if __name__ == "__main__":
    print("Starting the script...")
    try:
        with open("knesset_corpus.jsonl", "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]
        corpus = pd.DataFrame(data)
        print(f"Loaded {len(corpus)} records from 'knesset_corpus.jsonl'.")
    except FileNotFoundError:
        print("Error: File 'knesset_corpus.jsonl' not found.")
        exit()

    k = 10  # Top k collocations
    lengths = [2, 3, 4]  # Collocation lengths
    t = 5  # Minimum frequency threshold

    # Precompute IDF
    idf_cache = compute_idf(corpus["sentence_text"].tolist())

    for n in lengths:
        for type in ["frequency", "tfidf"]:
            start_time = time.time()
            print(f"Processing n={n}, type={type}...")
            collocations = get_k_n_t_collocations(k=k, n=n, t=t, corpus=corpus, type=type, idf_cache=idf_cache)
            print(f"Generated collocations for n={n}, type={type}: {len(collocations['committee']) + len(collocations['plenary'])} items.")
            save_collocation_to_file(collocations, n, type)
            print(f"Completed n={n}, type={type} in {time.time() - start_time:.2f} seconds.")

    print("Collocations have been saved to 'knesset_collocations.txt'.")
