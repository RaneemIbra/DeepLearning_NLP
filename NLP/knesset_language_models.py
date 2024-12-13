import json
from collections import defaultdict, Counter
import math
import pandas as pd
import time
import random

class Trigram_LM:
    """
    A Trigram Language Model class that supports linear interpolation with Laplace smoothing.
    
    Attributes:
        trigrams (dict): Nested dicts holding trigram counts for 'committee' and 'plenary'.
        bigrams (dict): Nested dicts holding bigram counts for 'committee' and 'plenary'.
        unigrams (dict): Nested dicts holding unigram counts for 'committee' and 'plenary'.
        total (dict): Holds total token counts per protocol type.
        vocabulary (set): Holds all tokens seen by the model.
        vocab_size (int): Size of the vocabulary (unique tokens).
        default_type (str): Default protocol type, either 'committee' or 'plenary'.
        
        lambda1, lambda2, lambda3 (float): Interpolation weights for unigram, bigram, and trigram probabilities.
        
        frequency_penalty_factor (float): A factor to penalize extremely frequent tokens.
        punctuation_tokens (set): Set of punctuation tokens to penalize.
        per_token_penalty (dict): Additional penalties for specific tokens.
    """

    def __init__(self):
        # count dictionaries
        self.trigrams = {"committee": defaultdict(int), "plenary": defaultdict(int)}
        self.bigrams = {"committee": defaultdict(int), "plenary": defaultdict(int)}
        self.unigrams = {"committee": defaultdict(int), "plenary": defaultdict(int)}
        self.total = {"committee": 0, "plenary": 0}
        self.vocabulary = set()
        self.vocab_size = 0
        self.default_type = "plenary"

        # interpolation weights (chosen heuristically).
        self.lambda1 = 0.01
        self.lambda2 = 0.02
        self.lambda3 = 0.97

        # additional penalties (heuristic)
        self.frequency_penalty_factor = 2.0
        self.punctuation_tokens = {",", ".", ";", ":", "?", "!", "(", ")", "״", "\"", "׳", "’", "״", "…", "-", "–"}
        self.per_token_penalty = {
            "את": 0.3,
            "לא": 0.3,
            "של": 0.3,
            "אני": 0.3,
            "זה": 0.3,
            "כן": 0.3,
            "הכנסת": 0.3,
            "על": 0.3,
            "חבר": 0.3,
            "ראש": 0.3,
            "הוא": 0.3,
        }

    def get_probabilities(self, protocol_type, w, u, v):
        """
        Compute the interpolated probability of a token w given the two preceding tokens (v, u)
        using Laplace (add-one) smoothing and interpolation.

        Args:
            protocol_type (str): Either 'committee' or 'plenary'.
            w (str): Current token.
            u (str): Previous token.
            v (str): The token before the previous one.

        Returns:
            float: The interpolated probability P(w|v,u).
        """
        # add-one smoothing applied at each level.
        unigram_count = self.unigrams[protocol_type][w]
        unigram_prob = (unigram_count + 1) / (self.total[protocol_type] + self.vocab_size)

        u_count = self.unigrams[protocol_type][u]
        bigram_count = self.bigrams[protocol_type][(u, w)]
        bigram_prob = (bigram_count + 1) / (u_count + self.vocab_size)

        vu_count = self.bigrams[protocol_type][(v, u)]
        trigram_count = self.trigrams[protocol_type][(v, u, w)]
        trigram_prob = (trigram_count + 1) / (vu_count + self.vocab_size)

        # Linear interpolation of unigram, bigram, and trigram probabilities.
        combined_prob = self.lambda1 * unigram_prob + self.lambda2 * bigram_prob + self.lambda3 * trigram_prob
        return combined_prob

    def calculate_prob_of_sentence(self, sentence, protocol_type=None):
        """
        Calculate the log probability of a given sentence under the model.

        The sentence tokens are prefixed with s_0 and s_1 and suffixed with </s> internally.
        Uses the trigram formula with interpolation and add-one smoothing.

        Args:
            sentence (str): The input sentence (space-separated tokens).
            protocol_type (str): 'committee' or 'plenary'. Uses default_type if None.

        Returns:
            float: The log probability of the sentence.
        """
        if protocol_type is None:
            protocol_type = self.default_type
        tokens = ["s_0", "s_1"] + sentence.split() + ["</s>"]
        log_prob = 0.0
        for i in range(2, len(tokens)):
            w = tokens[i]
            u = tokens[i - 1]
            v = tokens[i - 2]
            p = self.get_probabilities(protocol_type, w, u, v)
            log_prob += math.log(p)
        return log_prob

    def generate_next_token(self, prefix, protocol_type=None):
        """
        Predict the next token given a prefix of tokens using the trigram model with heuristics.

        Applies frequency penalties, punctuation penalties, and per-token penalties.

        Args:
            prefix (str): A prefix string of tokens (space-separated).
            protocol_type (str): 'committee' or 'plenary'. Uses default_type if None.

        Returns:
            (str, float): A tuple of (best_token, log_probability_of_best_token).
        """
        if protocol_type is None:
            protocol_type = self.default_type

        prefix_tokens = prefix.split()
        if len(prefix_tokens) == 0:
            v, u = "s_0", "s_1"
        elif len(prefix_tokens) == 1:
            v, u = "s_0", prefix_tokens[-1]
        else:
            v, u = prefix_tokens[-2], prefix_tokens[-1]

        max_score = float("-inf")
        best_token = None

        excluded_tokens = {"s_0", "s_1", "</s>"}

        for token in self.vocabulary:
            if token in excluded_tokens:
                continue

            p = self.get_probabilities(protocol_type, token, u, v)

            # Apply heuristics (not required by the homework, just implemented here):
            token_frequency = self.unigrams[protocol_type][token] / (self.total[protocol_type] + self.vocab_size)
            final_p = p / (1 + self.frequency_penalty_factor * token_frequency)

            if token in self.punctuation_tokens:
                final_p *= 0.05

            if token in self.per_token_penalty:
                final_p *= self.per_token_penalty[token]

            score = math.log(final_p) if final_p > 0 else math.log(1e-10)

            if score > max_score:
                max_score = score
                best_token = token

        return best_token, max_score

def train_trigram_model(model, sentences, protocol_type):
    """
    Train the trigram language model counts from a list of sentences.

    Each sentence is augmented with s_0, s_1 at start and </s> at end.
    Updates unigrams, bigrams, trigrams, vocabulary, and total counts.

    Args:
        model (Trigram_LM): The trigram model instance.
        sentences (list[str]): List of sentences for this protocol type.
        protocol_type (str): 'committee' or 'plenary'.
    """
    for sentence in sentences:
        words = ["s_0", "s_1"] + sentence.split() + ["</s>"]
        model.vocabulary.update(words)
        model.total[protocol_type] += len(words)
        for i, w in enumerate(words):
            model.unigrams[protocol_type][w] += 1
            if i > 0:
                u = words[i-1]
                model.bigrams[protocol_type][(u, w)] += 1
            if i > 1:
                v = words[i-2]
                u = words[i-1]
                model.trigrams[protocol_type][(v, u, w)] += 1

def extract_ngrams(sentences, n):
    """
    Extract n-grams of length n from a list of sentences.

    Args:
        sentences (list[str]): List of sentences.
        n (int): N-gram length.

    Returns:
        Counter: A Counter of n-grams to their frequency.
    """
    ngrams = Counter()
    for sentence in sentences:
        words = sentence.split()
        if len(words) < n:
            continue
        for i in range(len(words)-n+1):
            ngram = tuple(words[i:i+n])
            ngrams[ngram] += 1
    return ngrams

def get_k_n_t_collocations(k, n, t, corpus, type, idf_cache):
    """
    Compute the top-k collocations of length n that appear at least t times,
    scored by either frequency or TF-IDF from a given corpus.

    Args:
        k (int): Number of collocations to return.
        n (int): N-gram length.
        t (int): Minimum frequency threshold.
        corpus (pd.DataFrame): The corpus dataframe with 'protocol_type' and 'sentence_text'.
        type (str): "frequency" or "tfidf" measure.
        idf_cache (dict): A precomputed IDF cache.

    Returns:
        dict: {"committee": dict_of_collocations, "plenary": dict_of_collocations}
              Each dict maps from n-gram (tuple) to score.
    """
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
                if freq < t:
                    continue
                term = " ".join(ngram)
                tf = freq / total_terms if total_terms > 0 else 0
                idf = idf_cache.get(term, 0)
                tf_idf_scores[ngram] = tf * idf
            sorted_tfidf = sorted(tf_idf_scores.items(), key=lambda x: x[1], reverse=True)
            results[protocol_type] = dict(sorted_tfidf[:k])
    return results

def save_collocation_to_file(results, n, type):
    """
    Append collocation results to knesset_collocations.txt.
    Prints the results in a specified format:
    <n>-gram collocations (<Type>):
    Committee corpus:
    <collocations>
    
    Plenary corpus:
    <collocations>
    """
    with open("knesset_collocations.txt", "a", encoding="utf-8") as file:
        header = f"{n}-gram collocations ({type.capitalize()}):\n\n"
        file.write(header)
        for protocol_type, collocations in results.items():
            file.write(f"{protocol_type.capitalize()} corpus:\n")
            if collocations:
                for collocation, value in collocations.items():
                    collocation_text = " ".join(collocation)
                    file.write(f"{collocation_text}: {value:.4f}\n")
            else:
                file.write("No collocations found.\n")
            file.write("\n")
        file.write("\n")

def mask_tokens_in_sentences(sentences, x):
    """
    Mask x% of tokens in each sentence with the special token [*].

    At least one token will be masked if x > 0.

    Args:
        sentences (list[str]): List of sentences.
        x (float): Percentage of tokens to mask.

    Returns:
        (list[str], list[list[int]]): 
            A list of masked sentences and a parallel list of lists of masked token indices.
    """
    masked_sentences = []
    masked_indices = []
    for sentence in sentences:
        tokens = sentence.split()
        num_to_mask = max(1, int(len(tokens)*x/100))
        indices_to_mask = random.sample(range(len(tokens)), num_to_mask)
        masked_sentence = tokens[:]
        for idx in indices_to_mask:
            masked_sentence[idx] = "[*]"
        masked_sentences.append(" ".join(masked_sentence))
        masked_indices.append(indices_to_mask)
    return masked_sentences, masked_indices

def mask_sentences(corpus, num_sentences, mask_percent):
    """
    Select num_sentences random sentences from the committee corpus that have at least 5 tokens.
    Mask mask_percent% of each selected sentence's tokens.

    Args:
        corpus (list[str]): List of sentences from committee corpus.
        num_sentences (int): Number of sentences to choose and mask.
        mask_percent (float): Percentage of tokens to mask.

    Returns:
        (list[str], list[str], list[list[int]]): original_sentences, masked_sentences, masked_indices
    """
    filtered = [s for s in corpus if len(s.split()) >= 5]
    original_sentences = random.sample(filtered, num_sentences)
    masked_sentences, masked_indices = mask_tokens_in_sentences(original_sentences, mask_percent)
    return original_sentences, masked_sentences, masked_indices

def save_sentences_to_file(sentences, file_name):
    """
    Save a list of sentences to a file, one sentence per line.

    Args:
        sentences (list[str]): Sentences to save.
        file_name (str): Output file name.
    """
    with open(file_name, "w", encoding="utf-8") as file:
        for sentence in sentences:
            file.write(sentence + "\n")

def generate_results(original_sentences, masked_sentences, masked_indices, trigram_model_plenary, trigram_model_committee):
    """
    Use the plenary model to guess masked tokens and then print results.

    For each masked sentence:
    - Predict each masked token with the plenary model.
    - Reconstruct the sentence.
    - Compute the log probability of the reconstructed sentence with both plenary and committee models.
    - Print results to sampled_sents_results.txt in the specified format.

    Args:
        original_sentences (list[str]): The original selected sentences.
        masked_sentences (list[str]): The sentences with tokens masked.
        masked_indices (list[list[int]]): The indices of masked tokens in each sentence.
        trigram_model_plenary (Trigram_LM): The plenary trigram model.
        trigram_model_committee (Trigram_LM): The committee trigram model.
    """
    results = []
    trigram_model_committee.default_type = "committee"
    trigram_model_plenary.default_type = "plenary"

    for original, masked, indices in zip(original_sentences, masked_sentences, masked_indices):
        tokens = masked.split()
        guessed_tokens = []
        # Predict each masked token using plenary model
        for index in indices:
            prefix_tokens = tokens[:index]
            prefix_str = " ".join(prefix_tokens) if prefix_tokens else ""
            guessed_token, _ = trigram_model_plenary.generate_next_token(prefix_str)
            guessed_tokens.append(guessed_token)
            tokens[index] = guessed_token

        plenary_sentence = " ".join(tokens)
        plenary_prob_plenary = trigram_model_plenary.calculate_prob_of_sentence(plenary_sentence, "plenary")
        plenary_prob_committee = trigram_model_committee.calculate_prob_of_sentence(plenary_sentence, "committee")

        result = (
            f"original_sentence: {original}\n"
            f"masked_sentence: {masked}\n"
            f"plenary_sentence: {plenary_sentence}\n"
            f"plenary_tokens: {','.join(guessed_tokens)}\n"
            f"probability of plenary sentence in plenary corpus: {plenary_prob_plenary:.2f}\n"
            f"probability of plenary sentence in committee corpus: {plenary_prob_committee:.2f}\n"
        )
        results.append(result)

    with open("sampled_sents_results.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(results))

def calculate_perplexity(masked_sentences, masked_indices, trigram_model):
    """
    Calculate the average perplexity of the plenary model on masked tokens only.

    For each masked token, compute its log probability and aggregate.
    Return the exponential of the negative average log probability.

    Args:
        masked_sentences (list[str]): Masked sentences.
        masked_indices (list[list[int]]): Indices of masked tokens in each sentence.
        trigram_model (Trigram_LM): The plenary trigram model.

    Returns:
        float: The perplexity for masked tokens.
    """
    trigram_model.default_type = "plenary"
    all_log_probs = []
    total_masked = 0
    for sentence, indices in zip(masked_sentences, masked_indices):
        tokens = ["s_0", "s_1"] + sentence.split() + ["</s>"]
        for idx in indices:
            w = tokens[idx+2]
            u = tokens[idx+1]
            v = tokens[idx]
            p = trigram_model.get_probabilities("plenary", w, u, v)
            all_log_probs.append(math.log(p))
            total_masked += 1
    if total_masked == 0:
        return float("inf")
    avg_log_prob = sum(all_log_probs) / total_masked
    perplexity = math.exp(-avg_log_prob)
    return perplexity

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

    def compute_idf(docs):
        """
        Compute IDF values for all terms appearing in the given docs (list of sentences).

        IDF(t) = log(|D| / (|{d in D: t in d}|))

        Args:
            docs (list[str]): The documents (sentences).

        Returns:
            dict: Maps term (string) to its IDF value.
        """
        total_docs = len(docs)
        doc_frequency = defaultdict(int)
        for doc in docs:
            words = doc.split()
            # Generate all n-grams from length 1 to 4 for IDF calculation
            unique_terms = set(" ".join(words[i:i+n]) for n in range(1,5) for i in range(len(words)-n+1))
            for term in unique_terms:
                if term.strip() not in {",", ".", ";", ":", "–"}:
                    doc_frequency[term] += 1
        return {term: math.log(total_docs/(1+freq)) for term, freq in doc_frequency.items()}

    idf_cache = compute_idf(corpus["sentence_text"].tolist())
    k = 10
    lengths = [2, 3, 4]
    t = 5

    # Clear the collocations file
    with open("knesset_collocations.txt", "w", encoding="utf-8") as f:
        f.write("")

    # Extract and save collocations for each n and type
    for n in lengths:
        for ctype in ["frequency", "tfidf"]:
            start_time = time.time()
            print(f"Processing {n}-gram collocations, type: {ctype}...")
            collocations = get_k_n_t_collocations(k=k, n=n, t=t, corpus=corpus, type=ctype, idf_cache=idf_cache)
            save_collocation_to_file(collocations, n, ctype)
            print(f"Completed {n}-gram collocations for type: {ctype} in {time.time() - start_time:.2f} seconds.")
    print("Collocations have been saved to 'knesset_collocations.txt'.")

    # Split corpus by protocol type
    committee_sentences = corpus[corpus["protocol_type"] == "committee"]["sentence_text"].tolist()
    plenary_sentences = corpus[corpus["protocol_type"] == "plenary"]["sentence_text"].tolist()

    # Initialize models
    trigram_model_committee = Trigram_LM()
    trigram_model_plenary = Trigram_LM()

    # Train models
    train_trigram_model(trigram_model_committee, committee_sentences, "committee")
    train_trigram_model(trigram_model_plenary, plenary_sentences, "plenary")

    # Update vocab sizes
    trigram_model_committee.vocab_size = len(trigram_model_committee.vocabulary)
    trigram_model_plenary.vocab_size = len(trigram_model_plenary.vocabulary)

    # Mask sentences from committee corpus
    original_sentences, masked_sentences, masked_indices = mask_sentences(committee_sentences, 10, 10)
    save_sentences_to_file(original_sentences, "original_sampled_sents.txt")
    save_sentences_to_file(masked_sentences, "masked_sampled_sents.txt")

    # Generate results for masked tokens
    generate_results(original_sentences, masked_sentences, masked_indices, trigram_model_plenary, trigram_model_committee)
    print("Results saved to 'sampled_sents_results.txt'.")

    # Calculate perplexity
    avg_perplexity = calculate_perplexity(masked_sentences, masked_indices, trigram_model_plenary)
    with open("perplexity_result.txt", "w", encoding="utf-8") as f:
        f.write(f"{avg_perplexity:.2f}\n")

    print(f"Average perplexity for masked tokens: {avg_perplexity:.2f}")
