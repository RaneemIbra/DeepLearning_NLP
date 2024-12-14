import json
from collections import defaultdict, Counter
import math
import pandas as pd
import time
import random

class Trigram_LM:
    """
    A Trigram Language Model that uses interpolation and add-one smoothing.

    This model learns from two types of protocols:
    - 'committee'
    - 'plenary'

    It counts how often single words (unigrams), pairs of words (bigrams),
    and triplets of words (trigrams) appear.

    It then uses these counts to guess the probability of seeing a word after some previous words.

    We also add some extra penalties to make certain words or punctuation less likely.
    (These penalties are not required by the original instructions, but are included here.)
    """

    def __init__(self):
        # Store counts of unigrams, bigrams, and trigrams for both committee and plenary.
        self.trigrams = {"committee": defaultdict(int), "plenary": defaultdict(int)}
        self.bigrams = {"committee": defaultdict(int), "plenary": defaultdict(int)}
        self.unigrams = {"committee": defaultdict(int), "plenary": defaultdict(int)}

        # Track total number of tokens for each type (committee or plenary).
        self.total = {"committee": 0, "plenary": 0}

        # A set of all different words seen.
        self.vocabulary = set()

        # The size of the vocabulary will be set later, after we count all words.
        self.vocab_size = 0

        # Default protocol type to use if none is given: 'plenary'.
        self.default_type = "plenary"

        # These are the interpolation weights for unigrams, bigrams, and trigrams.
        # They control how we mix their probabilities.
        self.lambda1 = 0.01
        self.lambda2 = 0.02
        self.lambda3 = 0.97

        # We add penalties to common words and punctuation to avoid them being chosen too often.
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
            "גם": 0.3,
        }

    def get_probabilities(self, protocol_type, w, u, v):
        """
        Compute the probability of seeing word w after words v and u.

        We use add-one smoothing: we add 1 to counts to avoid zero probabilities.
        We mix unigram, bigram, and trigram probabilities using the lambda weights.

        Returns a number between 0 and 1.
        """
        # Counts with add-one smoothing
        unigram_count = self.unigrams[protocol_type][w]
        unigram_prob = (unigram_count + 1) / (self.total[protocol_type] + self.vocab_size)

        u_count = self.unigrams[protocol_type][u]
        bigram_count = self.bigrams[protocol_type][(u, w)]
        bigram_prob = (bigram_count + 1) / (u_count + self.vocab_size)

        vu_count = self.bigrams[protocol_type][(v, u)]
        trigram_count = self.trigrams[protocol_type][(v, u, w)]
        trigram_prob = (trigram_count + 1) / (vu_count + self.vocab_size)

        # Combine using interpolation weights
        combined_prob = self.lambda1 * unigram_prob + self.lambda2 * bigram_prob + self.lambda3 * trigram_prob
        return combined_prob

    def calculate_prob_of_sentence(self, sentence, protocol_type=None):
        """
        Calculate the log probability of a full sentence.

        We add special start tokens s_0, s_1 at the beginning and </s> at the end.
        For each word in the sentence, we get the probability and then sum up the logs.
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
        Given a prefix of words, predict the next word.

        We check every word in the vocabulary and pick the one with the highest adjusted probability.

        We apply extra penalties to frequent words, punctuation, and certain tokens.
        This is a heuristic and not part of the basic trigram model.
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

            # Apply heuristic penalties.
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
    Train the model with sentences of a given protocol type.

    We add s_0, s_1 at the start and </s> at the end of each sentence.
    We update the counts of unigrams, bigrams, trigrams, and the total token count.
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
    Extract n-grams from sentences.

    For each sentence, we take every sequence of n words and count them.
    Returns a Counter with n-gram frequencies.
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
    Find the top-k n-grams (collocations) that appear at least t times in the corpus.

    If type = "frequency", we pick the most frequent n-grams.
    If type = "tfidf", we pick by highest TF-IDF value.

    Returns a dictionary with separate results for 'committee' and 'plenary'.
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
    Save collocation results to 'knesset_collocations.txt'.

    The format is:
    <n>-gram collocations (Type):

    Committee corpus:
    <collocations>

    Plenary corpus:
    <collocations>
    """
    ngram_string = None
    if(n==2):
        ngram_string = "Two"
    elif(n==3):
        ngram_string = "Three"
    elif(n==4):
        ngram_string = "Four"
    with open("knesset_collocations.txt", "a", encoding="utf-8") as file:
        header = f"{ngram_string}-gram collocations:\n{type.capitalize()}:\n"
        file.write(header)
        for protocol_type, collocations in results.items():
            file.write(f"{protocol_type.capitalize()} corpus:\n")
            if collocations:
                for collocation, value in collocations.items():
                    collocation_text = " ".join(collocation)
                    file.write(f"{collocation_text}: {value:.4f}\n")
            else:
                file.write("No collocations found.\n")

def mask_tokens_in_sentences(sentences, x):
    """
    Mask x% of the tokens in each sentence with '[*]'.

    If x% of tokens is less than 1 token, we still mask at least one token.
    Returns the masked sentences and the indices of masked tokens.
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
    Choose num_sentences random sentences (each with at least 5 tokens) from the corpus.
    Mask mask_percent% of their tokens.

    Returns the original sentences, the masked versions, and the indices of masked tokens.
    """
    filtered = [s for s in corpus if len(s.split()) >= 5]
    original_sentences = random.sample(filtered, num_sentences)
    masked_sentences, masked_indices = mask_tokens_in_sentences(original_sentences, mask_percent)
    return original_sentences, masked_sentences, masked_indices

def save_sentences_to_file(sentences, file_name):
    """
    Save a list of sentences to a file, one sentence per line.
    """
    with open(file_name, "w", encoding="utf-8") as file:
        for sentence in sentences:
            file.write(sentence + "\n")

def generate_results(original_sentences, masked_sentences, masked_indices, trigram_model_plenary, trigram_model_committee):
    """
    For each masked sentence:
    1. Use the plenary model to guess each masked token.
    2. Rebuild the full sentence with the guessed tokens.
    3. Calculate the probability of this rebuilt sentence under both plenary and committee models.
    4. Write these details to 'sampled_sents_results.txt'.

    The output includes:
    - original sentence
    - masked sentence
    - the sentence after filling in the guessed tokens (plenary_sentence)
    - the guessed tokens (plenary_tokens)
    - the probabilities in plenary and committee models
    """
    results = []
    trigram_model_committee.default_type = "committee"
    trigram_model_plenary.default_type = "plenary"

    for original, masked, indices in zip(original_sentences, masked_sentences, masked_indices):
        tokens = masked.split()
        guessed_tokens = []
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
    Calculate the perplexity on only the masked tokens after they are guessed.

    Perplexity measures how well the model predicts these tokens.
    A lower perplexity means the model predictions fit better.

    Steps:
    - For each masked token, get its probability from the model.
    - Take the log of these probabilities, average them, and then use exp(-average_log_prob).

    Returns a single perplexity value.
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
        Compute IDF (Inverse Document Frequency) values for all terms.

        IDF(t) = log(total_docs / (1 + number_of_docs_with_t))

        We consider all n-grams of length 1 to 4 as terms.
        """
        total_docs = len(docs)
        doc_frequency = defaultdict(int)
        for doc in docs:
            words = doc.split()
            # Generate n-grams of length 1 to 4 for IDF
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

    # Extract and save collocations for each n and type (frequency, tfidf)
    for n in lengths:
        for ctype in ["frequency", "tfidf"]:
            start_time = time.time()
            print(f"Processing {n}-gram collocations, type: {ctype}...")
            collocations = get_k_n_t_collocations(k=k, n=n, t=t, corpus=corpus, type=ctype, idf_cache=idf_cache)
            save_collocation_to_file(collocations, n, ctype)
            print(f"Completed {n}-gram collocations for type: {ctype} in {time.time() - start_time:.2f} seconds.")
    print("Collocations have been saved to 'knesset_collocations.txt'.")

    # Get committee and plenary sentences
    committee_sentences = corpus[corpus["protocol_type"] == "committee"]["sentence_text"].tolist()
    plenary_sentences = corpus[corpus["protocol_type"] == "plenary"]["sentence_text"].tolist()

    # Create models
    trigram_model_committee = Trigram_LM()
    trigram_model_plenary = Trigram_LM()

    # Train models
    train_trigram_model(trigram_model_committee, committee_sentences, "committee")
    train_trigram_model(trigram_model_plenary, plenary_sentences, "plenary")

    # Set vocabulary size
    trigram_model_committee.vocab_size = len(trigram_model_committee.vocabulary)
    trigram_model_plenary.vocab_size = len(trigram_model_plenary.vocabulary)

    # Mask sentences from committee corpus
    original_sentences, masked_sentences, masked_indices = mask_sentences(committee_sentences, 10, 10)
    save_sentences_to_file(original_sentences, "original_sampled_sents.txt")
    save_sentences_to_file(masked_sentences, "masked_sampled_sents.txt")

    # Predict masked tokens and save results
    generate_results(original_sentences, masked_sentences, masked_indices, trigram_model_plenary, trigram_model_committee)
    print("Results saved to 'sampled_sents_results.txt'.")

    # Calculate perplexity
    avg_perplexity = calculate_perplexity(masked_sentences, masked_indices, trigram_model_plenary)
    with open("perplexity_result.txt", "w", encoding="utf-8") as f:
        f.write(f"{avg_perplexity:.2f}\n")

    print(f"Average perplexity for masked tokens: {avg_perplexity:.2f}")
