import os
import random
from collections import Counter, defaultdict
import math
import pandas as pd
import sys
import json

class Trigram_LM:
    def __init__(self):
        self.lambda_trigram = 0.7
        self.lambda_bigram = 0.01
        self.lambda_unigram = 0.29
        self.total_tokens = 0
        self.vocab_size = 0
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()

    # a function to calculate the smoothed probability of an ngram using laplace smoothing
    def compute_smoothed_probability(self, ngram, ngram_type):
        if ngram_type == 1:
            count = self.unigram_counts[ngram[0]]
            total = self.total_tokens
        elif ngram_type == 2:
            count = self.bigram_counts[ngram]
            total = self.unigram_counts[ngram[0]] if ngram[0] in self.unigram_counts else 0
        elif ngram_type == 3:
            count = self.trigram_counts[ngram]
            bigram = (ngram[0], ngram[1])
            total = self.bigram_counts[bigram] if bigram in self.bigram_counts else 0
        else:
            raise ValueError("Invalid ngram type")

        return (count + 1) / (total + self.vocab_size)


    # the following function calculates the probability of a sentence
    def calculate_prob_of_sentence(self, sentence):
        tokens = self.prepare_tokens(sentence)
        total_log_prob = sum(self.calculate_token_log_prob(tokens, idx) for idx in range(2, len(tokens)))
        return total_log_prob

    # a function to prepare each sentence for a trigram model by adding dummy tokens
    def prepare_tokens(self, sentence):
        if isinstance(sentence, str):
            sentence = sentence.split()
        return ['s_1', 's_1'] + sentence + ['s_2']

    # a function to calculate the log probability for a token at a specific index
    def calculate_token_log_prob(self, tokens, idx):
        unigram = (tokens[idx],)
        bigram = (tokens[idx - 1], tokens[idx])
        trigram = (tokens[idx - 2], tokens[idx - 1], tokens[idx])

        unigram_prob = self.compute_smoothed_probability(unigram, 1)
        bigram_prob = self.compute_smoothed_probability(bigram, 2)
        trigram_prob = self.compute_smoothed_probability(trigram, 3)

        interpolated_prob = (
            self.lambda_unigram * unigram_prob +
            self.lambda_bigram * bigram_prob +
            self.lambda_trigram * trigram_prob
        )
        return math.log2(interpolated_prob)

    # a function to generate the next token in a given context based on probabilities
    def generate_next_token(self, context):
        tokens = self.prepare_context_tokens(context)
        prev_2, prev_1 = tokens[-2], tokens[-1]
        best_token, best_log_prob = self.find_best_candidate(prev_2, prev_1)
        return best_token, best_log_prob

    # a function to prepare each sentence if there is a need to add dummy tokens
    def prepare_context_tokens(self, context):
        tokens = context.split() if isinstance(context, str) else context
        return ['s_1', 's_1'] + tokens[-2:] if len(tokens) < 2 else tokens

    # a function that computes the probability of the ngrams and then calculates the combined probability
    def compute_combined_probability(self, candidate, prev_2, prev_1):
        unigram_prob = self.compute_smoothed_probability((candidate,), 1)
        bigram_prob = self.compute_smoothed_probability((prev_1, candidate), 2)
        trigram_prob = self.compute_smoothed_probability((prev_2, prev_1, candidate), 3)

        return (
            self.lambda_unigram * unigram_prob +
            self.lambda_bigram * bigram_prob +
            self.lambda_trigram * trigram_prob
        )

    # a function that will find the token with the highest probability and returns it with the log of its probability
    def find_best_candidate(self, prev_2, prev_1):
        skip_tokens = {'s_1', 's_2'}
        best_token = None
        best_log_prob = float('-inf')

        for candidate in self.unigram_counts.keys():
            if candidate in skip_tokens:
                continue

            combined_probability = self.compute_combined_probability(candidate, prev_2, prev_1)
            log_probability = math.log2(combined_probability)

            if log_probability > best_log_prob:
                best_log_prob = log_probability
                best_token = candidate

        return best_token, best_log_prob

    # a function to train / fit the model on the sentences
    def fit_model_to_sentences(self, input_sentences):
        for sentence in input_sentences:
            tokens = self.prepare_tokens(sentence)
            self.update_ngram_counts(tokens)
        self.vocab_size = len(self.unigram_counts)
    
    # a function to update the count of each ngram
    def update_ngram_counts(self, tokens):
        self.total_tokens += len(tokens)
        for i, token in enumerate(tokens):
            self.unigram_counts[token] += 1
            if i >= 1:
                self.bigram_counts[(tokens[i-1], token)] += 1
            if i >= 2:
                self.trigram_counts[(tokens[i-2], tokens[i-1], token)] += 1

    # a function that gets the top k with length n that appears the most in the corpus according to a certain metric
    def get_k_n_t_collocations(self, data, top_k, ngram_size, min_threshold, scoring_metric):
        num_documents = data['protocol_name'].nunique()
        global_ngram_counts = Counter()
        document_ngram_counts = Counter()
        protocol_ngrams = {}

        for protocol_name, sentences in data.groupby('protocol_name')['sentence_text']:
            ngram_counts = self.calculate_ngram_frequencies(sentences, ngram_size)
            global_ngram_counts.update(ngram_counts)

            for ngram in ngram_counts.keys():
                document_ngram_counts[ngram] += 1

            if scoring_metric == 'tfidf':
                protocol_ngrams[protocol_name] = ngram_counts

        valid_ngrams = {
            ngram for ngram, count in global_ngram_counts.items()
            if count >= min_threshold and 's_1' not in ngram and 's_2' not in ngram
        }

        if scoring_metric == 'frequency':
            return sorted(valid_ngrams, key=lambda ngram: global_ngram_counts[ngram], reverse=True)[:top_k]

        elif scoring_metric == 'tfidf':
            return self.get_top_tfidf_ngrams(protocol_ngrams, valid_ngrams, document_ngram_counts, num_documents, top_k)
        
        else:
            raise ValueError("wrong metric used")

    # a function to get the top tfidf scoring ngrams
    def get_top_tfidf_ngrams(self, protocol_ngrams, valid_ngrams, document_ngram_counts, num_documents, top_k):
        tfidf_scores = defaultdict(list)

        for _, ngram_counts in protocol_ngrams.items():
            for ngram in valid_ngrams & set(ngram_counts.keys()):
                score = self.compute_tfidf(ngram, ngram_counts, document_ngram_counts, num_documents)
                tfidf_scores[ngram].append(score)

        average_tfidf_scores = {ngram: sum(scores) / len(scores) for ngram, scores in tfidf_scores.items()}
        ranked_ngrams = sorted(average_tfidf_scores.items(), key=lambda item: item[1], reverse=True)

        return [ngram for ngram, _ in ranked_ngrams[:top_k]]

    # a function to calculate the frquency of the ngrams
    def calculate_ngram_frequencies(self, text_data, n):
        ngram_frequencies = Counter()
        for sentence in text_data:
            tokens = self.tokenize(sentence, n)
            ngram_frequencies.update(self.extract_ngrams(tokens, n))

        return ngram_frequencies

    # a function to calculate the tfidf score according to the formula that was taught
    def compute_tfidf(self, ngram, ngram_frequencies, document_frequencies, num_documents):
        term_frequency = ngram_frequencies[ngram] / sum(ngram_frequencies.values())
        inverse_document_frequency = math.log((num_documents + 1) / (document_frequencies[ngram] + 1))
        return term_frequency * inverse_document_frequency

    # a function to tokenize and prepare the sentence
    def tokenize(self, sentence, n):
        tokens = sentence.split() if isinstance(sentence, str) else sentence
        return ['s_1'] * (n - 1) + tokens + ['s_2']

    # a function that will extract the ngrams
    def extract_ngrams(self, tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    # a function to save the output for the collocations in a file
    def save_collocations(self, output_file, committee_data, plenary_data):
        with open(output_file, 'w', encoding='utf-8') as file:
            for ngram_size in [2, 3, 4]:
                ngram_label = {2: "Two", 3: "Three", 4: "Four"}.get(ngram_size, "Unknown")
                file.write(f"{ngram_label}-gram collocations:\n")

                for metric in ["frequency", "tfidf"]:
                    file.write(f"{metric.capitalize()}:\n")
                    for corpus_label, corpus_data in [("Committee corpus", committee_data), 
                                                    ("Plenary corpus", plenary_data)]:
                        file.write(f"{corpus_label}:\n")

                        collocations = self.get_k_n_t_collocations(corpus_data, top_k=10, 
                                                                ngram_size=ngram_size, min_threshold=5, 
                                                                scoring_metric=metric)

                        for collocation in collocations:
                            file.write(" ".join(collocation) + "\n")
    
    # a function to mask a percentage of random tokens
    def mask_tokens_in_sentences(self, sentences, mask_percentage):
        masked_sentences = []
        for sentence in sentences:
            token_list = sentence.split()
            tokens_to_mask = max(1, int(len(token_list) * mask_percentage / 100))
            mask_positions = random.sample(range(len(token_list)), tokens_to_mask)
            for position in mask_positions:
                token_list[position] = "[*]"

            masked_sentences.append(" ".join(token_list))

        return masked_sentences

# a function to calculate the perplexity of the model
def compute_masked_perplexity(model, original_sentence, masked_sentence):
    # an inner function that will be only used in the outer function to get the ngrams probabilities
    def get_ngram_probabilities(model, tokens, index):
        unigram = (tokens[index],)
        bigram = (tokens[index - 1], tokens[index])
        trigram = (tokens[index - 2], tokens[index - 1], tokens[index])

        p_unigram = model.compute_smoothed_probability(unigram, 1)
        p_bigram = model.compute_smoothed_probability(bigram, 2)
        p_trigram = model.compute_smoothed_probability(trigram, 3)

        return (model.lambda_unigram * p_unigram +
                model.lambda_bigram * p_bigram +
                model.lambda_trigram * p_trigram)

    original_tokens = ['s_1', 's_1'] + original_sentence.split() + ['s_2']
    masked_tokens = ['s_1', 's_1'] + masked_sentence.split() + ['s_2']
    masked_indices = [idx for idx, token in enumerate(masked_tokens) if token == '[*]']

    probabilities = [
        get_ngram_probabilities(model, original_tokens, idx)
        for idx in masked_indices if idx >= 2
    ]

    if not probabilities:
        return float('inf')

    average_log_prob = sum(math.log2(p) for p in probabilities) / len(probabilities)
    return 2 ** (-average_log_prob)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(1)

    corpus_file = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    records = []
    try:
        with open(corpus_file, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line.strip())
                records.append(
                    (
                        entry.get('protocol_name', ''),
                        entry.get('protocol_type', '').lower(),
                        entry.get('sentence_text', '')
                    )
                )
    except FileNotFoundError:
        sys.exit(1)

    df = pd.DataFrame(records, columns=['protocol_name', 'protocol_type', 'sentence_text'])

    committee_data = df[df['protocol_type'] == 'committee']
    plenary_data = df[df['protocol_type'] == 'plenary']

    committee_sentences = committee_data['sentence_text'].tolist()
    plenary_sentences = plenary_data['sentence_text'].tolist()

    committee_model = Trigram_LM()
    plenary_model = Trigram_LM()

    committee_model.fit_model_to_sentences(committee_sentences)
    plenary_model.fit_model_to_sentences(plenary_sentences)

    collocation_file = os.path.join(output_dir, "knesset_collocations.txt")
    committee_model.save_collocations(collocation_file, committee_data, plenary_data)
    filtered_sentences = [s for s in committee_sentences if len(s.split()) >= 5]
    sampled_sentences = random.sample(filtered_sentences, min(10, len(filtered_sentences)))
    masked_sentences = plenary_model.mask_tokens_in_sentences(sampled_sentences, 10)

    original_file = os.path.join(output_dir, "original_sampled_sents.txt")
    masked_file = os.path.join(output_dir, "masked_sampled_sents.txt")

    with open(original_file, 'w', encoding='utf-8') as orig_f:
        orig_f.writelines(s + '\n' for s in sampled_sentences)

    with open(masked_file, 'w', encoding='utf-8') as mask_f:
        mask_f.writelines(s + '\n' for s in masked_sentences)

    results_file = os.path.join(output_dir, "sampled_sents_results.txt")
    results = []
    for original, masked in zip(sampled_sentences, masked_sentences):
        current_masked = masked
        predicted_tokens = []
        while '[*]' in current_masked:
            prefix = current_masked.split('[*]', 1)[0]
            token, _ = plenary_model.generate_next_token(prefix)
            predicted_tokens.append(token)
            current_masked = current_masked.replace('[*]', token, 1)

        final_sentence = current_masked
        plenary_prob = plenary_model.calculate_prob_of_sentence(final_sentence)
        committee_prob = committee_model.calculate_prob_of_sentence(final_sentence)
        results.append((original, masked, final_sentence, predicted_tokens, plenary_prob, committee_prob))

    with open(results_file, 'w', encoding='utf-8') as res_f:
        for original, masked, final, tokens, p_plen, p_comm in results:
            res_f.write(f"original_sentence: {original}\n")
            res_f.write(f"masked_sentence: {masked}\n")
            res_f.write(f"plenary_sentence: {final}\n")
            res_f.write(f"plenary_tokens: {', '.join(tokens)}\n")
            res_f.write(f"probability of plenary sentence in plenary corpus: {p_plen:.2f}\n")
            res_f.write(f"probability of plenary sentence in committee corpus: {p_comm:.2f}\n")

    perplexity_file = os.path.join(output_dir, "perplexity_result.txt")
    total_perplexity = sum(
        compute_masked_perplexity(plenary_model, o, m) for o, m in zip(sampled_sentences, masked_sentences)
    )
    average_perplexity = total_perplexity / len(sampled_sentences)
    with open(perplexity_file, 'w', encoding='utf-8') as pp_f:
        pp_f.write(f"{average_perplexity:.2f}\n")
