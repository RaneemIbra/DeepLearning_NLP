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

    def compute_smoothed_probability(self, ngram, ngram_type):
        if ngram_type == 1:
            word = ngram[0]
            count = self.unigram_counts[word]
            total = self.total_tokens
        elif ngram_type == 2:
            first_word, second_word = ngram
            count = self.bigram_counts[(first_word, second_word)]
            total = self.unigram_counts[first_word]
        elif ngram_type == 3:
            first_word, second_word, third_word = ngram
            count = self.trigram_counts[(first_word, second_word, third_word)]
            total = self.bigram_counts[(first_word, second_word)]
        else:
            raise ValueError("Error")

        smoothed_probability = (count + 1) / (total + self.vocab_size)
        return smoothed_probability

    def calculate_prob_of_sentence(self, sentence):
        token_list = ['s_1', 's_1'] + (sentence.split() if isinstance(sentence, str) else sentence) + ['s_2']
        total_log_probability = 0.0
        for idx in range(2, len(token_list)):
            current_trigram = (token_list[idx - 2], token_list[idx - 1], token_list[idx])
            current_bigram = (token_list[idx - 1], token_list[idx])
            current_unigram = (token_list[idx],)
            
            unigram_prob = self.compute_smoothed_probability(current_unigram, 1)
            bigram_prob = self.compute_smoothed_probability(current_bigram, 2)
            trigram_prob = self.compute_smoothed_probability(current_trigram, 3)

            interpolated_prob = (
                self.lambda_unigram * unigram_prob +
                self.lambda_bigram * bigram_prob +
                self.lambda_trigram * trigram_prob
            )
            total_log_probability += math.log2(interpolated_prob)

        return total_log_probability

    def generate_next_token(self, context):
        context_tokens = context.split() if isinstance(context, str) else context
        if len(context_tokens) < 2:
            context_tokens = ['s_1', 's_1'] + context_tokens

        prev_2, prev_1 = context_tokens[-2], context_tokens[-1]
        highest_probability = float('-inf')
        most_probable_token = None
        skip_tokens = {'s_1', 's_2'}
        for candidate in self.unigram_counts.keys():
            if candidate in skip_tokens:
                continue

            unigram_probability = self.compute_smoothed_probability((candidate,), 1)
            bigram_probability = self.compute_smoothed_probability((prev_1, candidate), 2)
            trigram_probability = self.compute_smoothed_probability((prev_2, prev_1, candidate), 3)

            combined_probability = (
                self.lambda_unigram * unigram_probability +
                self.lambda_bigram * bigram_probability +
                self.lambda_trigram * trigram_probability
            )

            log_probability = math.log2(combined_probability)

            if log_probability > highest_probability:
                highest_probability = log_probability
                most_probable_token = candidate

        return most_probable_token, highest_probability

    def fit_model_to_sentences(self, input_sentences):
        for sentence in input_sentences:
            words = sentence.split() if isinstance(sentence, str) else sentence
            words = ['s_1', 's_1'] + words + ['s_2']
            self.total_tokens += len(words)
            for i in range(len(words)):
                self.unigram_counts[words[i]] += 1
                if i > 0:
                    self.bigram_counts[(words[i-1], words[i])] += 1
                if i > 1:
                    self.trigram_counts[(words[i-2], words[i-1], words[i])] += 1

        self.vocab_size = len(self.unigram_counts)

    def get_k_n_t_collocations(self, data, top_k, ngram_size, min_threshold, scoring_metric):
        num_documents = data['protocol_name'].nunique()
        global_ngram_counts = Counter()
        document_ngram_counts = Counter()
        protocol_sentences = data.groupby('protocol_name')['sentence_text'].apply(list)
        protocol_ngram_frequencies = {}
        for protocol_name, sentences in protocol_sentences.items():
            ngram_counts = self.calculate_ngram_frequencies(sentences, ngram_size)
            global_ngram_counts.update(ngram_counts)
            for ngram in ngram_counts:
                document_ngram_counts[ngram] += 1

            if scoring_metric == 'tfidf':
                protocol_ngram_frequencies[protocol_name] = ngram_counts

        valid_ngrams = {
            ngram for ngram, count in global_ngram_counts.items()
            if count >= min_threshold and 's_1' not in ngram and 's_2' not in ngram
        }

        if scoring_metric == 'frequency':
            ranked_ngrams = sorted(valid_ngrams, key=lambda ngram: global_ngram_counts[ngram], reverse=True)
            return ranked_ngrams[:top_k]

        elif scoring_metric == 'tfidf':
            tfidf_scores = defaultdict(list)
            for protocol_name, ngram_counts in protocol_ngram_frequencies.items():
                protocol_valid_ngrams = {ngram for ngram in ngram_counts if ngram in valid_ngrams}
                for ngram in protocol_valid_ngrams:
                    score = self.compute_tfidf(ngram, ngram_counts, document_ngram_counts, num_documents)
                    tfidf_scores[ngram].append(score)

            average_tfidf_scores = {ngram: sum(scores) / len(scores) for ngram, scores in tfidf_scores.items()}
            ranked_ngrams = sorted(average_tfidf_scores.items(), key=lambda item: item[1], reverse=True)
            return [ngram for ngram, _ in ranked_ngrams[:top_k]]

    def calculate_ngram_frequencies(self, text_data, n):
        ngram_frequencies = Counter()

        for entry in text_data:
            tokens = entry.split() if isinstance(entry, str) else entry
            padded_tokens = ['s_1'] * (n - 1) + tokens + ['s_2']

            for start_idx in range(len(padded_tokens) - n + 1):
                ngram = tuple(padded_tokens[start_idx : start_idx + n])
                ngram_frequencies[ngram] += 1

        return ngram_frequencies

    def compute_tfidf(self, ngram, ngram_frequencies, document_frequencies, num_documents):
        term_frequency = ngram_frequencies[ngram] / sum(ngram_frequencies.values())
        inverse_document_frequency = math.log((num_documents + 1) / (document_frequencies[ngram] + 1))
        return term_frequency * inverse_document_frequency
    
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

def compute_masked_perplexity(model, original_sentence, masked_sentence):
    original_tokens = ['s_1', 's_1'] + original_sentence.split() + ['s_2']
    masked_tokens = ['s_1', 's_1'] + masked_sentence.split() + ['s_2']
    masked_indices = [idx for idx, token in enumerate(masked_tokens) if token == '[*]']
    probabilities = []
    for index in masked_indices:
        if index < 2:
            continue

        unigram = (original_tokens[index],)
        bigram = (original_tokens[index - 1], original_tokens[index])
        trigram = (original_tokens[index - 2], original_tokens[index - 1], original_tokens[index])

        unigram_prob = model.compute_smoothed_probability(unigram, 1)
        bigram_prob = model.compute_smoothed_probability(bigram, 2)
        trigram_prob = model.compute_smoothed_probability(trigram, 3)

        combined_prob = (
            model.lambda_unigram * unigram_prob +
            model.lambda_bigram * bigram_prob +
            model.lambda_trigram * trigram_prob
        )

        probabilities.append(combined_prob)

    if not probabilities:
        return float('inf')

    average_log_prob = sum(math.log2(prob) for prob in probabilities) / len(probabilities)
    perplexity = 2 ** (-average_log_prob)
    return perplexity

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
