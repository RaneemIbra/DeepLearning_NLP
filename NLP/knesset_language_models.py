import json
from collections import defaultdict, Counter
import math

class Trigram_LM:
    def __init__(self, vocab_size):
        self.models = {
            "committee": defaultdict(int),
            "plenary": defaultdict(int)
        }
        self.bigrams = {
            "committee": defaultdict(int),
            "plenary": defaultdict(int)
        }
        self.unigrams = {
            "committee": defaultdict(int),
            "plenary": defaultdict(int)
        }
        self.total = {
            "committee": 0,
            "plenary": 0
        }
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
            trigrams = (tokens[i-2], tokens[i-1], tokens[i])
            bigrams = (tokens[i-1], tokens[i])
            unigrams = tokens[i]

            unigrams_count = self.unigrams[protocol_type][unigrams]
            bigrams_count = self.bigrams[protocol_type][bigrams]
            trigrams_count = self.models[protocol_type][trigrams]
            total_count = self.total[protocol_type]

            if(total_count > 0):
                unigram_prob = (unigrams_count + self.laplace_constant) / (total_count + self.vocab_size)
            else:
                unigram_prob = 0

            if(self.unigrams[protocol_type][tokens[i-1]] > 0):
                bigrams_prob = (bigrams_count + self.laplace_constant) / (self.unigrams[protocol_type][tokens[i-1]] + self.vocab_size)
            else:
                bigrams_prob = 0

            if(self.bigrams[protocol_type][(tokens[i-2], tokens[i-1])] > 0):
                trigrams_prob = (trigrams_count + self.laplace_constant) / (self.bigrams[protocol_type][(tokens[i-2], tokens[i-1])] + self.vocab_size)
            else:
                trigrams_prob = 0

            # P(Wk| Wk-2,Wk-1) = alpha * P_trigram + beta * P_bigram + gamma * P_unigram
            # give more weight to the last word no?
            probability = trigrams_prob * 0.1 + bigrams_prob * 0.3 + unigram_prob * 0.6
            if(probability > 0):
                log_prob += math.log(probability)
            else:
                log_prob += float("-inf")
        return log_prob
    
    def generate_next_token(self, space_separated_tokens):
        tokens = space_separated_tokens.split()
        if(len(tokens) < 2):
            tokens = ["<s>"] * (2 - len(tokens)) + tokens
        last_token = tokens[-1]
        second_last_token = tokens[-2]
        generated_token = None
        max_probability = float("-inf")
        tokens_string = last_token.join(second_last_token)
        for i in self.vocabulary:
            # should we keep in mind the protocol type?
            probability = self.calculate_prob_of_sentence(i, tokens_string)
            if(probability > max_probability):
                max_probability = probability
                generated_token = i
        return generated_token, max_probability
