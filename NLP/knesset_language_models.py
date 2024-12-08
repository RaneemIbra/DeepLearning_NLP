import json
from collections import defaultdict, Counter
import math
import pandas as pd
import time

# define a class just as proposed in the first section of the homework
class Trigram_LM:
    # initialize an instance of the class with a vocabulary size
    def __init__(self, vocab_size):
        # define the models data structues which will store the frequencies 
        self.models = {"committee": defaultdict(int), "plenary": defaultdict(int)}
        self.bigrams = {"committee": defaultdict(int), "plenary": defaultdict(int)}
        self.unigrams = {"committee": defaultdict(int), "plenary": defaultdict(int)}
        # track the number of tokens
        self.total = {"committee": 0, "plenary": 0}
        # stores a default value type for the protocol type just for testing
        self.default_type = "committee"
        # stores a laplace constant
        self.laplace_constant = 1
        # stores the vocab size
        self.vocab_size = vocab_size
        # this is a set that contains all the tokens
        self.vocabulary = set()
    # a function to calculate the log probability of a sentence based on the trigrams model
    def calculate_prob_of_sentence(self, space_separated_tokens):
        # split the tokens to a list
        tokens = space_separated_tokens.split()
        # append the initial and the last token to provide context for the beginning and the ending of the sentence
        tokens = ["<s>", "<s>"] + tokens + ["</s>"]
        # initialize the log probability
        log_prob = 0.0
        protocol_type = self.default_type
        # iterate over the tokens starting from the third one to ensure a valid trigram
        for i in range(2, len(tokens)):
            # define the ngrams
            trigrams = (tokens[i - 2], tokens[i - 1], tokens[i])
            bigrams = (tokens[i - 1], tokens[i])
            unigrams = tokens[i]
            # fetch the ngrams count which will be the frequency of each model
            unigrams_count = self.unigrams[protocol_type][unigrams]
            bigrams_count = self.bigrams[protocol_type][bigrams]
            trigrams_count = self.models[protocol_type][trigrams]
            total_count = self.total[protocol_type]
            # start by calculating the probabilities for each model and apply laplace smoothing
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
            # calculate the probability
            probability = trigrams_prob * 0.1 + bigrams_prob * 0.3 + unigram_prob * 0.6
            # then calculate the log of the probability and assign a fallback value
            if probability > 0:
                log_prob += math.log(probability)
            else:
                log_prob += float("-inf")
        # then return the probability
        return log_prob

    # a function to generate the next token in the sentence
    def generate_next_token(self, space_separated_tokens):
        # split the string into a list of tokens
        tokens = space_separated_tokens.split()
        # if the length is less than 2 then we want to add dummy tokens
        if len(tokens) < 2:
            # calculcate how many dummy tokens we should add
            tokens = ["<s>"] * (2 - len(tokens)) + tokens
        # extract the last two tokens
        last_token = tokens[-1]
        second_last_token = tokens[-2]
        # initialize the generated token and the highest probability
        generated_token = None
        max_probability = float("-inf")
        # now iterate over the tokens to start the calculations
        for i in self.vocabulary:
            # define the ngrams we want to work with
            trigram = (second_last_token, last_token, i)
            bigram = (last_token, i)
            unigram = i
            # here we fetch the counts, each line is the frequency of each token in the corpus
            unigrams_count = self.unigrams[self.default_type][unigram]
            bigrams_count = self.bigrams[self.default_type][bigram]
            trigrams_count = self.models[self.default_type][trigram]
            total_count = self.total[self.default_type]
            # if the total count is more than 0 then we calculate the probability and also apply laplace smoothing
            if total_count > 0:
                unigram_prob = (unigrams_count + self.laplace_constant) / (total_count + self.vocab_size)
            # else the probability is 0
            else:
                unigram_prob = 0
            # if the unigram has been seen then calculate the probability for bigrams and apply laplace
            if self.unigrams[self.default_type][last_token] > 0:
                bigrams_prob = (bigrams_count + self.laplace_constant) / (
                    self.unigrams[self.default_type][last_token] + self.vocab_size
                )
            # else assign the probability the value 0
            else:
                bigrams_prob = 0
            # repeat the same process for the bigrams to calculate the trigrams probability
            if self.bigrams[self.default_type][(second_last_token, last_token)] > 0:
                trigrams_prob = (trigrams_count + self.laplace_constant) / (
                    self.bigrams[self.default_type][(second_last_token, last_token)] + self.vocab_size
                )
            else:
                trigrams_prob = 0
            # now we combine the probabilites and give them different weights
            probability = trigrams_prob * 0.1 + bigrams_prob * 0.3 + unigram_prob * 0.6
            # if the probability is more than the max then we want to take this token
            if probability > max_probability:
                max_probability = probability
                generated_token = i
        # we return the values that we found
        return generated_token, max_probability

# a function to calculate the idf values for all the ngrams
def compute_idf(documents):
    # initialize a variable that will store the total number of documents
    total_docs = len(documents)
    # initialize a dictionary to track the frequency of each ngram
    doc_frequency = defaultdict(int)
    # iterate over all the documents
    for doc in documents:
        # first we need to generate ngrams so we split the document into words
        words = doc.split()
        # we generate ngrams from 1 to 4 and then we combine them and then we use the
        # set method in order to store only unique ngrams
        unique_terms = set(" ".join(words[i:i + n]) for i in range(len(words)) for n in range(1, 5) if i + n <= len(words))
        # now we loop over the unique terms to increment the count
        for term in unique_terms:
            doc_frequency[term] += 1
    # in here we calculate the idf according to the fomrula and then we return it
    return {term: math.log(total_docs / (1 + freq)) for term, freq in doc_frequency.items()}


def extract_ngrams(sentences, n):
    # define a counter to count each ngram occurence
    ngrams = Counter()
    # we want to iterate over the sentences
    for sentence in sentences:
        # then we split the sentence into words to be able to work with them
        words = sentence.split()
        # if the words aren't in the desired length then we continue
        if len(words) < n:
            continue
        # in here we generate the ngrams from the sentence
        ngrams.update(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    # then we return the counter object containing the ngrams as the keys and their counts as values
    return ngrams

# a function that returns the most frequent collocations
def get_k_n_t_collocations(k, n, t, corpus, type, idf_cache):
    # we define a dictionary to store the collocations for each corresponding type
    results = {"committee": {}, "plenary": {}}
    # first we iterate over the types to start extracting
    for protocol_type in ["committee", "plenary"]:
        # now we filter and extract the sentences for each type
        sentences = corpus[corpus["protocol_type"] == protocol_type]["sentence_text"].tolist()
        # call the extract function to get the ngrams with length n
        ngrams = extract_ngrams(sentences, n)
        # now we check if the type is frequency
        if type == "frequency":
            # we filter the collocations with a frequency over the threshold
            filtered_collocations = {coll: freq for coll, freq in ngrams.items() if freq >= t}
            # then we sort them in a decending order
            sorted_collocations = sorted(filtered_collocations.items(), key=lambda x: x[1], reverse=True)
            # we store the top k collocations in the correct protocol type key in the dict
            results[protocol_type] = dict(sorted_collocations[:k])
        # if the type is tfidf then we want to do other calculations
        elif type == "tfidf":
            # we first initialize a dictionaru for all the scores
            tf_idf_scores = {}
            # then calculate the total number of terms in the ngrams
            total_terms = sum(ngrams.values())
            # now we iterate over each ngram
            for ngram, freq in ngrams.items():
                # we first convert the ngrams to a string 
                term = " ".join(ngram)
                # then we calculate the tf according to the formula
                tf = freq / total_terms if total_terms > 0 else 0
                # we retrieve the precomputed idf value and set the default to 0 if it doesn't exist
                idf = idf_cache.get(term, 0)
                # now we store the product
                tf_idf_scores[ngram] = tf * idf
                # this is a print statement for debugging only
                print(f"TF: {tf}, IDF: {idf}, Term: {term}")
            # we sort the ngrams in a decending order based on their scores
            sorted_tfidf = sorted(tf_idf_scores.items(), key=lambda x: x[1], reverse=True)
            # and then we store only the top k scores in the results dict
            results[protocol_type] = dict(sorted_tfidf[:k])
    # then we return the dict
    return results

# this is a function to save the collocations to the txt file
def save_collocation_to_file(results, n, type):
    # we open a file with append mode and a utf-8 encoding to support non ascii values like hebrew letters
    with open("knesset_collocations.txt", "a", encoding="utf-8") as file:
        # we assing a value to the header as written in the format
        header = f"{n}-gram collocations ({type.capitalize()}):\n\n"
        # then we write it to the file
        file.write(header)
        # now we iterate over the protocol types in the results dict
        for protocol_type, collocations in results.items():
            # we write the protocol type just like in the format
            file.write(f"{protocol_type.capitalize()} corpus:\n")
            # we check if there are collocations for the current protocol type
            if collocations:
                # then we iterate over the collocations dict
                for collocation, value in collocations.items():
                    # then convert the tuple into a string
                    collocation_text = " ".join(collocation)
                    # and then write it to the file in the correct format
                    file.write(f"{collocation_text}: {value:.4f}\n")
            # if there are no collocations then we print a message
            else:
                file.write("No collocations found.\n")
            # the following new lines are for formatting
            file.write("\n")
        file.write("\n")

# this is the main entry for the program
if __name__ == "__main__":
    # a printing statement just to debug as well
    print("Starting the script...")
    # try to open the corpus file if it exists in reading mode
    try:
        with open("knesset_corpus.jsonl", "r", encoding="utf-8") as file:
            # load the file and parse it
            data = [json.loads(line) for line in file]
        # then convert the corpus to a dataframe
        corpus = pd.DataFrame(data)
        # printing statement for debugging
        print(f"Loaded {len(corpus)} records from 'knesset_corpus.jsonl'.")
    # catch the error if it occured and we didn't find a file then exit the program
    except FileNotFoundError:
        print("Error: File 'knesset_corpus.jsonl' not found.")
        exit()
    # define the arguments we want to pass to the function as requested in question 2
    k = 10
    lengths = [2, 3, 4]
    t = 5
    # precompute the idf for all the ngrams in the corpus
    idf_cache = compute_idf(corpus["sentence_text"].tolist())
    # iterate through each ngram of size n
    for n in lengths:
        # now we iterate over the score types
        for type in ["frequency", "tfidf"]:
            # i recorded the time for debugging as well because it was taking too long
            start_time = time.time()
            # printing statement also for debugging
            print(f"Processing {n}-gram collocations, type: {type}...")
            # call the function to get the required collocations
            collocations = get_k_n_t_collocations(k=k, n=n, t=t, corpus=corpus, type=type, idf_cache=idf_cache)
            # then we save the collocations to a the file
            save_collocation_to_file(collocations, n, type)
            # print the time to conclude the debugging process
            print(f"Completed {n}-gram collocations for type: {type} in {time.time() - start_time:.2f} seconds.")
    # print that the program has finished and the collocation has been saved
    print("Collocations have been saved to 'knesset_collocations.txt'.")
