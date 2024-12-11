import json
from collections import defaultdict, Counter
import math
import pandas as pd
import time
import random

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
        self.default_type = "plenary"
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
        tokens = space_separated_tokens.split()
        if len(tokens) < 2:
            tokens = ["<s>"] * (2 - len(tokens)) + tokens

        last_token = tokens[-1]
        second_last_token = tokens[-2]
        generated_token = None
        max_score = float("-inf")

        for token in self.vocabulary:
            if token in {"<s>", "</s>"}:  # Skip dummy tokens
                continue

            # Define n-grams
            trigram = (second_last_token, last_token, token)
            bigram = (last_token, token)
            unigram = token

            # Fetch counts
            trigram_count = self.models[self.default_type][trigram]
            bigram_count = self.bigrams[self.default_type][bigram]
            unigram_count = self.unigrams[self.default_type][unigram]
            total_count = self.total[self.default_type]

            # Calculate probabilities
            unigram_prob = (unigram_count + self.laplace_constant) / (total_count + self.vocab_size) if total_count > 0 else 0
            bigram_prob = (bigram_count + self.laplace_constant) / (
                self.unigrams[self.default_type][last_token] + self.vocab_size) if self.unigrams[self.default_type][last_token] > 0 else 0
            trigram_prob = (trigram_count + self.laplace_constant) / (
                self.bigrams[self.default_type][(second_last_token, last_token)] + self.vocab_size) if self.bigrams[self.default_type][(second_last_token, last_token)] > 0 else 0

            # Dynamic weights
            position = len(tokens)
            sentence_length = len(space_separated_tokens.split())
            trigram_weight = 0.05
            bigram_weight = 0.1
            unigram_weight = 0.85

            # Combine scores with weights
            combined_score = trigram_prob * trigram_weight + bigram_prob * bigram_weight + unigram_prob * unigram_weight

            # Penalize frequent or punctuation tokens
            if token in {",", ".", ";"}:
                combined_score /= 2

            # Update max score
            if combined_score > max_score:
                max_score = combined_score
                generated_token = token

        return generated_token, math.log(max_score) if max_score > 0 else float("-inf")


def train_trigram_model(model, sentences, protocol_type):
    for sentence in sentences:
        words = ["<s>", "<s>"] + sentence.split() + ["</s>"]
        model.vocabulary.update(words)  # Add words to the vocabulary
        model.total[protocol_type] += len(words)
        for i in range(len(words)):
            # Update unigrams
            model.unigrams[protocol_type][words[i]] += 1
            if i > 0:
                # Update bigrams
                model.bigrams[protocol_type][(words[i - 1], words[i])] += 1
            if i > 1:
                # Update trigrams
                model.models[protocol_type][(words[i - 2], words[i - 1], words[i])] += 1

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

# a function to mask the tokens in the corpus
def mask_tokens_in_sentences(sentences, x):
    # define the list of the masked sentences
    masked_sentences = []
    # iterate over the sentences
    for sentence in sentences:
        # split the sentences to tokens in order to be able to work with them
        tokens = sentence.split()
        # calculate the number of tokens we need to mask
        number_of_tokens_to_mask = max(1, int(len(tokens) * x / 100))
        # select the indecies to mark randomly
        indecies_to_mask = random.sample(range(len(tokens)), number_of_tokens_to_mask)
        # iterate over the indecies and in each index we need to change the token to the [*]
        for index in indecies_to_mask:
            tokens[index] = "[*]"
        # join the sentences back into a sentence
        masked_sentences.append(" ".join(tokens))
    # return the masked sentences
    return masked_sentences

# a function that returns the origial sentences that we selected
def mask_sentences(corpus, num_sentences, mask_precent):
    # select the original sentences
    original_sentences = random.sample(corpus, num_sentences)
    # mask the original sentences and save them in another list
    masked_sentences = mask_tokens_in_sentences(original_sentences, mask_precent)
    # return both of the lists
    return original_sentences, masked_sentences

# a function to save the sentences in a file, we pass the name of the file just to use the function twice
def save_sentences_to_file(sentences, file_name):
    # open a file with the file name we passed and encoding utf-8 to support non ascii values
    with open(file_name, "w", encoding="utf-8") as file:
        # then iterate over all the sentences and write them to the file
        for sentence in sentences:
            file.write(sentence + "\n")

# a function to guess the masked tokens
def generate_results(original_sentences, masked_sentences, trigram_model_plenary, trigram_model_committee):
    # a list for the resulted tokens
    results = []
    # define the default type of the model
    trigram_model_committee.default_type = "committee"
    trigram_model_plenary.default_type = "plenary"
    print(trigram_model_plenary.default_type)
    print(trigram_model_committee.default_type)
    print(f"committee: {trigram_model_committee.vocab_size}")
    print(f"plenary: {trigram_model_plenary.vocab_size}")
    print(trigram_model_plenary.vocabulary)
    print(trigram_model_committee.vocabulary)
    # iterate over the sentences to start guessing the words
    for original, masked in zip(original_sentences, masked_sentences):
        # a list to store the guessed tokens
        tokens = []
        # this is the sentence that we will iterate over to guess the masked tokens
        current_sentence = masked
        while "[*]" in current_sentence.split():
            # call the generate function to get the token we want
            guessed_token, _ = trigram_model_plenary.generate_next_token(current_sentence)
            if guessed_token is None:
                print("Error: Unable to generate a token. Check the model and vocabulary.")
                break
            # add the guessed tokens to the list
            tokens.append(guessed_token)
            print(f"guessed token: {guessed_token}")
            # replace the first apperance of [*] with the guessed token
            current_sentence = current_sentence.replace("[*]", guessed_token, 1)
        # assign the value of the plenary sentence
        plenary_sentence = current_sentence
        # join the plenary tokens into a string with commas as separators
        plenary_tokens = ", ".join(tokens)
        # calculate the probability of the sentences
        probability_of_plenary_sentence_in_plenary_corpus = trigram_model_plenary.calculate_prob_of_sentence(plenary_sentence)
        probability_of_plenary_sentence_in_committee_corpus = trigram_model_committee.calculate_prob_of_sentence(plenary_sentence)
        # format the result in order to write them to the file
        result = (f"original_sentence: {original}\n" f"masked_sentence: {masked}\n"
        f"plenary_sentence: {plenary_sentence}\n" f"plenary_tokens: {plenary_tokens}\n"
        f"probability of plenary sentence in plenary corpus: {probability_of_plenary_sentence_in_plenary_corpus}\n"
        f"probability of plenary sentence in committee corpus: {probability_of_plenary_sentence_in_committee_corpus}\n")
        # append the result to the results list
        results.append(result)
    # write the results to the file
    with open("sampled_sents_results.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(results))

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
    # fetch all the committee sentences and put them in a list
    committee_sentences = corpus[corpus["protocol_type"] == "committee"]["sentence_text"].tolist()
    plenary_sentences = corpus[corpus["protocol_type"] == "plenary"]["sentence_text"].tolist()
    trigram_model_committee = Trigram_LM(vocab_size=len(committee_sentences))
    trigram_model_plenary = Trigram_LM(vocab_size=len(plenary_sentences))

    train_trigram_model(trigram_model_committee, committee_sentences, "committee")
    train_trigram_model(trigram_model_plenary, plenary_sentences, "plenary")

    # get the original sentences and the masked sentences using the function that we wrote before
    original_sentences, masked_sentences = mask_sentences(committee_sentences, 10, 10)
    # write the sentences to the files
    save_sentences_to_file(original_sentences, "original_sampled_sents.txt")
    save_sentences_to_file(masked_sentences, "masked_sampled_sents.txt")

    # now you might want to sit tight because it will crash so bad :)
    generate_results(original_sentences, masked_sentences, trigram_model_plenary, trigram_model_committee)
    print("file saved")