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
        tokens = ["<s>", "<s>"] + space_separated_tokens.split() + ["</s>"]
        log_prob = 0.0
        protocol_type = self.default_type

        for i in range(2, len(tokens)):
            trigram = (tokens[i - 2], tokens[i - 1], tokens[i])
            bigram = (tokens[i - 1], tokens[i])
            unigram = tokens[i]

            total_count = self.total[protocol_type]

            # Calculate smoothed probabilities
            unigram_count = self.unigrams[protocol_type][unigram]
            unigram_prob = (unigram_count + self.laplace_constant) / (total_count + self.vocab_size)

            bigram_count = self.bigrams[protocol_type][bigram]
            prev_unigram_count = self.unigrams[protocol_type][tokens[i - 1]]
            bigram_prob = (bigram_count + self.laplace_constant) / (
                prev_unigram_count + self.vocab_size
            ) if prev_unigram_count > 0 else 1e-10

            trigram_count = self.models[protocol_type][trigram]
            prev_bigram_count = self.bigrams[protocol_type][(tokens[i - 2], tokens[i - 1])]
            trigram_prob = (trigram_count + self.laplace_constant) / (
                prev_bigram_count + self.vocab_size
            ) if prev_bigram_count > 0 else 1e-10

            # Dynamic weighting based on sentence position
            position = i - 2  # Actual sentence index excluding initial <s>
            sentence_length = len(tokens) - 4  # Exclude <s> and </s>
            trigram_weight = 0.1 + 0.4 * (position / sentence_length) if sentence_length > 0 else 0.5
            bigram_weight = 0.3
            unigram_weight = 1 - trigram_weight - bigram_weight

            probability = (
                trigram_weight * trigram_prob + bigram_weight * bigram_prob + unigram_weight * unigram_prob
            )

            if probability > 0:
                log_prob += math.log(probability)
            else:
                log_prob += math.log(1e-10)

        return log_prob



    # a function to generate the next token in the sentence
    def generate_next_token(self, space_separated_tokens):
        tokens = ["<s>", "<s>"] + space_separated_tokens.split()
        last_token = tokens[-1]
        second_last_token = tokens[-2]
        protocol_type = self.default_type

        max_score = float("-inf")
        best_token = None

        for token in self.vocabulary:
            if token in {"<s>", "</s>"}:
                continue

            trigram = (second_last_token, last_token, token)
            bigram = (last_token, token)
            unigram = token

            trigram_count = self.models[protocol_type][trigram]
            bigram_count = self.bigrams[protocol_type][bigram]
            unigram_count = self.unigrams[protocol_type][unigram]
            total_count = self.total[protocol_type]

            # Calculate probabilities with smoothing
            unigram_prob = (unigram_count + self.laplace_constant) / (total_count + self.vocab_size)
            bigram_prob = (bigram_count + self.laplace_constant) / (
                self.unigrams[protocol_type][last_token] + self.vocab_size
            ) if self.unigrams[protocol_type][last_token] > 0 else 1e-10
            trigram_prob = (trigram_count + self.laplace_constant) / (
                self.bigrams[protocol_type][(second_last_token, last_token)] + self.vocab_size
            ) if self.bigrams[protocol_type][(second_last_token, last_token)] > 0 else 1e-10

            # Weighted interpolation
            combined_prob = 0.6 * unigram_prob + 0.3 * bigram_prob + 0.1 * trigram_prob

            # Penalize very frequent tokens dynamically
            token_frequency = unigram_count / (total_count + 1)
            penalty_factor = max(0.1, 1 - token_frequency)  # More frequent tokens get penalized more
            combined_prob *= penalty_factor

            # Update the best token
            if combined_prob > max_score:
                max_score = combined_prob
                best_token = token

        return best_token, math.log(max_score) if max_score > 0 else math.log(1e-10)


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
    return {term: math.log(total_docs / (1 + freq)) for term, freq in doc_frequency.items() if term.strip() not in {",", ".", ";", ":", "–"}}

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
    masked_sentences = []
    masked_indices = []

    for sentence in sentences:
        tokens = sentence.split()
        num_to_mask = max(1, int(len(tokens) * x / 100))
        indices_to_mask = random.sample(range(len(tokens)), num_to_mask)

        masked_sentence = tokens[:]
        for index in indices_to_mask:
            masked_sentence[index] = "[*]"

        masked_sentences.append(" ".join(masked_sentence))
        masked_indices.append(indices_to_mask)

    return masked_sentences, masked_indices

# a function that returns the origial sentences that we selected
def mask_sentences(corpus, num_sentences, mask_precent):
    # select the original sentences
    original_sentences = random.sample(corpus, num_sentences)
    # mask the original sentences and save them in another list
    masked_sentences, masked_indices = mask_tokens_in_sentences(original_sentences, mask_precent)
    # return both of the lists and the indices
    return original_sentences, masked_sentences, masked_indices

# a function to save the sentences in a file, we pass the name of the file just to use the function twice
def save_sentences_to_file(sentences, file_name):
    # open a file with the file name we passed and encoding utf-8 to support non ascii values
    with open(file_name, "w", encoding="utf-8") as file:
        # then iterate over all the sentences and write them to the file
        for sentence in sentences:
            # Ensure only strings are written
            if isinstance(sentence, str):
                file.write(sentence + "\n")
            else:
                print(f"Error: Sentence is not a string: {sentence}")

# a function to guess the masked tokens
def generate_results(original_sentences, masked_sentences, masked_indices, trigram_model_plenary, trigram_model_committee):
    results = []
    trigram_model_committee.default_type = "committee"
    trigram_model_plenary.default_type = "plenary"

    for original, masked, indices in zip(original_sentences, masked_sentences, masked_indices):
        tokens = masked.split()
        guessed_tokens = []

        for index in indices:
            context = " ".join(tokens[:index])  # Use only preceding tokens for context
            guessed_token, _ = trigram_model_plenary.generate_next_token(context)
            guessed_tokens.append(guessed_token)
            tokens[index] = guessed_token

        plenary_sentence = " ".join(tokens)
        plenary_tokens = ", ".join(guessed_tokens)
        plenary_prob = round(trigram_model_plenary.calculate_prob_of_sentence(plenary_sentence), 2)
        committee_prob = round(trigram_model_committee.calculate_prob_of_sentence(plenary_sentence), 2)

        result = (
            f"original_sentence: {original}\n"
            f"masked_sentence: {masked}\n"
            f"plenary_sentence: {plenary_sentence}\n"
            f"plenary_tokens: {plenary_tokens}\n"
            f"probability of plenary sentence in plenary corpus: {plenary_prob}\n"
            f"probability of plenary sentence in committee corpus: {committee_prob}\n"
        )
        results.append(result)

    with open("sampled_sents_results.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(results))

def calculate_perplexity(masked_sentences, masked_indices, trigram_model):
    perplexities = []

    for sentence, indices in zip(masked_sentences, masked_indices):
        tokens = ["<s>", "<s>"] + sentence.split() + ["</s>"]
        log_prob_sum = 0.0
        masked_count = len(indices)

        for index in indices:
            prev_2 = tokens[index - 2] if index - 2 >= 0 else "<s>"
            prev_1 = tokens[index - 1] if index - 1 >= 0 else "<s>"
            token = tokens[index]

            trigram = (prev_2, prev_1, token)
            bigram = (prev_1, token)
            unigram = token

            protocol_type = trigram_model.default_type
            total_count = trigram_model.total[protocol_type]

            unigram_count = trigram_model.unigrams[protocol_type][unigram]
            bigram_count = trigram_model.bigrams[protocol_type][bigram]
            trigram_count = trigram_model.models[protocol_type][trigram]

            unigram_prob = (unigram_count + trigram_model.laplace_constant) / (total_count + trigram_model.vocab_size)
            bigram_prob = (bigram_count + trigram_model.laplace_constant) / (
                trigram_model.unigrams[protocol_type][prev_1] + trigram_model.vocab_size
            ) if trigram_model.unigrams[protocol_type][prev_1] > 0 else 1e-10
            trigram_prob = (trigram_count + trigram_model.laplace_constant) / (
                trigram_model.bigrams[protocol_type][(prev_2, prev_1)] + trigram_model.vocab_size
            ) if trigram_model.bigrams[protocol_type][(prev_2, prev_1)] > 0 else 1e-10

            combined_prob = 0.6 * unigram_prob + 0.3 * bigram_prob + 0.1 * trigram_prob
            log_prob_sum += math.log(combined_prob) if combined_prob > 0 else math.log(1e-10)

        if masked_count > 0:
            perplexity = math.exp(-log_prob_sum / masked_count)
            perplexities.append(perplexity)

    return sum(perplexities) / len(perplexities) if perplexities else float("inf")




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
    original_sentences, masked_sentences, masked_indices = mask_sentences(committee_sentences, 10, 10)
    # write the sentences to the files
    save_sentences_to_file(original_sentences, "original_sampled_sents.txt")
    save_sentences_to_file(masked_sentences, "masked_sampled_sents.txt")

    # now you might want to sit tight because it will crash so bad :)
    generate_results(original_sentences, masked_sentences, masked_indices, trigram_model_plenary, trigram_model_committee)
    print("file saved")

    avg_perplexity = calculate_perplexity(masked_sentences, masked_indices, trigram_model_plenary)
    print(f"Average perplexity for masked tokens: {avg_perplexity:.2f}")