import json
import re
from gensim.models import Word2Vec
import sys

# load the corpus from the file
def load_corpus(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sentences.append(data['sentence_text'])
    return sentences

# preprocess the sentences to remove non-Hebrew characters and tokenize the sentences
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

if __name__ == '__main__':
    # check if the correct number of arguments was given
    if(len(sys.argv) != 3):
        print("Usage: knesset_word2vec.py <corpus_file_path>")
        sys.exit(1)
    knesset_corpus_path = sys.argv[1]
    output_dir = sys.argv[2]

    # load the corpus, preprocess the sentences and train the word2vec model
    tokenized_sentences = preprocess_sentences(load_corpus(knesset_corpus_path))
    model = Word2Vec(tokenized_sentences, vector_size=50, window=5, min_count=1, workers=4)

    model_path = output_dir + '/knesset_word2vec.model'
    output_file_path = output_dir + '/knesset_similar_words.txt'
    
    model.save(model_path)

    # write the similar words to the output file
    words_to_check = ["ישראל", "גברת", "ממשלה", "חבר", "בוקר", "מים", "אסור", "רשות", "זכויות"]
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for word in words_to_check:
            similar_words = model.wv.most_similar(word, topn=5)
            formatted_similar_words = ", ".join(
                f"({similar_word}, {similarity_score:.4f})"
                for similar_word, similarity_score in similar_words
            )
            f.write(f"{word}: {formatted_similar_words}\n")