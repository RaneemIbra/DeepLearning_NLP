import json
import re
from gensim.models import Word2Vec
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

# compute the sentence embeddings
def compute_sentence_embeddings(sentences, model):
    sentence_embeddings = []
    # compute the sentence embedding as the average of the word embeddings
    for sentence in sentences:
        valid_tokens = [word for word in sentence if word in model.wv]
        # if there are no valid tokens, the sentence embedding is a zero vector
        if valid_tokens:
            # compute the sentence embedding as the average of the word embeddings
            word_vectors = np.array([model.wv[word] for word in valid_tokens])
            sentence_vector = np.mean(word_vectors, axis=0)
        else:
            sentence_vector = np.zeros(model.vector_size)
        sentence_embeddings.append(sentence_vector)
    return sentence_embeddings


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_most_similar_sentences(raw_sentences, embeddings, chosen_indices):
    # Ensure embeddings are 2D
    embeddings = np.array(embeddings)
    if embeddings.ndim == 1:  # Handle single-dimensional case
        embeddings = embeddings.reshape(-1, 1)

    results = []

    # Compute pairwise cosine similarities
    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)

    for idx in chosen_indices:
        if idx >= len(raw_sentences) or idx >= len(embeddings):
            raise IndexError(f"Index {idx} is out of bounds for the sentences or embeddings.")

        chosen_sentence = raw_sentences[idx]

        # Set self-similarity to -1 to exclude it
        similarities = similarity_matrix[idx]
        similarities[idx] = -1

        # Find the index of the most similar sentence
        most_similar_idx = np.argmax(similarities)
        most_similar_sentence = raw_sentences[most_similar_idx]

        results.append((chosen_sentence, most_similar_sentence))

    return results

if __name__ == '__main__':
    # check if the correct number of arguments was given
    if(len(sys.argv) != 3):
        print("Usage: knesset_word2vec.py <corpus_file_path> <output_dir>")
        sys.exit(1)
    knesset_corpus_path = sys.argv[1]
    output_dir = sys.argv[2]

    # load the corpus, preprocess the sentences and train the word2vec model
    raw_sentences = load_corpus(knesset_corpus_path)
    tokenized_sentences = preprocess_sentences(raw_sentences)
    model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1)

    model_path = output_dir + '/knesset_word2vec.model'
    output_file_path_similar_words = output_dir + '/knesset_similar_words.txt'
    
    model.save(model_path)

    # write the similar words to the output file
    words_to_check = ["ישראל", "גברת", "ממשלה", "חבר", "בוקר", "מים", "אסור", "רשות", "זכויות"]
    with open(output_file_path_similar_words, 'w', encoding='utf-8') as f:
        for word in words_to_check:
            similar_words = model.wv.most_similar(word, topn=5)
            formatted_similar_words = ", ".join(
                f"({similar_word}, {similarity_score:.4f})"
                for similar_word, similarity_score in similar_words
            )
            f.write(f"{word}: {formatted_similar_words}\n")

    # compute the sentence embeddings
    sentence_embeddings = compute_sentence_embeddings(tokenized_sentences, model)

    chosen_indices = [i for i, sentence in enumerate(tokenized_sentences) if len(sentence) >= 4][:10]
    similar_sentences = find_most_similar_sentences(raw_sentences, sentence_embeddings, chosen_indices)

    output_file_similar_sentences = output_dir + '/knesset_similar_sentences.txt'
    with open(output_file_similar_sentences, "w", encoding="utf-8") as file:
        for chosen_sentence, most_similar_sentence in similar_sentences:
            file.write(f"{chosen_sentence}: most similar sentence: {most_similar_sentence}\n")

    sentences = [
        "בעוד מספר דקות נתחיל את הדיון בנושא השבת החטופים .",
        "בתור יושבת ראש הוועדה , אני מוכנה להאריך את ההסכם באותם תנאים .",
        "בוקר טוב , אני פותח את הישיבה .",
        "שלום , אנחנו שמחים להודיע שחברינו היקר קיבל קידום .",
        "אין מניעה להמשיך לעסוק בנושא ."
    ]

    red_words_context = {
        "דקות": {"positive": ["דקה", "רגעים", "זמן"], "negative":[]},
        "הדיון": {"positive": ["דיבור", "מפגש", "מועצה", "שיחה"], "negative": ["קונפליקט", "ויכוח", "סכסוך"]},
        "הוועדה": {"positive": ["ועידה","אסיפה","רשות"], "negative": []},
        "אני": {"positive": ["עצמי"], "negative": []},
        "ההסכם": {"positive": [], "negative": []},
        "בוקר": {"positive": ["צהריים", "יום", "אור"], "negative": []},
        "פותח": {"positive": ["עוצר"], "negative": []},
        "שלום": {"positive": ["נעים", "ברוכים","ברוך","וברכה","וסהלן"],"negative": ["מלחמה"]},
        "שמחים": {"positive": [], "negative": []},
        "היקר": {"positive": ["מוערך"], "negative": ["זול"]},
        "קידום": {"positive": ["עצום","שיפור","עלייה","תוספות"], "negative": []},
        "מניעה": {"positive": ["אישורים", "הרשאה", "אפשרות", "היתר"], "negative": []}
    }

    output_file_red_words = output_dir + '/red_words_sentences.txt'
    with open(output_file_red_words, "w", encoding="utf-8") as file:
        for i, sentence in enumerate(sentences, start=1):
            original_sentence = sentence
            modified_sentence = sentence
            replacements = []

            for word, context in red_words_context.items():
                if word in sentence:
                    positive = context["positive"]
                    negative = context["negative"]

                    try:
                        similar_words = model.wv.most_similar(positive=positive + [word], negative=negative, topn=3)
                        replacement_word = similar_words[0][0]

                        modified_sentence = modified_sentence.replace(word, replacement_word)
                        replacements.append((word, replacement_word))
                        print(f"Top 3 replacements for '{word}' in '{sentence}':")
                        for similar_word, score in similar_words:
                            print(f"  {similar_word} ({score:.4f})")

                    except KeyError:
                        replacement_word = word

            file.write(
                f"{i}: {original_sentence}: {modified_sentence}\n"
                f"replaced words: {', '.join(f'({orig}:{rep})' for orig, rep in replacements)}\n"
            )