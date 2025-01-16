from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import sys

tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
model = AutoModelForMaskedLM.from_pretrained("avichr/heBERT")

def load_masked_sentences(filepath):
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            original_sentence = line.strip()
            masked_sentence = original_sentence.replace("*", "[MASK]")
            sentences.append((original_sentence, masked_sentence))
    return sentences

def predict_masked_tokens(masked_sentence):
    inputs = tokenizer(masked_sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    mask_token_index = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    predicted_tokens = []
    for mask_index in mask_token_index:
        top_token_id = logits[0, mask_index].argmax(dim=-1).item()
        predicted_tokens.append(tokenizer.decode([top_token_id]).strip())

    return predicted_tokens

def process_and_save_results(sentences, output_filepath):
    with open(output_filepath, 'w', encoding='utf-8') as file:
        for original_sentence, masked_sentence in sentences:
            predicted_tokens = predict_masked_tokens(masked_sentence)

            dictaBERT_sentence = masked_sentence
            for token in predicted_tokens:
                dictaBERT_sentence = dictaBERT_sentence.replace("MASK", token, 1)

            file.write(f"original_sentence: {original_sentence}\n")
            file.write(f"masked_sentence: {masked_sentence}\n")
            file.write(f"dictaBERT_sentence: {dictaBERT_sentence}\n")
            file.write(f"dictaBERT tokens: {', '.join(predicted_tokens)}\n")
            file.write("\n")

if __name__ == '__main__':
    if(len(sys.argv) != 3):
        print("Usage: knesset_word2vec_classification.py <corpus_file_path> <model_path>")
        sys.exit(1)

    masked_sentences_file = sys.argv[1]
    output_file = sys.argv[2]
    sentences = load_masked_sentences(masked_sentences_file)
    process_and_save_results(sentences, output_file)