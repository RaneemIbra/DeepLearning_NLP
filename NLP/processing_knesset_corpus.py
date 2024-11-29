import os
import re
import json
import sys
from docx import Document

# function to split text into sentences
def divide_into_sentences(text):
    sentences = []
    sentence = ""
    for char in text:
        sentence += char
        if char in ".!?":  # sentence-ending punctuation
            if len(sentence.strip()) > 1:  # avoid single-character sentences
                sentences.append(sentence.strip())
                sentence = ""
    if sentence.strip():  # add any remaining text as the last sentence
        sentences.append(sentence.strip())
    return sentences

# function to check if a sentence is valid
def is_valid_sentence(sentence):
    # check if the sentence contains Hebrew characters
    if not re.search(r"[\u0590-\u05FF]", sentence):
        return False
    # exclude sentences with only special characters
    if re.fullmatch(r"[^\w\u0590-\u05FF]+", sentence):
        return False
    # avoid sentences with placeholder symbols
    if re.search(r"\.\.\.|---", sentence):
        return False
    return True

# function to tokenize a sentence into words and symbols
def tokenize_sentence(sentence):
    tokens = []
    for word in sentence.split():  # split the sentence into words
        tokens.extend(re.findall(r"\w+|[^\w\s]", word))
    return tokens

# main processing function
def process_protocol_files(input_folder, output_file):
    protocol_files = [f for f in os.listdir(input_folder) if f.endswith(".docx")]
    jsonl_data = []

    print("Found files:", protocol_files)
    
    for file in protocol_files:
        match = re.search(r'(\d+)_pt', file)
        knesset_number = int(match.group(1)) if match else -1  # extract knesset number
        
        # determine the protocol type based on file name
        if "ptm" in file:
            protocol_type = "plenary"
        elif "ptv" in file:
            protocol_type = "committee"
        else:
            protocol_type = "undefined"
        
        protocol_number = None
        
        try:
            doc_path = os.path.join(input_folder, file)
            doc = Document(doc_path)  # open the document
            
            # extract the protocol number from the first few paragraphs
            for paragraph in doc.paragraphs[:10]:
                match = re.search(r"פרוטוקול מס'? (\d+)", paragraph.text)
                if match:
                    protocol_number = int(match.group(1))
                    break
            
            if protocol_number is None:
                protocol_number = -1
            
            last_speaker = None
            
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue
                
                speaker_match = re.match(r"^([\u0590-\u05FF\w\s\(\)]+):", text)
                if speaker_match:
                    raw_name = speaker_match.group(1)
                    name = re.sub(r"\s*\(.*?\)", "", raw_name).strip()
                    spoken_text = text[len(speaker_match.group(0)):].strip()
                    last_speaker = name
                    
                    sentences = divide_into_sentences(spoken_text)
                    for sentence in sentences:
                        if is_valid_sentence(sentence):
                            tokens = tokenize_sentence(sentence)
                            if len(tokens) >= 4:
                                jsonl_data.append({
                                    "protocol_name": file,
                                    "knesset_number": knesset_number,
                                    "protocol_type": protocol_type,
                                    "protocol_number": protocol_number,
                                    "speaker_name": name,
                                    "sentence_text": " ".join(tokens)
                                })
                elif last_speaker:
                    additional_sentences = divide_into_sentences(text)
                    for sentence in additional_sentences:
                        if is_valid_sentence(sentence):
                            tokens = tokenize_sentence(sentence)
                            if len(tokens) >= 4:
                                jsonl_data.append({
                                    "protocol_name": file,
                                    "knesset_number": knesset_number,
                                    "protocol_type": protocol_type,
                                    "protocol_number": protocol_number,
                                    "speaker_name": last_speaker,
                                    "sentence_text": " ".join(tokens)
                                })
        
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # save the results to the specified JSONL file
    with open(output_file, "w", encoding="utf-8") as jsonl_file:
        for entry in jsonl_data:
            jsonl_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"JSONL file saved at: {output_file}")

# entry point for the script
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python processing_knesset_corpus.py <path/to/input_corpus_dir> <path/to/output_file_name.jsonl>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_file = sys.argv[2]
    
    process_protocol_files(input_folder, output_file)
