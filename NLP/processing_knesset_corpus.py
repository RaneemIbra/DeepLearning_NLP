import os
import re
import json
import sys
from docx import Document

# function to split text into sentences by seeing the punctuation and splitting from there
def divideToSentences(text):
    # we create a list to store the sentences and initialize an empty sentence to add to it the sentences that we find
    sentences = []
    sentence = ""
    # loop over the whole text and gather the sentences
    for char in text:
        # we add all the chars to the sentence
        sentence += char
        # if the char is a punctuation that ends the sentence then we finish the sentence
        if char == "." or char == "!" or char == "?":
            # we check if the sentence has a meaning and not only a single char then we add it
            if len(sentence.strip()) > 1:
                # we add the sentence to the list, and we use strip to avoid white spaces
                sentences.append(sentence.strip())
                # init the sentence to blank again to avoid accumulation 
                sentence = ""
    # after we finish looping over the text we add any remaining text and then return the list
    if sentence.strip():
        sentences.append(sentence.strip())
    return sentences

# function to check if a sentence is valid
def ValidSentence(sentence):
    # check if the sentence contains Hebrew characters, the regex is the unicode for hebrew letters
    if not re.search(r"[\u0590-\u05FF]", sentence):
        return False
    # if the sentence doesn't have any meaningful text like special chars only then we want to not include it
    if re.fullmatch(r"[^\w\u0590-\u05FF]+", sentence):
        return False
    # if the sentence has placeholders we also want to avoid it
    if re.search(r"\.\.\.|---", sentence):
        return False
    return True

# function to tokenize a sentence into words and symbols
def Tokenize(sentence):
    # a list that will hold all the tokens
    tokens = []
    # split the sentence into words from the whitespaces
    for word in sentence.split():
        # this line also splits the tokens to separate the special chars so "hello," will be "hello" + ","
        tokens.extend(re.findall(r"\w+|[^\w\s]", word))
    return tokens

# main processing function, where almost all the requirements happen (explained in each line)
# the function takes the input folder where all the docx file are, and also where the output file will be stored
def workOnFilesFunc(inputFolder, outputFile):
    # we read the protocol files to start processing them
    protocol_files = [f for f in os.listdir(inputFolder) if f.endswith(".docx")]
    # for debugging to check if we are reading the correct files
    print("Found files:", protocol_files)
    # init an empty list for the data that we will store in the jsonl file
    jsonlList = []
    # loop over the files to start the processing
    for file in protocol_files:
        # we use regex here to extract the kenest number from the docx title
        match = re.search(r'(\d+)_pt', file)
        # extract knesset number and if we didn't find a number then as a fallback we assign -1
        knessetNumber = int(match.group(1)) if match else -1
        # the following if else statements check if the file is for a plenary or a committee and as a fallback we assign undefined
        if "ptm" in file:
            protocolType = "plenary"
        elif "ptv" in file:
            protocolType = "committee"
        else:
            protocolType = "undefined"
        # we start with a none value for the protocol number
        protocolNum = None
        # we use a try except block as proposed in the hw document to start processing
        try:
            # we create the path for the docx and we open it using the Document func
            docPath = os.path.join(inputFolder, file)
            doc = Document(docPath)
            # start by extracting the protocol number from the first 10 paragraphs (as per our choice)
            for paragraph in doc.paragraphs[:10]:
                # search for the protocol number using the text in hebrew
                foundNum = re.search(r"פרוטוקול מס'? (\d+)", paragraph.text)
                # if the found number isn't none then we assign it to the protocol number and break out of the loop to stop the search for the number
                if foundNum:
                    protocolNum = int(foundNum.group(1))
                    break
            # if we didn't find a protocol number then we set it to -1 as a fall back as requested
            if protocolNum is None:
                protocolNum = -1
            # we define a last that will benifit us to attribute texts that belongs to no one from the context
            lastSpeaker = None
            # loop over all the paragraphs in the document
            for paragraph in doc.paragraphs:
                # remove the white spaces
                text = paragraph.text.strip()
                # if the paragraph is empty then we skip it
                if not text:
                    continue
                # we used regex again to find the speaker name by checking for a name followed by :
                speakerFound = re.match(r"^([\u0590-\u05FF\w\s\(\)]+):", text)
                if speakerFound:
                    #we extract the raw name
                    rawName = speakerFound.group(1)
                    # here we remove any additional information like titles for example
                    name = re.sub(r"\s*\(.*?\)", "", rawName).strip()
                    # we extract the text that was said after the :
                    spokenText = text[len(speakerFound.group(0)):].strip()
                    # then we assign the name to the last speaker so that we can attribute what is said after
                    lastSpeaker = name
                    # call the divideToSentences function to divide the text
                    sentences = divideToSentences(spokenText)
                    # loop over the sentences to process them further
                    for sentence in sentences:
                        # check if the sentence is valid according to the requirements
                        if ValidSentence(sentence):
                            # if the sentence is valid then we tokenize it
                            tokens = Tokenize(sentence)
                            # now we check if the token is longer than 4 (or equal)
                            if len(tokens) >= 4:
                                # if the token satisfies then we want to store it in the list
                                jsonlList.append({
                                    "protocol_name": file,
                                    "knesset_number": knessetNumber,
                                    "protocol_type": protocolType,
                                    "protocol_number": protocolNum,
                                    "speaker_name": name,
                                    "sentence_text": " ".join(tokens)
                                })
                # now we repeat the same process if there was no speaker found by attributing the text to the last speaker
                elif lastSpeaker:
                    additional_sentences = divideToSentences(text)
                    for sentence in additional_sentences:
                        if ValidSentence(sentence):
                            tokens = Tokenize(sentence)
                            if len(tokens) >= 4:
                                jsonlList.append({
                                    "protocol_name": file,
                                    "knesset_number": knessetNumber,
                                    "protocol_type": protocolType,
                                    "protocol_number": protocolNum,
                                    "speaker_name": lastSpeaker,
                                    "sentence_text": " ".join(tokens)
                                })
        # here we catch the exception if a problem occured while processing a file
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    # here we open the output file in writing mode and we write each item from the list to here
    with open(outputFile, "w", encoding="utf-8") as jsonl_file:
        for entry in jsonlList:
            jsonl_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # print where the file was stored to see if it worked
    print(f"JSONL file saved at: {outputFile}")

if __name__ == "__main__":
    # check if the command is wrong
    if len(sys.argv) != 3:
        print("Usage: python processing_knesset_corpus.py <path/to/input_corpus_dir> <path/to/outputFile_name.jsonl>")
        sys.exit(1)
    # otherwise we want to take the path for the input and the path for the output
    inputFolder = sys.argv[1]
    outputFile = sys.argv[2]
    # pass the paths to the processing function
    workOnFilesFunc(inputFolder, outputFile)
