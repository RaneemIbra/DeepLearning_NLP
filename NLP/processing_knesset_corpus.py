import os
import re
import json
import sys
from docx import Document

# function to split text into sentences by seeing the punctuation and splitting from there
def divideToSentences(text):
    sentences = []
    sentence = ""
    for char in text:
        sentence += char
        if char in ".!?":
            if len(sentence.strip()) > 1:
                sentences.append(sentence.strip())
                sentence = ""
    if sentence.strip():
        sentences.append(sentence.strip())
    return sentences

# function to check if a sentence is valid
def ValidSentence(sentence):
    containsHebrew = any("\u0590" <= char <= "\u05FF" for char in sentence)
    if not containsHebrew:
        return False
    words = re.split(r'\s+', sentence.strip())
    if all(re.fullmatch(r"[^\w\u0590-\u05FF]+", word) for word in words):
        return False
    if "..." in sentence or "---" in sentence:
        return False
    return True

# Function to tokenize a sentence into words and symbols
def Tokenize(sentence):
    hebrew_letters = r"[\u0590-\u05FF]+"
    nikud_pattern = r"[\u05B0-\u05C7]+"
    punctuation_pattern = r"[^\w\u0590-\u05FF\s]"
    number_pattern = r"[$₪€£]?\d+(?:\.\d+)?%?"
    url_pattern = r"https?://\S+|www\.\S+"
    email_pattern = r"[\w.+-]+@\w+\.\w+"
    hyphenated_word_pattern = r"[\u0590-\u05FF]+(?:-[\u0590-\u05FF]+)*"
    abbreviation_pattern = r"\b(?:[\u0590-\u05FF]\.){2,}"
    parenthesis_pattern = r"\([^()]*\)"
    
    combined_pattern = f"({url_pattern}|{email_pattern}|{number_pattern}|{hebrew_letters}|{nikud_pattern}|{hyphenated_word_pattern}|{abbreviation_pattern}|{punctuation_pattern}|{parenthesis_pattern})"
    tokens = re.findall(combined_pattern, sentence)
    return [token for token in tokens if token.strip()]

# Function to extract and validate speaker names
def extractSpeakerName(text):
    """
    Extracts the speaker name from the text based on a predefined pattern.
    Validates the extracted name based on common speaker title keywords or format.
    """
    valid_title_keywords = [
        "יושב ראש", "חבר הכנסת", "חברת הכנסת", 
        "שר", "שרה", "ראש הממשלה", "נשיא", 
        "סגן", "מזכיר", "מזכירה", "יו\"ר"
    ]
    # Regex pattern to match speaker names followed by a colon
    speaker_pattern = r"^([\u0590-\u05FF\s\(\)\"'`]+):"
    speakerFound = re.match(speaker_pattern, text)
    
    if speakerFound:
        # Extract and clean the raw speaker name
        rawName = speakerFound.group(1).strip()
        cleanedName = re.sub(r"\s*\(.*?\)", "", rawName)  # Remove parentheses
        cleanedName = re.sub(r"\"|'", "", cleanedName)   # Remove quotes
        
        # Check if the cleaned name contains valid titles or Hebrew characters
        isValidName = (
            any(keyword in cleanedName for keyword in valid_title_keywords) or
            bool(re.fullmatch(r"[\u0590-\u05FF\s]+", cleanedName))
        )
        
        # If valid, return the cleaned name; otherwise, return "Unknown Speaker"
        return cleanedName if isValidName else "Unknown Speaker"
    else:
        return "Unknown Speaker"

# Main processing function
def workOnFilesFunc(inputFolder, outputFile):
    global global_token_count
    files = [file_name for file_name in os.listdir(inputFolder) if file_name.endswith(".docx")]
    print(f"files: {files}")
    jsonlList = []
    for file in files:
        match = re.search(r'(\d+)_pt', file)
        knessetNumber = int(match.group(1)) if match else -1
        protocolType = "plenary" if "ptm" in file else "committee" if "ptv" in file else "undefined"
        protocolNum = None
        try:
            docPath = os.path.join(inputFolder, file)
            doc = Document(docPath)
            for paragraph in doc.paragraphs[:10]:
                foundNum = re.search(r"פרוטוקול מס'? (\d+)", paragraph.text)
                if foundNum:
                    protocolNum = int(foundNum.group(1))
                    break
            if protocolNum is None:
                protocolNum = -1
            lastSpeaker = None
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue
                
                # Extract and validate speaker name
                speakerName = extractSpeakerName(text)
                
                # If a valid speaker name is found, update the last speaker context
                if speakerName != "Unknown Speaker":
                    lastSpeaker = speakerName
                    spokenText = text[len(speakerName) + 1:].strip()
                    sentences = divideToSentences(spokenText)
                    for sentence in sentences:
                        if ValidSentence(sentence):
                            tokens = Tokenize(sentence)
                            if len(tokens) >= 4:
                                global_token_count += len(tokens)
                                jsonlList.append({
                                    "protocol_name": file,
                                    "knesset_number": knessetNumber,
                                    "protocol_type": protocolType,
                                    "protocol_number": protocolNum,
                                    "speaker_name": speakerName,
                                    "sentence_text": " ".join(tokens)
                                })
                # If no valid speaker name is found, assign the last valid speaker
                elif lastSpeaker:
                    additionalSentences = divideToSentences(text)
                    for sentence in additionalSentences:
                        if ValidSentence(sentence):
                            tokens = Tokenize(sentence)
                            if len(tokens) >= 4:
                                global_token_count += len(tokens)
                                jsonlList.append({
                                    "protocol_name": file,
                                    "knesset_number": knessetNumber,
                                    "protocol_type": protocolType,
                                    "protocol_number": protocolNum,
                                    "speaker_name": lastSpeaker,
                                    "sentence_text": " ".join(tokens)
                                })
        except Exception as e:
            print(f"{file}: {e}")
    with open(outputFile, "w", encoding="utf-8") as jsonl_file:
        for entry in jsonlList:
            jsonl_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"file saved at: {outputFile}")
    print(f"Tokens count: {global_token_count}")

if __name__ == "__main__":
    global_token_count = 0
    if len(sys.argv) != 3:
        sys.exit(1)
    inputFolder = sys.argv[1]
    outputFile = sys.argv[2]
    workOnFilesFunc(inputFolder, outputFile)
