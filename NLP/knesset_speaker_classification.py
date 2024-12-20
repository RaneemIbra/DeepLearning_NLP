import numpy as np
import pandas as pd
import sys
import json

class Speaker:
    def __init__(self, name, sentences):
        self.name = name
        self.sentences = sentences

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_jsonl_file> <output_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    speaker_data = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            speaker_name = record['speaker_name']
            sentence = record['sentence_text']

            if speaker_name not in speaker_data:
                speaker_data[speaker_name] = []
            speaker_data[speaker_name].append(sentence)

    speaker_counts = {speaker: len(sentences) for speaker, sentences in speaker_data.items()}

    top_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:2]

    data = []
    for speaker_name, sentences in speaker_data.items():
        if speaker_name == top_speakers[0][0]:
            label = 'first speaker'
        elif speaker_name == top_speakers[1][0]:
            label = 'second speaker'
        else:
            label = 'other'
        for sentence in sentences:
            data.append({"sentence": sentence, "label": label, "Speaker name": speaker_name})

    df = pd.DataFrame(data)

    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"Top Speakers:")
    print(f"1. {top_speakers[0][0]}: {top_speakers[0][1]} sentences")
    print(f"2. {top_speakers[1][0]}: {top_speakers[1][1]} sentences")
    print(f"Data saved to {output_file}")
