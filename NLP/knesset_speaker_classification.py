import pandas as pd
import sys
import json

class Speaker:
    def __init__(self, name, sentences):
        self.name = name
        self.sentences = sentences

if __name__ == '__main__':
    if len(sys.argv) != 3:
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

    speaker_classes = {}
    for speaker_name, _ in top_speakers:
        speaker_classes[speaker_name] = Speaker(speaker_name, speaker_data[speaker_name])

    binary_data = []
    for speaker_name, speaker_obj in speaker_classes.items():
        for sentence in speaker_obj.sentences:
            binary_data.append({"speaker": speaker_name, "sentence": sentence})

    df = pd.DataFrame(binary_data)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(df.head())
    print(f"Binary classification data saved to {output_file}")
