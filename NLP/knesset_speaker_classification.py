import numpy as np
import pandas as pd
import sys
import json
import random
random.seed(42)
np.random.seed(42)

class Speaker:
    def __init__(self, name):
        self.name = name
        self.objects = []

map_speakers_to_aliases = {
    "ר' ריבלין": "ראובן ריבלין",
    "ראובן ריבלין": "ראובן ריבלין",
    "רובי ריבלין": "ראובן ריבלין",
    "אברהם בורג": "א' בורג",
    "א' בורג": "א' בורג"
}

# define an arrow function to retrieve the name under the alias
def get_speker_name_from_alias(name: str) -> str:
    return map_speakers_to_aliases.get(name, name)

if __name__ == '__main__': 
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_jsonl_file> <output_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    speaker_data = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            objects = json.loads(line.strip())
            raw_name = objects['speaker_name']
            name_under_the_alias = get_speker_name_from_alias(raw_name)
            if name_under_the_alias not in speaker_data:
                speaker_data[name_under_the_alias] = Speaker(name_under_the_alias)
            speaker_data[name_under_the_alias].objects.append(objects)

    speaker_counts = {
        speaker_name: len(speaker_object.objects) 
        for speaker_name, speaker_object in speaker_data.items()
    }

    top_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:2]

    data = []
    for speaker_name, speaker_object in speaker_data.items():
        if speaker_name == top_speakers[0][0]:
            label = 'first speaker'
        elif speaker_name == top_speakers[1][0]:
            label = 'second speaker'
        else:
            label = 'other'
        for objects in speaker_object.objects:
            row = {
                "protocol_name": objects["protocol_name"],
                "knesset_number": objects["knesset_number"],
                "protocol_type": objects["protocol_type"],
                "protocol_number": objects["protocol_number"],
                "speaker_name": objects["speaker_name"],
                "sentence_text": objects["sentence_text"],
                "class": label
            } # surely the other class has more sentences
            data.append(row)
    df = pd.DataFrame(data)

    # df.to_csv(output_file, index=False, encoding='utf-8-sig')
    minimum_counter = df['class'].value_counts().min()
    df_downsampled = (
        df. groupby('class', group_keys= False)
        .apply(lambda x: x.sample(n = minimum_counter, random_state = 42))
    )
    print(df_downsampled['class'].value_counts()) # great, downsampled successfully
    print(f"Top Speakers:")
    print(f"1. {top_speakers[0][0]}: {top_speakers[0][1]} sentences")
    print(f"2. {top_speakers[1][0]}: {top_speakers[1][1]} sentences")
    print(f"Data saved to {output_file}")
