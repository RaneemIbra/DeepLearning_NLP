import sys
import json
import pandas as pd

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_jsonl_file> <output_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Set to store unique speaker names
    unique_speakers = set()

    # Read the JSONL file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            speaker_name = record.get('speaker_name', '').strip()
            if speaker_name:
                unique_speakers.add(speaker_name)

    # Convert the set to a DataFrame
    df = pd.DataFrame(list(unique_speakers), columns=['Speaker Name'])

    # Save the unique speaker names to a CSV file
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"Extracted {len(unique_speakers)} unique speaker names.")
    print(f"Unique speaker names saved to {output_file}")