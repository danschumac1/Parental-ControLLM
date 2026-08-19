'''
2026-08-13
How to run:
   python ./src/convert_jsonl_to_tsv.py
'''

import json
import csv

JSONL_INPUT_PATH = "./data/generated/answers/gpt-4o-mini.jsonl"

def main():
    out_path = JSONL_INPUT_PATH.replace(".jsonl", ".tsv")
    data = []
    with open(JSONL_INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys(), delimiter="\t")
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
    print(f"Converted {len(data)} to {out_path}")

if __name__ == "__main__":
    main()