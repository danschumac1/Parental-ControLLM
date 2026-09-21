"""
Prepare the fixed annotation dataset.

Run:
    python src/apps/annotate_situation/prepare_data.py
"""

from pathlib import Path
import hashlib
import json
import random

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data/generated/situations"
OUT_PATH = DATA_DIR / "annotation_sample_1500.jsonl"

SOURCE_FILES = {
    "google__gemma-4-12B-it": DATA_DIR / "google__gemma-4-12B-it.jsonl",
    "gpt-5.4-nano": DATA_DIR / "gpt-5.4-nano.jsonl",
    "microsoft__phi-4": DATA_DIR / "microsoft__phi-4.jsonl",
}

N_PER_MODEL = 500
SEED = 42


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_id(model, row_idx, row):
    text = f"{model}|{row_idx}|{row.get('code')}|{row.get('situation')}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def main():
    pair_1, pair_2 = [], []

    for model, path in SOURCE_FILES.items():
        rows = load_jsonl(path)

        if len(rows) < N_PER_MODEL:
            raise ValueError(f"{model} only has {len(rows)} rows.")

        rng = random.Random(f"{SEED}:{model}")
        indices = rng.sample(range(len(rows)), N_PER_MODEL)

        selected = []
        for row_idx in indices:
            row = dict(rows[row_idx])
            row.update({
                "sample_id": make_id(model, row_idx, row),
                "generator_model": model,
                "source_jsonl": path.name,
                "source_row_index": row_idx,
            })
            selected.append(row)

        # 250 examples/model for each annotation pair
        pair_1.extend(selected[:250])
        pair_2.extend(selected[250:])

    random.Random(SEED).shuffle(pair_1)
    random.Random(SEED + 1).shuffle(pair_2)

    final = []

    for pair_id, rows in [("pair_1", pair_1), ("pair_2", pair_2)]:
        for pair_index, row in enumerate(rows, start=1):
            row["pair_id"] = pair_id
            row["pair_index"] = pair_index
            final.append(row)

    for i, row in enumerate(final, start=1):
        row["sample_index"] = i

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in final:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(final)} rows to {OUT_PATH}")
    print(f"Pair 1: {len(pair_1)} rows")
    print(f"Pair 2: {len(pair_2)} rows")


if __name__ == "__main__":
    main()