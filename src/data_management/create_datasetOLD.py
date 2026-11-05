"""
Create paired 5,000-row situation and educational-question datasets.

Run:
    python ./src/create_dataset.py

Outputs:
    data/generated/situations.tsv
    data/generated/eduQ.tsv

The existing sample.tsv is never modified.

Both datasets share the same expectations, parental configurations,
pretend ages, and ground-truth labels.
"""
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

SAMPLE_PATH = Path("data/generated/sample.tsv")
EDUQ_DIR = Path("data/generated/eduQ_generation")
SITUATIONS_PATH = Path("data/generated/situations.tsv")
EDUQ_DIR = Path("data/generated/queries/eduQs")
N_SAMPLE_ROWS = 5000
SEED = 42
QUESTION_COLUMNS = ("eduQ", "question", "generated_question", "educational_question", "situation")
MATCH_COLUMNS = ("code",)

def load_tsv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def save_tsv(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in data for key in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(data)


def normalize(value) -> str:
    return " ".join(str(value or "").split())


def expectation_key(row: dict) -> tuple[str, str, str]:
    return tuple(normalize(row.get(col)) for col in MATCH_COLUMNS)


def get_model_name(row: dict) -> str:
    return normalize(row.get("situation_gen_model_name") or row.get("generation_model_name"))


def get_question(row: dict) -> str:
    for col in QUESTION_COLUMNS:
        value = row.get(col)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"No educational question found. Available columns: {list(row)}")


def load_eduq() -> dict:
    index = defaultdict(list)
    files = sorted(EDUQ_DIR.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files in {EDUQ_DIR}")

    for path in files:
        model_name = path.stem.replace("__", "/")
        count = 0

        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON: {path}:{line_number}") from e

                question = get_question(row)
                key = expectation_key(row)

                if not all(key):
                    raise ValueError(f"Missing expectation metadata: {path}:{line_number}")

                row["_eduq_id"] = f"eduQ__{path.stem}__{line_number:06d}"
                row["_eduq_model"] = model_name
                row["_eduq_question"] = question
                index[key].append(row)
                count += 1

        print(f"Loaded {count:>5} eduQ rows from {path.name}")

    return index


def match_eduq(situations: list[dict], eduq_index: dict) -> list[dict]:
    rng = random.Random(SEED)
    matches = []
    missing = []
    fallback_models = 0
    usage = defaultdict(int)

    for situation in situations:
        key = expectation_key(situation)
        candidates = eduq_index.get(key, [])

        if not candidates:
            missing.append((situation["id"], key))
            continue

        model_name = get_model_name(situation)
        same_model = [row for row in candidates if row["_eduq_model"] == model_name]

        if same_model:
            candidates = same_model
        else:
            fallback_models += 1

        # Rotate through questions for a given expectation/model rather
        # than selecting the same educational question for every variant.
        usage_key = (key, model_name)
        index = usage[usage_key] % len(candidates)
        usage[usage_key] += 1
        eduq = candidates[index]
        if normalize(situation["expectation"]) != normalize(eduq["expectation"]):
            print(
                f'WARNING | {situation["id"]} | {key} | '
                f'Expectation text differs between sources.',
                flush=True
            )

        paired = {
            **situation,
            "id": f"eduQ__{situation['id']}",
            "situation": eduq["_eduq_question"],
            "source_type": "eduQ",
            "eduq_source_id": eduq["_eduq_id"],
            "eduq_gen_model_name": eduq["_eduq_model"],
        }
        matches.append(paired)

    if missing:
        print(f"\nERROR: {len(missing)} situation rows have no matching eduQ.")
        for row_id, key in missing[:10]:
            print(f"  {row_id} | {key}")
        raise ValueError("Cannot create a fully paired dataset. Missing eduQ expectations.")

    print(f"Matched eduQ rows:       {len(matches)}")
    print(f"Different-model matches: {fallback_models}")
    return matches


def validate_pairs(situations: list[dict], eduq: list[dict]) -> None:
    if len(situations) != N_SAMPLE_ROWS or len(eduq) != N_SAMPLE_ROWS:
        raise ValueError("Both datasets must contain exactly 5,000 rows.")

    shared = [
        "expectation", "module_code", "grade_code", "pretend_age",
        "pretend_age_type", "config_type", "config",
        "blocked_cats", "refusal_gt"
    ]

    for i, (s, e) in enumerate(zip(situations, eduq), 1):
        for col in shared:
            if normalize(s.get(col)) != normalize(e.get(col)):
                raise ValueError(f"Pair {i}: mismatch in {col}")

        if s["id"] == e["id"]:
            raise ValueError(f"Pair {i}: duplicate IDs")

    ids = [row["id"] for row in eduq]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate eduQ IDs detected.")

    print("Validation passed: all 5,000 pairs share experimental conditions.")


def main():
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(SAMPLE_PATH)

    situations = load_tsv(SAMPLE_PATH)
    print(f"Existing sample rows: {len(situations)}")

    if len(situations) != N_SAMPLE_ROWS:
        raise ValueError(f"Expected {N_SAMPLE_ROWS} existing sample rows.")

    ids = [row["id"] for row in situations]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in existing sample.")

    eduq_index = load_eduq()
    eduq = match_eduq(situations, eduq_index)
    validate_pairs(situations, eduq)

    # Copy existing situation sample without modifying its contents.
    if SAMPLE_PATH.resolve() != SITUATIONS_PATH.resolve():
        shutil.copy2(SAMPLE_PATH, SITUATIONS_PATH)

    save_tsv(eduq, EDUQ_PATH)

    print(f"\nSituations: {SITUATIONS_PATH} ({len(situations)} rows)")
    print(f"EduQ:       {EDUQ_PATH} ({len(eduq)} rows)")
    print(f"Original:   {SAMPLE_PATH} (unchanged)")


if __name__ == "__main__":
    main()