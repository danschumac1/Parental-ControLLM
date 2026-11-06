
"""
2026-11-05

Generate missing EduQs for one model at a time.

Matches situations and EduQs using:
    (HECAT code, normalized expectation text)

Uses the original EduQ generation prompt and model functions.
Appends only missing generations to existing JSONL files.

Run:
    python ./src/data_management/generate_missing_eduqs.py \
        --backend openai \
        --model gpt-5.4-nano

    python ./src/data_management/generate_missing_eduqs.py \
        --backend vllm \
        --model google/gemma-4-12B-it \
        --port 8002
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, "./src")

from prompt_for_queries import (
    create_client,
    format_instruct_prompt,
    generate_instruct,
    clean_generation,
)
from utils.file_io import load_instruct_yaml


# =============================================================================
# CONFIGURATION
# =============================================================================

SITUATIONS_DIR = Path("data/generated/queries/situations")
EDUQ_DIR = Path("data/generated/queries/eduQs")

PROMPT_PATH = Path(
    "data/prompts/eduQ_generation/instruct.yaml"
)

MAX_EXPECTATION_LENGTH = 300
BAD_EXPECTATION_PHRASES = ["Knowledge Expectations Grades"]

QUESTION_COLUMNS = (
    "eduQ",
    "question",
    "generated_question",
    "educational_question",
)


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--backend",
        choices=["openai", "vllm"],
        required=True,
    )

    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8002)

    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_tokens", type=int, default=100)
    p.add_argument("--top_p", type=float, default=0.95)

    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--dry_run", action="store_true")

    args = p.parse_args()

    args.vllm_base_url = f"http://127.0.0.1:{args.port}/v1"

    return args


# =============================================================================
# HELPERS
# =============================================================================

def normalize(value):
    return " ".join(str(value or "").split())


def expectation_key(row):
    return (
        normalize(row["code"]),
        normalize(row["expectation"]),
    )


def valid_expectation(row):
    expectation = normalize(row.get("expectation"))

    return (
        bool(expectation)
        and len(expectation) <= MAX_EXPECTATION_LENGTH
        and not any(
            phrase.lower() in expectation.lower()
            for phrase in BAD_EXPECTATION_PHRASES
        )
    )


def get_question(row):
    for col in QUESTION_COLUMNS:
        value = row.get(col)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def load_jsonl(path):
    if not path.exists():
        return []

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON: {path}:{line_number}"
                ) from e

            rows.append(row)

    return rows


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(row, ensure_ascii=False) + "\n"
        )
        f.flush()


# =============================================================================
# FIND MISSING EDUQS
# =============================================================================

def find_missing(situations, eduqs):
    """Find expectations without a generated EduQ."""

    existing = {
        expectation_key(row)
        for row in eduqs
        if valid_expectation(row) and get_question(row)
    }

    missing = {}

    for row in situations:
        if not valid_expectation(row):
            continue

        key = expectation_key(row)

        if key not in existing:
            missing[key] = row

    return list(missing.values())


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    safe_model = args.model.replace("/", "__")

    situation_path = (
        SITUATIONS_DIR / f"{safe_model}.jsonl"
    )

    eduq_path = (
        EDUQ_DIR / f"{safe_model}.jsonl"
    )

    if not situation_path.exists():
        raise FileNotFoundError(situation_path)

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(PROMPT_PATH)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    situations = load_jsonl(situation_path)
    eduqs = load_jsonl(eduq_path)

    missing = find_missing(situations, eduqs)

    print("\n==============================================")
    print("GENERATING MISSING EDUQS")
    print("==============================================")

    print(f"Model:               {args.model}")
    print(f"Backend:             {args.backend}")
    print(f"Situations:          {len(situations):,}")
    print(f"Existing EduQs:      {len(eduqs):,}")
    print(f"Missing expectations: {len(missing):,}")
    print(f"Output:              {eduq_path}")

    if not missing:
        print("\nNothing to generate.")
        return

    print("\nMissing expectations:")

    for row in missing:
        print(
            f"  {row['code']}: "
            f"{normalize(row['expectation'])}"
        )

    if args.dry_run:
        print("\nDry run complete. No generations made.")
        return

    if args.limit > 0:
        missing = missing[:args.limit]

    # -------------------------------------------------------------------------
    # Initialize
    # -------------------------------------------------------------------------

    prompt_data = load_instruct_yaml(
        str(PROMPT_PATH)
    )

    client = create_client(args)

    generated = 0
    failed = 0

    # -------------------------------------------------------------------------
    # Generate missing questions
    # -------------------------------------------------------------------------

    for row in tqdm(
        missing,
        desc="Generating EduQs",
        unit="question",
    ):
        code = row["code"]

        try:
            messages = format_instruct_prompt(
                prompt_data,
                row,
            )

            generation = generate_instruct(
                backend=args.backend,
                client=client,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
            )

            generation = clean_generation(generation)

            if not generation:
                raise ValueError("Empty generation")

            # Preserve original metadata and situation-independent fields.
            output_row = {
                k: v
                for k, v in row.items()
                if k != "situation"
            }

            output_row["eduQ"] = generation

            # Append immediately so interrupted runs can resume.
            append_jsonl(
                eduq_path,
                output_row,
            )

            generated += 1

            tqdm.write(
                f"Generated {code}: {generation}"
            )

        except Exception as e:
            failed += 1
            tqdm.write(
                f"ERROR {code}: {e}"
            )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    remaining = find_missing(
        situations,
        load_jsonl(eduq_path),
    )

    print("\n==============================================")
    print("GENERATION SUMMARY")
    print("==============================================")

    print(f"Generated: {generated:,}")
    print(f"Failed:    {failed:,}")
    print(f"Remaining: {len(remaining):,}")
    print(f"Output:    {eduq_path}")


if __name__ == "__main__":
    main()
