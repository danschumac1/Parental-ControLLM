"""
Measure the actual token lengths of the rendered TEKS prompts.

Run:

python ./src/experiments/inspect_teks_prompt_lengths.py \
    --input_data_path ./data/generated/sample.tsv \
    --model google/gemma-4-31B-it \
    --max_tokens 1024
"""

import sys
sys.path.append("./src")

import argparse
import statistics

import pandas as pd
from transformers import AutoProcessor

from utils.file_io import load_tsv_file, load_single_yaml_prompt
from experiments.teks_filter import (
    build_prompt,
    TEKS_FILE,
    PROMPT_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_data_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n_rows", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--top_n", type=int, default=10)

    return parser.parse_args()


def count_prompt_tokens(
    messages: list[dict],
    processor,
) -> int:

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    return inputs["input_ids"].shape[-1]

def percentile(values: list[int], q: float) -> int:
    values = sorted(values)

    index = int((len(values) - 1) * q)

    return values[index]


def main():
    args = parse_args()

    print(f"Loading processor: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)

    data = load_tsv_file(args.input_data_path)
    teks_df = pd.DataFrame(load_tsv_file(TEKS_FILE))
    prompt_template = load_single_yaml_prompt(PROMPT_PATH)

    teks_df["grade_code"] = pd.to_numeric(teks_df["grade_code"])
    teks_df["module_code"] = teks_df["module_code"].astype(str).str.strip()

    if args.n_rows > 0:
        data = data[:args.n_rows]

    results = []

    print(f"Measuring {len(data):,} prompts...\n")

    for i, row in enumerate(data, start=1):

        messages = build_prompt(
            row=row,
            teks_df=teks_df,
            prompt_template=prompt_template,
        )

        prompt_tokens = count_prompt_tokens(
            messages=messages,
            processor=processor,
        )

        total_tokens = prompt_tokens + args.max_tokens

        results.append(
            {
                "id": row["id"],
                "pretend_age": row["pretend_age"],
                "blocked_cats": row["blocked_cats"],
                "prompt_tokens": prompt_tokens,
                "max_output_tokens": args.max_tokens,
                "max_total_tokens": total_tokens,
            }
        )

        current_counts = [x["prompt_tokens"] for x in results]

        print(
            f"{i:>5,} / {len(data):,} | "
            f"tokens={prompt_tokens:>7,} | "
            f"min={min(current_counts):>7,} | "
            f"mean={statistics.mean(current_counts):>9,.1f} | "
            f"max={max(current_counts):>7,} | "
            f"age={row['pretend_age']} | "
            f"blocked={row['blocked_cats']}",
            flush=True,
        )

    token_counts = [x["prompt_tokens"] for x in results]

    print()
    print("########################################")
    print("PROMPT TOKEN STATISTICS")
    print("########################################")
    print(f"Prompts:       {len(token_counts):,}")
    print(f"Minimum:       {min(token_counts):,}")
    print(f"Mean:          {statistics.mean(token_counts):,.1f}")
    print(f"Median:        {statistics.median(token_counts):,.1f}")
    print(f"P90:           {percentile(token_counts, 0.90):,}")
    print(f"P95:           {percentile(token_counts, 0.95):,}")
    print(f"P99:           {percentile(token_counts, 0.99):,}")
    print(f"Maximum:       {max(token_counts):,}")

    max_prompt = max(token_counts)

    print()
    print("########################################")
    print("CONTEXT REQUIREMENTS")
    print("########################################")
    print(f"Maximum prompt:            {max_prompt:,}")
    print(f"Maximum output requested:  {args.max_tokens:,}")
    print(f"Required context:          {max_prompt + args.max_tokens:,}")

    print()
    print("########################################")
    print(f"TOP {args.top_n} LONGEST PROMPTS")
    print("########################################")

    longest = sorted(
        results,
        key=lambda x: x["prompt_tokens"],
        reverse=True,
    )[:args.top_n]

    for row in longest:
        print(
            f'{row["prompt_tokens"]:>7,} tokens | '
            f'age={row["pretend_age"]} | '
            f'blocked={row["blocked_cats"]} | '
            f'id={row["id"]}'
        )

    print()
    print("########################################")
    print("BY PRETEND AGE")
    print("########################################")

    df = pd.DataFrame(results)

    age_summary = (
        df.groupby("pretend_age")["prompt_tokens"]
        .agg(["count", "min", "mean", "median", "max"])
        .round(1)
    )

    print(age_summary.to_string())


if __name__ == "__main__":
    main()