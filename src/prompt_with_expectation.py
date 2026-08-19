'''
2026-08-19

OpenAI:
python ./src/prompt_with_expectation.py \
    --query_column expectation \
    --backend openai \
    --model gpt-5-nano \
    --module_codes all \
    --max_tokens 4096

Local vLLM:

TERMINAL 1:
# vllm serve Qwen/Qwen2.5-1.5B-Instruct \
#     --port 8002
vllm serve Qwen/Qwen3.8-27B \
    --port 8002 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90


TERMINAL 2:
python ./src/prompt_with_expectation.py \
    --query_column expectation \
    --backend vllm \
    --model Qwen/Qwen3.8-27B \
    --n_rows 5 \
    --module_codes SH \
    --temperature 0.0
'''

import os
import argparse
import json
import random

import dotenv
from openai import OpenAI
from tqdm import tqdm

from utils.file_io import load_tsv_file, load_single_yaml_prompt


INPUT_DATA_PATH = "data/cleaned/hecat_standards.tsv"
PROMPT_TEMPLATE_PATH = "./data/prompts/simple.yaml"

VLLM_BASE_URL = "http://localhost:8002/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query_column",
        choices=["generated_question", "expectation"],
        required=True,
    )

    parser.add_argument(
        "--backend",
        choices=["openai", "vllm"],
        required=True,
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--n_rows",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--module_codes",
        nargs="+",
        default=None,
        choices=[
            "all",
            "AOD",
            "FN",
            "MEH",
            "PA",
            "PHW",
            "S",
            "SH",
            "T",
            "V",
        ],
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )

    return parser.parse_args()


def save_config(args, output_path):
    config = {
        "query_column": args.query_column,
        "backend": args.backend,
        "model": args.model,
        "n_rows": args.n_rows,
        "module_codes": args.module_codes,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
    }

    config_path = output_path.replace(
        ".jsonl",
        "_config.json",
    )

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=4,
        )


def load_completed_items(output_path):
    if not os.path.exists(output_path):
        return set()

    completed_items = set()

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            completed_items.add(
                row["item_number"]
            )

    return completed_items


def filter_data(
    data,
    module_codes,
    n_rows,
):
    random.shuffle(data)

    if module_codes is None or "all" in module_codes:
        module_codes = sorted(
            {
                row["module_code"]
                for row in data
            }
        )

    return [
        row
        for code in module_codes
        for row in [
            r
            for r in data
            if r["module_code"] == code
        ][:n_rows if n_rows > 0 else None]
    ]


def construct_messages(
    data,
    prompt_template,
    query_column,
):
    messages = []

    for row in data:
        messages.append(
            [
                {
                    "role": "system",
                    "content": prompt_template[
                        "system_prompt"
                    ],
                },
                {
                    "role": "user",
                    "content": prompt_template[
                        "user_prompt"
                    ].format(
                        question=row[
                            query_column
                        ]
                    ),
                },
            ]
        )

    return messages


def generate_openai(
    messages,
    model,
    max_tokens,
):
    dotenv.load_dotenv(
        "./resources/.env"
    )

    client = OpenAI(
        api_key=os.environ[
            "OPENAI_API_KEY"
        ],
    )

    results = []

    for prompt in tqdm(messages):
        response = client.responses.create(
            model=model,
            input=prompt,
            reasoning={
                "effort": "low",
            },
            max_output_tokens=max_tokens,
        )

        results.append(
            response.output_text.strip()
        )

    return results


def generate_vllm(
    messages,
    model,
    temperature,
    max_tokens,
    top_p,
):
    client = OpenAI(
        api_key="EMPTY",
        base_url=VLLM_BASE_URL,
    )

    results = []

    for prompt in tqdm(messages):
        response = client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        results.append(
            response.choices[0]
            .message.content
            .strip()
        )

    return results


def generate(
    messages,
    model,
    backend,
    temperature,
    max_tokens,
    top_p,
):
    assert backend in [
        "openai",
        "vllm",
    ], f"Unknown backend: {backend}"

    if backend == "openai":
        return generate_openai(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
        )

    if backend == "vllm":
        return generate_vllm(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )


def main():
    args = parse_args()

    random.seed(42)

    # -------------------------
    # Output path
    # -------------------------

    output_dir = "./data/generated/answers"

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    safe_model_name = args.model.replace(
        "/",
        "--",
    )

    output_path = (
        f"{output_dir}/"
        f"{safe_model_name}"
        f"__{args.query_column}.jsonl"
    )

    # -------------------------
    # Load and filter data
    # -------------------------

    data = load_tsv_file(
        INPUT_DATA_PATH
    )

    data = filter_data(
        data,
        args.module_codes,
        args.n_rows,
    )

    # -------------------------
    # Resume logic
    # -------------------------

    completed_items = load_completed_items(
        output_path
    )

    if completed_items:
        print(
            f"Found {len(completed_items)} "
            "completed items."
        )

        print(
            "Removing completed items "
            "from this run."
        )

    data = [
        row
        for row in data
        if row["item_number"]
        not in completed_items
    ]

    if not data:
        print(
            "All requested items "
            "have already been completed."
        )
        return

    print(
        f"{len(data)} items remaining."
    )

    # -------------------------
    # Construct prompts
    # -------------------------

    prompt_template = (
        load_single_yaml_prompt(
            PROMPT_TEMPLATE_PATH
        )
    )

    messages = construct_messages(
        data,
        prompt_template,
        args.query_column,
    )

    print(
        f"Example message:\n"
        f"{messages[0]}"
    )

    # -------------------------
    # Generate
    # -------------------------

    results = generate(
        messages=messages,
        model=args.model,
        backend=args.backend,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )

    # -------------------------
    # Build output rows
    # -------------------------

    out_data = [
        {
            "item_number": row["item_number"],
            "module_code": row["module_code"],
            "grade_span": row["grade_span"],
            args.query_column: row[
                args.query_column
            ],
            "answer": result,
        }
        for row, result in zip(
            data,
            results,
        )
    ]

    # -------------------------
    # Append results
    # -------------------------

    with open(
        output_path,
        "a",
        encoding="utf-8",
    ) as f:
        for line in out_data:
            f.write(
                json.dumps(
                    line,
                    ensure_ascii=False,
                )
                + "\n"
            )

    save_config(
        args,
        output_path,
    )

    print(
        f"Results saved to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()