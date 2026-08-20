'''
2026-08-19

OpenAI:
nohup python ./src/prompt_with_expectation.py \
    --query_column expectation \
    --backend openai \
    --model gpt-5-nano \
    --module_codes all \
    --max_tokens 4096 \
    > prompt_with_expectation_openai.log 2>&1 &

# TO KILL: 1881611
    
Local vLLM:

TERMINAL 1:
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --port 8002
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --port 8002 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90

# later to kill ---
# kill 1879466

TERMINAL 2:
nohup python ./src/prompt_with_expectation.py \
    --query_column expectation \
    --backend vllm \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --n_rows -1 \
    --module_codes all \
    --temperature 0.7 \
    > prompt_with_expectation.log 2>&1 &

# TO KILL: 1880766

'''

import os
import argparse
import json
import random


from utils.file_io import (
    append_jsonl_file, load_tsv_file, load_single_yaml_prompt, remove_completed_items,
    save_args_as_config_json)
from utils.prompting import construct_messages, generate


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
    parser.add_argument("--backend", choices=["openai", "vllm"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n_rows", type=int, default=-1)
    parser.add_argument(
        "--module_codes",
        nargs="+",
        default=None,
        choices=["all", "AOD", "FN", "MEH", "PA", "PHW", "S", "SH", "T", "V"],
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=0.9)

    return parser.parse_args()


def filter_data(
    data,
    module_codes,
    n_rows,
):
    random.shuffle(data)

    if module_codes is None or "all" in module_codes:
        module_codes = sorted({row["module_code"] for row in data})

    return [
        row for code in module_codes
        for row in [
            r for r in data
            if r["module_code"] == code
        ][:n_rows if n_rows > 0 else None]
    ]


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

    data = remove_completed_items(
        output_path,
        idx_key="item_number"
    )

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
        user_map={"question": args.query_column},
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
        vllm_base_url=VLLM_BASE_URL,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )

    # -------------------------
    # SAVE
    # -------------------------
    out_data = [{**row, "answer": result,} for row, result in zip(data, results)]
    append_jsonl_file(output_path,out_data)    
    save_args_as_config_json(args,output_path,)
    print(f"Results saved to {output_path}")



if __name__ == "__main__":
    main()