'''
2026-07-22
How to run:
   python ./src/refusal_prompting.py \
        --backend openai \
        --model gpt-4o-mini

   python ./src/refusal_prompting.py \
        --backend vllm \
        --model Qwen/Qwen2.5-7B-Instruct \
'''

import os
from typing import Any
import argparse
import json
import random

from utilsOLD.prompters import ChatPrompt, build_prompter, construct_chat_prompts
from utils.file_io import load_tsv_file, load_single_yaml_prompt

INPUT_DATA_PATH = "./data/generated/Qwen__Qwen2.5-7B-Instruct__sample_all__synthetic_child_qa_dataset.tsv"
PROMPT_TEMPLATE_PATH = "./data/prompts/simple.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["openai", "vllm"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    return parser.parse_args()

def main():
    args = parse_args()

    # Load TSV data
    data = load_tsv_file(INPUT_DATA_PATH)

    # Shuffle for reproducibility
    random.seed(42)
    random.shuffle(data)

    current_codes = [
        "AOD",
        "FN",
        "MEH",
        "PA",
        "PHW",
        "S",
        "SH",
        "T",
        "V",
    ]

    data_dict = {}
    for code in current_codes:
        data_dict[code] = [row for row in data if row["module_code"] == code]

    ten_each_data = []
    for key, rows in data_dict.items():
        ten_each_data.extend(rows[:10])

    data = ten_each_data

    # load prompt template
    prompt_template = load_single_yaml_prompt(PROMPT_TEMPLATE_PATH)

    # construct chat prompts
    data = construct_chat_prompts(
        data=data,
        prompt_template=prompt_template,
        user_dict={
            "question": "generated_question",
        },
        sys_dict={} # no formatting needed for system message
    )


    prompter = build_prompter(
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    print("Constructed chat prompts for each row:")

    messages = [
        row[f"chat_prompt"].to_messages()
        for row in data
    ]

    print(f"Generating responses using {args.backend} backend...")
    print(f"Example Message: {messages[0]}")

    results = prompter.generate(
        prompts=messages,
    )

    out_data = []

    # source_file	
    # pdf_page	
    # module	
    # module_code	
    # standard_number	
    # standard_name	
    # expectation_type	
    # grade_code	
    # grade_span	
    # item_number	
    # code	
    # expectation	
    # hbo_codes	scenario_type	
    # situation	
    # generated_question	
    # generated_answer


    for row, result in zip(data, results):
        line = {
            "item_number": row["item_number"],
            "module_code": row["module_code"],
            "grade_span": row["grade_span"],
            "question": row["generated_question"],
            "answer": result,
        }
        out_data.append(line)

    os.makedirs("./data/generated/answers", exist_ok=True)

    output_path = f"./data/generated/answers/{args.model}.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for line in out_data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(
        f"Finished generating responses using {args.backend} backend. "
        f"Results saved to {output_path}"
    )


if __name__ == "__main__":
    main()