'''
TO RUN:
python ./src/llm_as_a_judge.py \
    --input_data_path ./data/generated/answers/gpt-5-nano__expectation.jsonl \
    --prompt_template_path ./data/prompts/llm_as_a_judge/refusal.yaml \
    --query_col expectation \
    --response_col answer \
    --judge_model gpt-5-nano \
    --backend openai
'''



import os
import argparse

from utils.file_io import (
    append_jsonl_file,
    load_jsonl_file,
    load_single_yaml_prompt,
    remove_completed_items,
    save_args_as_config_json,
)
from utils.prompting import construct_messages, generate



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_data_path", required=True)
    parser.add_argument("--prompt_template_path", required=True)

    parser.add_argument("--query_col", required=True)
    parser.add_argument("--response_col", required=True)

    parser.add_argument("--judge_model", required=True)
    parser.add_argument("--backend", choices=["openai", "vllm"], required=True)
    parser.add_argument("--vllm_base_url", default="")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument(
        "--output_dir",
        default="./data/generated/llmaj_refusal",
    )

    return parser.parse_args()



def main():
    args = parse_args()

    # -------------------------
    # Determine Output path
    # -------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    safe_model_name = args.judge_model.replace("/", "--")

    output_path = (
        f"{args.output_dir}/"
        f"{safe_model_name}__{args.query_col}.jsonl"
    )

    # -------------------------
    # Load Data With Resume
    # -------------------------
    data = load_jsonl_file(args.input_data_path)
    data = remove_completed_items(data,output_path,idx_key="item_number")

    if not data:
        print("All requested items have already been completed.")
        return

    # -------------------------
    # Set up prompts
    # -------------------------
    prompt_template = load_single_yaml_prompt(args.prompt_template_path)

    messages = construct_messages(
        data,
        prompt_template,
        user_map={
            "query": args.query_col,
            "response": args.response_col,
        },
    )

    print(f"Example message: {messages[0]}")

    # -------------------------
    # Generation
    # -------------------------
    results = generate(
        messages=messages,
        model=args.judge_model,
        backend=args.backend,
        vllm_base_url=args.vllm_base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )

    # -------------------------
    # Save
    # -------------------------
    out_data = [
        {**row,"refusal_judgement": result}
        for row, result in zip(data, results)
    ]

    append_jsonl_file(output_path, out_data)
    save_args_as_config_json(args, output_path)

    print(f"Results saved to {output_path}")



if __name__ == "__main__":
    main()