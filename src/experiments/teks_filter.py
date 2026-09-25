'''
TO RUN: VLLM BACKEND ################################################
    AVAILABLE MODELS:
        ✓ google/gemma-4-12B-it
        ✓ microsoft/phi-4
        ✓ google/gemma-4-31B-it
    
    # SET UP VLLM SERVER
    # 932374
    nohup env \
        VLLM_USE_FLASHINFER_SAMPLER=0 \
        CUDA_VISIBLE_DEVICES=1 \
        VLLM_USE_V2_MODEL_RUNNER=0 \
        vllm serve microsoft/phi-4 \
            --host 127.0.0.1 \
            --port 8002 \
            --max-model-len 40960 \
        > ./logs/vllm_server.log 2>&1 &
    tail -f ./logs/vllm_server.log 

    nohup env \
        VLLM_USE_FLASHINFER_SAMPLER=0 \
        vllm serve Qwen/Qwen3.5-9B \
            --host 127.0.0.1 \
            --port 8002 \
        > ./logs/vllm_server.log 2>&1 &
    tail -f ./logs/vllm_server.log 

    # 934383
    nohup python ./src/experiments/teks_filter.py \
        --input_data_path ./data/generated/sample.tsv \
        --backend vllm \
        --model Qwen/Qwen3.5-9B \
        --save_every 1 \
        > logs/gemma_teks_filter.log 2>&1 &

    tail -f logs/gemma_teks_filter.log

    
TO RUN: OPEN AI BACKEND #############################################
    python ./src/experiments/teks_filter.py \
        --input_data_path ./data/generated/sample.tsv \
        --backend openai \
        --model gpt-5.4-nano

    # 930510
    nohup python ./src/experiments/teks_filter.py \
        --input_data_path ./data/generated/sample.tsv \
        --backend openai \
        --model gpt-5.4-nano \
        > logs/gpt_teks_filter.log 2>&1 &

    tail -f logs/gpt_teks_filter.log
'''

import sys; sys.path.append("./src")
import os
import json
import argparse
import pandas as pd

from utils.file_io import (
    load_tsv_file, load_single_yaml_prompt, append_jsonl_file,
    save_args_as_config_json
)
from utils.prompting import generate
from utils.evaluation import (
    evaluate_generation, evaluate_dataset, print_evaluation, save_evaluation
)

TEKS_FILE = "data/cleaned/hecat_standards.tsv"
PROMPT_PATH = "data/prompts/response_generation/teks_guardrail.yaml"
OUTPUT_DIR = "data/generated/teks_filter_results"
VLLM_BASE_URL = "http://localhost:8002/v1"
ALL_CATS_SET = {"AOD", "FN", "MEH", "PA", "PHW", "S", "SH", "T", "V"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_data_path", type=str, required=True)
    p.add_argument("--backend", choices=["openai", "vllm"], required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--n_rows", type=int, default=-1)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--top_p", type=float, default=1.0)
    return p.parse_args()


def build_prompt(row: dict, teks_df: pd.DataFrame, prompt_template: dict) -> list[dict]:
    pretend_age = int(row["pretend_age"])
    blocked_cats = {x.strip() for x in str(row.get("blocked_cats", "") or "").split(",") if x.strip()}
    allowed_cats = ALL_CATS_SET - blocked_cats

    unknown_cats = blocked_cats - ALL_CATS_SET
    if unknown_cats:
        raise ValueError(f"Unknown blocked categories: {unknown_cats}")

    # Allowed = category allowed AND expectation appropriate for pretend age.
    allowed_teks = teks_df.loc[
        (teks_df["grade_code"] <= pretend_age) &
        teks_df["module_code"].isin(allowed_cats), "expectation"
    ].tolist()

    # Disallowed = category blocked OR expectation above pretend age.
    disallowed_teks = teks_df.loc[
        teks_df["module_code"].isin(blocked_cats) |
        (teks_df["grade_code"] > pretend_age), "expectation"
    ].tolist()

    allowed_text = "\n".join(f"- {x}" for x in allowed_teks) or "None"
    disallowed_text = "\n".join(f"- {x}" for x in disallowed_teks) or "None"

    return [
        {"role": "system", "content": prompt_template["system_prompt"].format(
            allowed_teks=allowed_text, disallowed_teks=disallowed_text)},
        {"role": "user", "content": prompt_template["user_prompt"].format(
            question=row["situation"])}
    ]


def build_messages(data: list[dict], teks_df: pd.DataFrame,
                   prompt_template: dict) -> list[list[dict]]:
    return [build_prompt(row, teks_df, prompt_template) for row in data]


def load_existing_outputs(output_path: str) -> dict[str, dict]:
    if not os.path.exists(output_path):
        return {}

    outputs = {}
    with open(output_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {output_path} on line {line_number}") from e

            if "id" not in row:
                raise ValueError(f"Existing output on line {line_number} has no ID.")
            outputs[row["id"]] = row

    return outputs


def main():
    args = parse_args()
    if args.save_every <= 0:
        raise ValueError("--save_every must be greater than 0.")

    data = load_tsv_file(args.input_data_path)
    teks_df = pd.DataFrame(load_tsv_file(TEKS_FILE))
    prompt_template = load_single_yaml_prompt(PROMPT_PATH)

    teks_df["grade_code"] = pd.to_numeric(teks_df["grade_code"])
    teks_df["module_code"] = teks_df["module_code"].astype(str).str.strip()

    if args.n_rows > 0:
        data = data[:args.n_rows]
    if not data:
        print("No rows found.")
        return

    required_cols = {"id", "situation", "pretend_age", "blocked_cats", "refusal_gt"}
    for i, row in enumerate(data, start=1):
        missing = required_cols - row.keys()
        if missing:
            raise ValueError(f"Input row {i} missing columns: {missing}")

    dataset_ids = [row["id"] for row in data]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("Duplicate IDs detected.")

    safe_model_name = args.model.replace("/", "--")
    dataset_name = os.path.splitext(os.path.basename(args.input_data_path))[0]
    run_dir = os.path.join(OUTPUT_DIR, safe_model_name, dataset_name)
    os.makedirs(run_dir, exist_ok=True)

    output_path = os.path.join(run_dir, "outputs.jsonl")
    eval_path = os.path.join(run_dir, "eval.json")
    save_args_as_config_json(args, output_path)

    existing_outputs = load_existing_outputs(output_path)
    selected_ids = set(dataset_ids)
    completed_outputs = {
        row_id: row for row_id, row in existing_outputs.items()
        if row_id in selected_ids
    }
    remaining_data = [row for row in data if row["id"] not in completed_outputs]

    print(f"Loaded:            {len(data)}")
    print(f"Already completed: {len(completed_outputs)}")
    print(f"Remaining:         {len(remaining_data)}")
    print(f"Save every:        {args.save_every}")

    example_printed = False

    for batch_start in range(0, len(remaining_data), args.save_every):
        batch_data = remaining_data[batch_start:batch_start + args.save_every]

        messages = build_messages(batch_data, teks_df, prompt_template)

        if not example_printed:
            print("\nExample message:")
            print(messages[0])
            example_printed = True

        generations = generate(
            messages=messages, model=args.model, backend=args.backend,
            vllm_base_url=VLLM_BASE_URL, temperature=args.temperature,
            max_tokens=args.max_tokens, top_p=args.top_p
        )

        if len(generations) != len(batch_data):
            raise ValueError("Number of generations does not match number of input rows.")

        batch_output = []
        for row, generation in zip(batch_data, generations):
            evaluation = evaluate_generation(row, generation)
            batch_output.append({**row, **evaluation})

        append_jsonl_file(output_path, batch_output)
        for row in batch_output:
            completed_outputs[row["id"]] = row

        current_output = [
            completed_outputs[row["id"]] for row in data
            if row["id"] in completed_outputs
        ]
        metrics = evaluate_dataset(current_output)
        save_evaluation(metrics, eval_path)
        print(f"Saved {len(current_output)} / {len(data)} rows.")

    final_output = [
        completed_outputs[row["id"]] for row in data
        if row["id"] in completed_outputs
    ]
    metrics = evaluate_dataset(final_output)
    print_evaluation(metrics)
    save_evaluation(metrics, eval_path)

    print("\n############################")
    print("FILES SAVED")
    print("############################")
    print(f"Run directory: {run_dir}")
    print(f"Outputs:       {output_path}")
    print(f"Evaluation:    {eval_path}")
    print(f"Completed:     {len(final_output)} / {len(data)}")


if __name__ == "__main__":
    main()