"""
2026-11-04
Upper-bound HECAT/TEKS filter: provide ONLY the ground-truth expectation.

Run (OpenAI):
    nohup python ./src/experiments/upper_bound_rag.py \
    --input_data_path ./data/splits/test.tsv \
    --query_row eduq \
    --backend openai \
    --model gpt-5.4-nano \
    > ./logs/upper_bound_rag_gpt.log 2>&1 &

Run (vLLM):
    python ./src/experiments/upper_bound_rag.py \
        --input_data_path ./data/splits/test.tsv \
        --query_row situation --backend vllm \
        --model google/gemma-4-12B-it --port 8001
"""

from pathlib import Path
import sys; sys.path.append("./src")
import argparse

import pandas as pd

from utils.file_io import (
    load_tsv_file, load_single_yaml_prompt, append_jsonl_file,
    save_args_as_config_json,
)
from utils.prompting import generate
from utils.evaluation import (
    evaluate_generation, evaluate_dataset, print_evaluation, save_evaluation,
)
from experiments.teks_filter_rag import build_prompt, load_existing_outputs, TEKS_FILE, PROMPT_PATH

OUTPUT_DIR = "data/generated/upper_bound_rag_results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_data_path", default="./data/splits/test.tsv")
    p.add_argument("--query_row", choices=["situation", "eduq"], required=True)
    p.add_argument("--backend", choices=["openai", "vllm"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--n_rows", type=int, default=-1)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--top_p", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.save_every <= 0:
        raise ValueError("--save_every must be positive.")

    data = load_tsv_file(args.input_data_path)
    teks_df = pd.DataFrame(load_tsv_file(TEKS_FILE))
    prompt_template = load_single_yaml_prompt(PROMPT_PATH)

    teks_df["expectation"] = teks_df["expectation"].astype(str).str.strip()
    teks_df["grade_code"] = pd.to_numeric(teks_df["grade_code"])
    teks_df["module_code"] = teks_df["module_code"].astype(str).str.strip()
    if teks_df.empty or teks_df["expectation"].eq("").any():
        raise ValueError("Expectation catalog is empty or contains blank expectations.")

    # Index the HECAT catalog once; each query will receive exactly one row.
    expectation_lookup = {
        expectation: group
        for expectation, group in teks_df.groupby("expectation", sort=False)
    }

    if args.n_rows > 0:
        data = data[:args.n_rows]
    if not data:
        print("No rows found.")
        return

    required_cols = {
        "id", args.query_row, "expectation", "pretend_age", "blocked_cats", "refusal_gt",
    }
    for i, row in enumerate(data, start=1):
        missing = required_cols - row.keys()
        if missing:
            raise ValueError(f"Input row {i} missing columns: {missing}")
        if not str(row[args.query_row]).strip():
            raise ValueError(f"Input row {i} has an empty {args.query_row}")
        if str(row["expectation"]).strip() not in expectation_lookup:
            raise ValueError(f"Input row {i} expectation not found in {TEKS_FILE}")

    dataset_ids = [row["id"] for row in data]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("Duplicate IDs detected.")

    run_dir = (
        Path(OUTPUT_DIR) / args.model.replace("/", "--") / args.query_row
        / Path(args.input_data_path).stem
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(run_dir / "outputs.jsonl")
    eval_path = str(run_dir / "eval.json")
    save_args_as_config_json(args, output_path)

    existing_outputs = load_existing_outputs(output_path)
    completed_outputs = {row_id: value for row_id, value in existing_outputs.items()
                         if row_id in set(dataset_ids)}
    remaining_data = [row for row in data if row["id"] not in completed_outputs]

    print(f"Query type:        {args.query_row}")
    print(f"Loaded:            {len(data)}")
    print("Context per query: 1 ground-truth expectation")
    print(f"Already completed: {len(completed_outputs)}")
    print(f"Remaining:         {len(remaining_data)}")
    print(f"Save every:        {args.save_every}")

    example_printed = False
    for batch_start in range(0, len(remaining_data), args.save_every):
        batch_data = remaining_data[batch_start:batch_start + args.save_every]
        contexts = []
        for row in batch_data:
            candidates = expectation_lookup[str(row["expectation"]).strip()]
            # Prefer an exact HECAT code match if expectation text is duplicated.
            if "code" in row and "code" in candidates.columns and str(row["code"]).strip():
                matched = candidates[candidates["code"].astype(str) == str(row["code"]).strip()]
                if not matched.empty:
                    candidates = matched
            contexts.append(candidates.iloc[[0]].copy())

        messages = [build_prompt(row, context, prompt_template, args.query_row)
                    for row, context in zip(batch_data, contexts, strict=True)]
        if not example_printed:
            print("\nExample message:")
            print(messages[0])
            example_printed = True

        generations = generate(
            messages=messages, model=args.model, backend=args.backend,
            vllm_base_url=f"http://127.0.0.1:{args.port}/v1",
            temperature=args.temperature, max_tokens=args.max_tokens, top_p=args.top_p,
        )
        if len(generations) != len(batch_data):
            raise ValueError("Number of generations does not match number of input rows.")

        batch_output = []
        for row, generation, context in zip(batch_data, generations, contexts, strict=True):
            evaluation = evaluate_generation(row, generation)
            batch_output.append({**row, **evaluation, "ground_truth_expectation": {
                "code": str(context.iloc[0].get("code", "")),
                "expectation": str(context.iloc[0]["expectation"]),
                "module_code": str(context.iloc[0]["module_code"]),
                "grade_code": int(context.iloc[0]["grade_code"]),
            }})

        append_jsonl_file(output_path, batch_output)
        for row in batch_output:
            completed_outputs[row["id"]] = row
        current_output = [completed_outputs[row["id"]]
                          for row in data if row["id"] in completed_outputs]
        save_evaluation(evaluate_dataset(current_output), eval_path)
        print(f"Saved {len(current_output)} / {len(data)} rows.")

    final_output = [completed_outputs[row["id"]]
                    for row in data if row["id"] in completed_outputs]
    if final_output:
        metrics = evaluate_dataset(final_output)
        print_evaluation(metrics)
        save_evaluation(metrics, eval_path)
    else:
        print("No completed outputs to evaluate.")

    print("\n############################")
    print("FILES SAVED")
    print("############################")
    print(f"Run directory: {run_dir}")
    print(f"Outputs:       {output_path}")
    print(f"Evaluation:    {eval_path}")
    print(f"Completed:     {len(final_output)} / {len(data)}")


if __name__ == "__main__":
    main()
