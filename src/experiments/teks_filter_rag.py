"""
2026-11-04
HECAT/TEKS filter with per-query cosine-similarity retrieval.

Run (vLLM):
    python ./src/experiments/teks_filter_rag.py \
        --input_data_path ./data/splits/test.tsv \
        --query_row situation --retrieval_corpus split --backend vllm \
        --model google/gemma-4-12B-it --port 8001 --n_r_contexts 20

Run (OpenAI):
    python ./src/experiments/teks_filter_rag.py \
        --input_data_path ./data/splits/test.tsv \
        --query_row situation --retrieval_corpus split --backend openai \
        --model gpt-5.4-nano --n_r_contexts 20

Retrieval corpus: full HECAT catalog or unique expectations in query split.
Each query gets its own top-K expectations. Existing age and parental-policy
filters are applied only AFTER retrieval.
"""

from pathlib import Path
import sys; sys.path.append("./src")
import os
import json
import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from utils.file_io import (
    load_tsv_file, load_single_yaml_prompt, append_jsonl_file,
    save_args_as_config_json,
)
from utils.prompting import generate
from utils.evaluation import (
    evaluate_generation, evaluate_dataset, print_evaluation, save_evaluation,
)

TEKS_FILE = "data/cleaned/hecat_standards.tsv"
PROMPT_PATH = "data/prompts/response_generation/teks_guardrail.yaml"
OUTPUT_DIR = "data/generated/teks_filter_rag_results"
ALL_CATS_SET = {"AOD", "FN", "MEH", "PA", "PHW", "S", "SH", "T", "V"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_data_path", type=str, default="./data/splits/test.tsv")
    p.add_argument("--query_row", choices=["situation", "eduq"], required=True)
    p.add_argument("--retrieval_corpus", choices=["full", "split"], default="full")
    p.add_argument("--backend", choices=["openai", "vllm"], required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--n_rows", type=int, default=-1)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--n_r_contexts", type=int, default=20)
    p.add_argument(
        "--embedding_model", type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    return p.parse_args()


class ExpectationRetriever:
    """Embed the selected catalog once; retrieve the top-K for each query."""

    def __init__(self, teks_df: pd.DataFrame, model_name: str):
        self.teks_df = teks_df.reset_index(drop=True)
        self.model = SentenceTransformer(model_name)
        self.embeddings = np.asarray(self.model.encode(
            self.teks_df["expectation"].astype(str).tolist(),
            normalize_embeddings=True,
            show_progress_bar=True,
        ))

    def retrieve_batch(self, queries: list[str], top_k: int) -> list[pd.DataFrame]:
        query_embeddings = np.asarray(self.model.encode(
            queries,
            normalize_embeddings=True,
            show_progress_bar=False,
        ))
        scores = query_embeddings @ self.embeddings.T
        k = min(top_k, len(self.teks_df))

        retrieved_dfs = []
        for query_scores in scores:
            indices = np.argsort(-query_scores, kind="stable")[:k]
            retrieved = self.teks_df.iloc[indices].copy()
            retrieved["similarity"] = query_scores[indices]
            retrieved_dfs.append(retrieved)
        return retrieved_dfs


def build_prompt(
    row: dict,
    teks_df: pd.DataFrame,
    prompt_template: dict,
    query_row: str,
) -> list[dict]:
    """The original filter logic, but teks_df is now query-specific Top-K."""
    pretend_age = int(row["pretend_age"])

    blocked_cats = {
        x.strip()
        for x in str(row.get("blocked_cats", "") or "").split(",")
        if x.strip()
    }
    allowed_cats = ALL_CATS_SET - blocked_cats

    unknown_cats = blocked_cats - ALL_CATS_SET
    if unknown_cats:
        raise ValueError(f"Unknown blocked categories: {unknown_cats}")

    # Allowed = category allowed AND expectation appropriate for age.
    allowed_teks = teks_df.loc[
        (teks_df["grade_code"] <= pretend_age)
        & teks_df["module_code"].isin(allowed_cats),
        "expectation",
    ].tolist()

    # Disallowed = category blocked OR expectation above age.
    disallowed_teks = teks_df.loc[
        teks_df["module_code"].isin(blocked_cats)
        | (teks_df["grade_code"] > pretend_age),
        "expectation",
    ].tolist()

    allowed_text = "\n".join(f"- {x}" for x in allowed_teks) or "None"
    disallowed_text = "\n".join(f"- {x}" for x in disallowed_teks) or "None"

    return [
        {
            "role": "system",
            "content": prompt_template["system_prompt"].format(
                allowed_teks=allowed_text,
                disallowed_teks=disallowed_text,
            ),
        },
        {
            "role": "user",
            "content": prompt_template["user_prompt"].format(
                question=row[query_row],
            ),
        },
    ]


def build_messages(
    data: list[dict],
    retrieved_dfs: list[pd.DataFrame],
    prompt_template: dict,
    query_row: str,
) -> list[list[dict]]:
    return [
        build_prompt(row, retrieved_df, prompt_template, query_row)
        for row, retrieved_df in zip(data, retrieved_dfs, strict=True)
    ]


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
                raise ValueError(
                    f"Invalid JSON in {output_path} on line {line_number}"
                ) from e
            if "id" not in row:
                raise ValueError(
                    f"Existing output on line {line_number} has no ID."
                )
            outputs[row["id"]] = row
    return outputs


def main():
    args = parse_args()
    if args.save_every <= 0 or args.n_r_contexts <= 0:
        raise ValueError("--save_every and --n_r_contexts must be positive.")

    data = load_tsv_file(args.input_data_path)
    teks_df = pd.DataFrame(load_tsv_file(TEKS_FILE))
    prompt_template = load_single_yaml_prompt(PROMPT_PATH)

    if args.retrieval_corpus == "split":
        split_expectations = {str(row["expectation"]).strip() for row in data}
        teks_df["expectation"] = teks_df["expectation"].astype(str).str.strip()
        teks_df = teks_df[
            teks_df["expectation"].isin(split_expectations)
        ].drop_duplicates(subset=["expectation"]).copy()

    teks_df["grade_code"] = pd.to_numeric(teks_df["grade_code"])
    teks_df["module_code"] = teks_df["module_code"].astype(str).str.strip()
    teks_df["expectation"] = teks_df["expectation"].astype(str)
    if teks_df.empty or teks_df["expectation"].str.strip().eq("").any():
        raise ValueError("Expectation catalog is empty or contains blank expectations.")

    if args.n_rows > 0:
        data = data[:args.n_rows]
    if not data:
        print("No rows found.")
        return

    required_cols = {
        "id", args.query_row, "expectation", "pretend_age",
        "blocked_cats", "refusal_gt",
    }
    for i, row in enumerate(data, start=1):
        missing = required_cols - row.keys()
        if missing:
            raise ValueError(f"Input row {i} missing columns: {missing}")
        if not str(row[args.query_row]).strip():
            raise ValueError(f"Input row {i} has an empty {args.query_row}")

    dataset_ids = [row["id"] for row in data]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("Duplicate IDs detected.")

    safe_model_name = args.model.replace("/", "--")
    safe_embedding_name = args.embedding_model.replace("/", "--")
    # Separate run directories protect baseline and different Top-K experiments.
    run_dir = (
        Path(OUTPUT_DIR) / safe_model_name / args.query_row / args.retrieval_corpus
        / f"top{args.n_r_contexts}_{safe_embedding_name}"
        / Path(args.input_data_path).stem
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(run_dir / "outputs.jsonl")
    eval_path = str(run_dir / "eval.json")
    save_args_as_config_json(args, output_path)

    existing_outputs = load_existing_outputs(output_path)
    selected_ids = set(dataset_ids)
    completed_outputs = {
        row_id: row for row_id, row in existing_outputs.items()
        if row_id in selected_ids
    }
    remaining_data = [
        row for row in data if row["id"] not in completed_outputs
    ]

    print(f"Query type:        {args.query_row}")
    print(f"Loaded:            {len(data)}")
    print(f"Retrieval corpus:  {args.retrieval_corpus}")
    print(f"Catalog size:      {len(teks_df)}")
    print(f"Retrieval Top-K:   {min(args.n_r_contexts, len(teks_df))}")
    print(f"Embedding model:   {args.embedding_model}")
    print(f"Already completed: {len(completed_outputs)}")
    print(f"Remaining:         {len(remaining_data)}")
    print(f"Save every:        {args.save_every}")

    # Avoid re-encoding the entire corpus for each query or batch.
    retriever = (
        ExpectationRetriever(teks_df, args.embedding_model)
        if remaining_data else None
    )
    example_printed = False

    for batch_start in range(0, len(remaining_data), args.save_every):
        batch_data = remaining_data[batch_start:batch_start + args.save_every]
        retrieved_dfs = retriever.retrieve_batch(  # type: ignore[union-attr]
            [row[args.query_row] for row in batch_data],
            args.n_r_contexts,
        )
        messages = build_messages(
            batch_data, retrieved_dfs, prompt_template, args.query_row,
        )

        if not example_printed:
            print("\nFirst query's retrieved expectations:")
            for rank, (_, exp) in enumerate(retrieved_dfs[0].head(5).iterrows(), 1):
                print(f"{rank}. ({exp['similarity']:.3f}) {exp['expectation']}")
            print("\nExample message:")
            print(messages[0])
            example_printed = True

        generations = generate(
            messages=messages,
            model=args.model,
            backend=args.backend,
            vllm_base_url=f"http://127.0.0.1:{args.port}/v1",
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
        )
        if len(generations) != len(batch_data):
            raise ValueError("Number of generations does not match number of input rows.")

        batch_output = []
        for row, generation, retrieved_df in zip(
            batch_data, generations, retrieved_dfs, strict=True,
        ):
            evaluation = evaluate_generation(row, generation)
            # Save retrieval metadata for auditing without changing evaluation.
            retrieval = [
                {
                    "code": str(exp.get("code", "")),
                    "expectation": str(exp["expectation"]),
                    "module_code": str(exp["module_code"]),
                    "grade_code": int(exp["grade_code"]),
                    "similarity": float(exp["similarity"]),
                }
                for _, exp in retrieved_df.iterrows()
            ]
            batch_output.append({**row, **evaluation, "retrieved_expectations": retrieval})

        append_jsonl_file(output_path, batch_output)
        for row in batch_output:
            completed_outputs[row["id"]] = row

        current_output = [
            completed_outputs[row["id"]]
            for row in data if row["id"] in completed_outputs
        ]
        metrics = evaluate_dataset(current_output)
        save_evaluation(metrics, eval_path)
        print(f"Saved {len(current_output)} / {len(data)} rows.")

    final_output = [
        completed_outputs[row["id"]]
        for row in data if row["id"] in completed_outputs
    ]
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
