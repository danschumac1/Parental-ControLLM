"""

kill 56704
kill 65913

2026-09-17

How to run with OpenAI
    Run prompting normally:
        python ./src/guardrail_prompting.py \
            --input_data_path ./data/generated/combined_dataset.tsv \
            --backend openai \
            --model gpt-5.4-nano \
            --save_every 25

    Run in background:
        nohup python ./src/guardrail_prompting.py \
            --input_data_path ./data/generated/combined_dataset.tsv \
            --backend openai \
            --model gpt-5.4-nano \
            --save_every 25 \
            > guardrail_prompting.log 2>&1 &

    TO watch log:
        tail -f guardrail_prompting.log

        

How to run with vLLM:
    # right now focus on these models
        # SMALLER
        ✓ google/gemma-4-12B-it
        ✓ microsoft/phi-4


    Setup vLLM server:

        or # 88652
        nohup env \
            VLLM_USE_FLASHINFER_SAMPLER=0 \
            vllm serve google/gemma-4-12B-it \
                --host 127.0.0.1 \
                --port 8002 \
            > vllm_server.log 2>&1 &

    
    Run prompting normally:

        python ./src/guardrail_prompting.py \
            --input_data_path ./data/generated/combined_dataset.tsv \
            --backend vllm \
            --model google/gemma-4-12B-it \
            --save_every 25


    Run in background:

        nohup python ./src/guardrail_prompting.py \
            --input_data_path ./data/generated/combined_dataset.tsv \
            --backend vllm \
            --model google/gemma-4-12B-it \
            --save_every 25 \
            > vllm_guardrail_prompting.log 2>&1 &


    Watch log:

        tail -f vllm_guardrail_prompting.log


This script:
    1. Loads the pre-generated guardrail configurations.
    2. Checks outputs.jsonl for IDs that were already completed.
    3. Skips completed rows when resuming.
    4. Constructs guardrail prompts in batches.
    5. Generates model responses.
    6. Detects whether each response is a refusal.
    7. Compares the predicted refusal against refusal_gt.
    8. Saves every N rows to outputs.jsonl.
    9. Recomputes and saves aggregate evaluation after every batch.
    10. Computes accuracy, precision, recall, and F1.
"""

import os
import json
import argparse

from utils.file_io import (
    append_jsonl_file,
    load_tsv_file,
    load_single_yaml_prompt,
    save_args_as_config_json,
)

from utils.prompting import (
    construct_messages,
    generate,
)

from utils.evaluation import (
    evaluate_generation,
    evaluate_dataset,
    print_evaluation,
    save_evaluation,
)


# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

VLLM_BASE_URL = "http://localhost:8002/v1"

PROMPT_PATH = "./data/prompts/response_generation/guardrail.yaml"

OUTPUT_DIR = "./data/generated/guardrail_results"


GUARDRAIL_MAP = {
    "AOD": "Alcohol and Other Drug-Use Prevention",
    "FN": "Food and Nutrition",
    "MEH": "Mental and Emotional Health",
    "PA": "Physical Activity",
    "PHW": "Physical Health and Wellness",
    "S": "Safety",
    "SH": "Sexual Health",
    "T": "Tobacco-Use Prevention",
    "V": "Violence Prevention",
}


# ---------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_data_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--query_column",
        type=str,
        default="situation",
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
        "--save_every",
        type=int,
        default=25,
        help="Save results after this many generations.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
    )

    return parser.parse_args()


# ---------------------------------------------------------
# DATA PARSING
# ---------------------------------------------------------

def parse_blocked_cats(
    value: str,
) -> list[str]:
    """
    Convert:

        AOD,FN,SH

    into:

        ["AOD", "FN", "SH"]
    """

    if not value:
        return []

    return [
        code.strip()
        for code in value.split(",")
        if code.strip()
    ]


# ---------------------------------------------------------
# GUARDRAIL FORMATTING
# ---------------------------------------------------------

def make_guardrail_string(
    blocked_cats: list[str],
) -> str:
    """
    Convert category codes into readable guardrail names.

    Example:

        ["AOD", "SH"]

    becomes:

        - Alcohol and Other Drug-Use Prevention
        - Sexual Health
    """

    if not blocked_cats:
        return "None"

    return "\n".join(
        f"- {GUARDRAIL_MAP[code]}"
        for code in blocked_cats
    )


def build_prompt_data(
    data: list[dict],
    query_column: str,
) -> list[dict]:
    """
    Add temporary fields used to construct prompts.
    """

    prompt_data = []

    for row in data:

        blocked_cats = parse_blocked_cats(
            row.get("blocked_cats", "")
        )

        guardrails = make_guardrail_string(
            blocked_cats
        )

        prompt_data.append(
            {
                **row,
                "_question": row[query_column],
                "_guardrails": guardrails,
            }
        )

    return prompt_data


# ---------------------------------------------------------
# RESUME LOGIC
# ---------------------------------------------------------

def load_existing_outputs(
    output_path: str,
) -> dict[str, dict]:
    """
    Load previously completed rows from outputs.jsonl.

    Returns a dictionary keyed by row ID.

    Example:

        {
            "google__gemma...__000001__all_blocked": {...},
            "google__gemma...__000001__all_allowed": {...},
        }

    If outputs.jsonl does not exist yet, return an empty dict.
    """

    if not os.path.exists(output_path):
        return {}

    existing_outputs = {}

    with open(
        output_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {output_path} "
                    f"on line {line_number}"
                ) from e

            if "id" not in row:
                raise ValueError(
                    f"Existing output row on line "
                    f"{line_number} has no ID."
                )

            existing_outputs[row["id"]] = row

    return existing_outputs


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    args = parse_args()

    # -----------------------------------------------------
    # Validate arguments
    # -----------------------------------------------------

    if args.save_every <= 0:
        raise ValueError(
            "--save_every must be greater than 0."
        )

    # -----------------------------------------------------
    # Build output paths
    # -----------------------------------------------------

    safe_model_name = args.model.replace(
        "/",
        "--",
    )

    run_dir = os.path.join(
        OUTPUT_DIR,
        safe_model_name,
    )

    os.makedirs(
        run_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        run_dir,
        "outputs.jsonl",
    )

    eval_path = os.path.join(
        run_dir,
        "eval.json",
    )

    # -----------------------------------------------------
    # Save run configuration
    # -----------------------------------------------------

    save_args_as_config_json(
        args,
        output_path,
    )

    # -----------------------------------------------------
    # Load prompt
    # -----------------------------------------------------

    prompt_template = load_single_yaml_prompt(
        PROMPT_PATH
    )

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    data = load_tsv_file(
        args.input_data_path
    )

    if args.n_rows > 0:
        data = data[:args.n_rows]

    print()
    print(f"Loaded {len(data)} rows.")

    if not data:
        print("No rows found.")
        return

    # -----------------------------------------------------
    # Verify IDs
    # -----------------------------------------------------

    for row_number, row in enumerate(
        data,
        start=1,
    ):

        if "id" not in row:
            raise ValueError(
                f"Input row {row_number} has no 'id' column. "
                f"Recreate the dataset with create_dataset.py."
            )

    dataset_ids = [
        row["id"]
        for row in data
    ]

    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError(
            "Duplicate IDs detected in input dataset."
        )

    # -----------------------------------------------------
    # Load previous results
    # -----------------------------------------------------

    existing_outputs = load_existing_outputs(
        output_path
    )

    selected_ids = set(dataset_ids)

    # Only consider previous results that belong to the
    # currently selected dataset rows.
    completed_outputs = {
        row_id: row
        for row_id, row in existing_outputs.items()
        if row_id in selected_ids
    }

    completed_ids = set(
        completed_outputs.keys()
    )

    # -----------------------------------------------------
    # Remove completed rows
    # -----------------------------------------------------

    remaining_data = [
        row
        for row in data
        if row["id"] not in completed_ids
    ]

    print(
        f"Already completed: {len(completed_ids)}"
    )

    print(
        f"Remaining:         {len(remaining_data)}"
    )

    print(
        f"Save every:        {args.save_every} rows"
    )

    # -----------------------------------------------------
    # Generate in batches
    # -----------------------------------------------------

    example_printed = False

    for batch_start in range(
        0,
        len(remaining_data),
        args.save_every,
    ):

        batch_data = remaining_data[
            batch_start:
            batch_start + args.save_every
        ]

        # -------------------------------------------------
        # Build prompts for this batch
        # -------------------------------------------------

        prompt_data = build_prompt_data(
            batch_data,
            args.query_column,
        )

        messages = construct_messages(
            prompt_data,
            prompt_template,
            sys_map={
                "guardrails": "_guardrails",
            },
            user_map={
                "question": "_question",
            },
        )

        # Print one example prompt only.
        if not example_printed:

            print()
            print("Example message:")
            print(messages[0])

            example_printed = True

        # -------------------------------------------------
        # Generate this batch
        # -------------------------------------------------

        generations = generate(
            messages=messages,
            model=args.model,
            backend=args.backend,
            vllm_base_url=VLLM_BASE_URL,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
        )

        if len(generations) != len(batch_data):
            raise ValueError(
                "Number of generations does not match "
                "number of input rows."
            )

        # -------------------------------------------------
        # Evaluate this batch
        # -------------------------------------------------

        batch_output = []

        for row, generation in zip(
            batch_data,
            generations,
        ):

            evaluation = evaluate_generation(
                row,
                generation,
            )

            output_row = {
                **row,
                **evaluation,
            }

            batch_output.append(
                output_row
            )

        # -------------------------------------------------
        # SAVE THIS BATCH IMMEDIATELY
        # -------------------------------------------------

        append_jsonl_file(
            output_path,
            batch_output,
        )

        # Keep an in-memory copy keyed by ID.
        for output_row in batch_output:
            completed_outputs[
                output_row["id"]
            ] = output_row

        # -------------------------------------------------
        # Recompute current evaluation
        # -------------------------------------------------

        current_output_data = [
            completed_outputs[row["id"]]
            for row in data
            if row["id"] in completed_outputs
        ]

        metrics = evaluate_dataset(
            current_output_data
        )

        save_evaluation(
            metrics,
            eval_path,
        )

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        completed_count = len(
            current_output_data
        )

        print()
        print(
            f"Saved {completed_count} / "
            f"{len(data)} rows."
        )

    # -----------------------------------------------------
    # Final evaluation
    # -----------------------------------------------------

    final_output_data = [
        completed_outputs[row["id"]]
        for row in data
        if row["id"] in completed_outputs
    ]

    metrics = evaluate_dataset(
        final_output_data
    )

    print_evaluation(
        metrics
    )

    save_evaluation(
        metrics,
        eval_path,
    )

    # -----------------------------------------------------
    # Done
    # -----------------------------------------------------

    print()
    print("############################")
    print("FILES SAVED")
    print("############################")
    print()

    print(f"Run directory: {run_dir}")
    print(f"Outputs:       {output_path}")
    print(f"Evaluation:    {eval_path}")
    print(
        "Config:        "
        f"{output_path.replace('.jsonl', '_config.json')}"
    )

    print()
    print(
        f"Completed:     "
        f"{len(final_output_data)} / {len(data)}"
    )


if __name__ == "__main__":
    main()