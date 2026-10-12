"""
2026-09-04

Generate realistic situations from HECAT education standards.

Examples:

    # Instruct model through OpenAI
    python ./src/prompt_for_situation.py \
        --backend openai \
        --model gpt-5.4-nano

    ######################################
    # VLLM
    ######################################
        # SMALLER
        ✓ Qwen/Qwen3.5-9B
        ✓ google/gemma-4-12B-it
        ✓ ibm-granite/granite-4.2-8b
        ✓ microsoft/phi-4


    # launch the server
    #  may need VLLM_USE_FLASHINFER_SAMPLER=0 
    VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve ibm-granite/granite-4.2-8b \
    --host 127.0.0.1 \
    --port 8002


    # Instruct model through vLLM
    
    python ./src/prompt_for_situation.py \
    --backend vllm \
    --model ibm-granite/granite-4.2-8b\
    --vllm_base_url http://localhost:8002/v1

    # Non-instruct/base model through vLLM
    python ./src/prompt_for_situation.py \
        --backend vllm \
        --prompt_type non_instruct \
        --model meta-llama/Llama-3.1-8B \
        --vllm_base_url http://localhost:8000/v1
"""

import argparse
import json
import os


from openai import OpenAI
from dotenv import load_dotenv

from utils.file_io import load_tsv_file, append_jsonl_file, load_instruct_yaml, load_non_instruct_yaml

DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_TOKENS = 100
DEFAULT_TOP_P = 0.95
DEFAULT_INSTRUCT_PROMPT = "./data/prompts/dataset_curation/instruct.yaml"
DEFAULT_NON_INSTRUCT_PROMPT = "./data/prompts/dataset_curation/non_instruct.yaml"

# ───────────────────────── Arguments ─────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate situations or educational questions from HECAT standards.")
    parser.add_argument("--input_path", default="data/cleaned/hecat_standards.tsv")
    parser.add_argument("--generation_type", choices=["situation_generation", "eduQ_generation"], default="situation_generation")
    parser.add_argument("--prompt_type", choices=["instruct", "non_instruct"], default="instruct")
    parser.add_argument("--backend", choices=["openai", "vllm"], default="vllm")
    parser.add_argument("--model", required=True)
    parser.add_argument("--vllm_base_url", default="http://localhost:8000/v1")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    safe_model_name = args.model.replace("/", "__")
    args.output_path = f"./data/generated/{args.generation_type}/{safe_model_name}.jsonl"
    args.prompt_folder = f"./data/prompts/{args.generation_type}"
    args.prompt_path = f"{args.prompt_folder}/{args.prompt_type}.yaml"
    args.output_column = args.generation_type.split("_")[0]

    if not os.path.isfile(args.prompt_path):
        parser.error(f"Prompt file does not exist: {args.prompt_path}")

    return args


def load_existing_outputs(path: str) -> dict[str, dict]:
    """
    Load existing generations keyed by standard code.
    """

    if not os.path.exists(path):
        return {}

    outputs = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            row = json.loads(line)

            code = row.get("code")
            if not code:
                raise ValueError(
                    f"Missing code at line {line_number}: {path}"
                )

            if code in outputs:
                raise ValueError(
                    f"Duplicate code {code} in {path}"
                )

            outputs[code] = row

    return outputs


from pathlib import Path
import tempfile


def save_outputs(path: str, outputs: dict[str, dict]) -> None:
    """
    Atomically rewrite the JSONL file, preserving existing
    columns and adding new generations.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = Path(f.name)

            for row in outputs.values():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        os.replace(temp_path, output_path)

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


# ───────────────────────── Prompting ─────────────────────────
def format_instruct_prompt(prompt_data: list[dict], row: dict) -> list[dict]:
    """
    Replace placeholders in each chat message.
    """

    education_standard = row["expectation"]
    grade_range = row["grade_span"]

    messages = []

    for message in prompt_data:
        content = message["content"].format(
            education_standard=education_standard,
            grade_range=grade_range,
        )

        messages.append(
            {
                "role": message["role"],
                "content": content,
            }
        )

    return messages


def format_non_instruct_prompt(prompt_data: str, row: dict) -> str:
    education_standard = row["expectation"]
    grade_range = row.get("grade_span", "")

    return prompt_data.format(
        education_standard=education_standard,
        grade_range=grade_range,
    )


def clean_generation(text: str) -> str:
    """
    Do minimal cleanup without altering the generated content.
    """

    text = text.strip()

    # A non-instruct model may continue into the next example.
    if "#####" in text:
        text = text.split("#####", 1)[0].strip()

    # It may also begin generating the next education standard.
    if "\nEducation Standard:" in text:
        text = text.split(
            "\nEducation Standard:",
            1,
        )[0].strip()

    return text


# ──────────────────────── LLM Clients ────────────────────────


def create_client(args: argparse.Namespace) -> OpenAI:
    if args.backend == "openai":
        load_dotenv("./resources/.env")
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")

        )

    if args.backend == "vllm":
        return OpenAI(
            base_url=args.vllm_base_url,
            api_key="EMPTY",
        )

    raise ValueError(
        f"Unknown backend: {args.backend}"
    )


def generate_instruct(
    backend: str,
    client: OpenAI,
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    top_p: float,
)-> str:
    
    assert backend in ["vllm","openai"], "backend must be either vllm or openai"
    if backend == "vllm":
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    else: # backend == "openai"
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens, # thius has to be different!
            top_p=top_p,
        )

    return response.choices[0].message.content.strip()


def generate_non_instruct(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> str:
    response = client.completions.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=["#####"],
    )

    return response.choices[0].text.strip()


# ─────────────────────────── Main ─────────────────────────────



def main():
    args = parse_args()

    print("Loading data...")
    rows = load_tsv_file(args.input_path)
    original_count = len(rows)
    rows = list({(row["code"], row["expectation"]): row for row in reversed(rows)}.values())
    rows.reverse()
    print(f"Removed {original_count - len(rows)} identical standards.")    

    if args.prompt_type == "instruct":
        prompt_data = load_instruct_yaml(args.prompt_path)
    else:
        prompt_data = load_non_instruct_yaml(args.prompt_path)

    # Load previous generations for this model.
    existing_outputs = load_existing_outputs(args.output_path)

    print(f"Loaded {len(existing_outputs)} existing output rows.")

    # Only skip standards that already have the requested column.
    remaining_rows = []

    for row in rows:
        code = row["code"]
        existing = existing_outputs.get(code, {})

        value = existing.get(args.output_column)

        if isinstance(value, str) and value.strip():
            continue

        remaining_rows.append(row)

    if args.limit is not None:
        remaining_rows = remaining_rows[:args.limit]

    print(
        f"{len(remaining_rows)} standards remaining "
        f"for {args.output_column}."
    )

    if not remaining_rows:
        print("Nothing to generate.")
        return

    client = create_client(args)

    for i, row in enumerate(remaining_rows, start=1):
        code = row["code"]
        expectation = row.get("expectation", "")

        print()
        print(f"[{i}/{len(remaining_rows)}] {code}")
        print(f"Standard: {expectation}")

        try:
            if args.prompt_type == "instruct":
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

            else:
                prompt = format_non_instruct_prompt(
                    prompt_data,
                    row,
                )

                generation = generate_non_instruct(
                    client=client,
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                )

            generation = clean_generation(generation)

            if not generation:
                print(f"WARNING: Empty generation for {code}.")
                continue

            print(f"{args.output_column}: {generation}")

            # Preserve existing generations and metadata.
            existing_row = existing_outputs.get(code, {})

            output_row = {
                **row,
                **existing_row,
                args.output_column: generation,
            }

            existing_outputs[code] = output_row

            # Save after each successful generation.
            save_outputs(args.output_path, existing_outputs)

        except Exception as e:
            print(f"ERROR on {code}: {e}")

    print()
    print("Done.")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()