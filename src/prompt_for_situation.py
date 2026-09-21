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
    parser = argparse.ArgumentParser(
        description="Generate situations from HECAT education standards."
    )

    parser.add_argument("--input_path",default="data/cleaned/hecat_standards.tsv",help="Path to HECAT standards TSV.",)
    parser.add_argument("--backend",choices=["openai", "vllm"],default="vllm",help="LLM backend.",)
    parser.add_argument("--model",required=True,help="Model name.",)
    parser.add_argument("--vllm_base_url",default="http://localhost:8000/v1",help="OpenAI-compatible vLLM endpoint.",)
    parser.add_argument("--temperature",type=float,default=DEFAULT_TEMPERATURE,)
    parser.add_argument("--max_tokens",type=int,default=DEFAULT_MAX_TOKENS,)
    parser.add_argument("--top_p",type=float,default=DEFAULT_TOP_P,)
    parser.add_argument("--limit",type=int,default=None,help="Optional number of rows to process.",)
    args = parser.parse_args()
    safe_model_name = args.model.replace("/", "__")
    args.output_path = f"./data/generated/situations/{safe_model_name}.jsonl"
    args.prompt_type = "instruct" # if ("instruct" in args.model.lower().strip()) or ("gpt" in args.model.lower().strip())\
            # else "non_instruct"
    return args


def load_completed_codes(path: str) -> set[str]:
    """
    Read existing output so interrupted runs can resume.
    """

    if not os.path.exists(path):
        return set()

    completed = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            if row.get("code"):
                completed.add(row["code"])

    return completed


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

    # Choose prompt file automatically if one wasn't supplied.
    if args.prompt_type == "instruct":
        args.prompt_path = DEFAULT_INSTRUCT_PROMPT
    else:
        args.prompt_path = DEFAULT_NON_INSTRUCT_PROMPT

    print("Loading data...")
    rows = load_tsv_file(args.input_path)

    print(f"Loaded {len(rows)} standards.")

    if args.prompt_type == "instruct":
        prompt_data = load_instruct_yaml(args.prompt_path)
    else:
        prompt_data = load_non_instruct_yaml(args.prompt_path)

    completed_codes = load_completed_codes(
        args.output_path
    )

    if completed_codes:
        print(f"Found {len(completed_codes)} previously completed standards.")

    rows = [ row for row in rows if row.get("code") not in completed_codes ]

    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"{len(rows)} standards remaining.")

    client = create_client(args)

    # ───────────────── Process standards ─────────────────

    for i, row in enumerate(rows, start=1):

        code = row.get("code", "UNKNOWN")
        expectation = row.get("expectation", "")

        print()
        print(
            f"[{i}/{len(rows)}] {code}"
        )
        print(
            f"Standard: {expectation}"
        )

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

            print(f"Situation: {generation}")

            output_row = {**row, "situation": generation,}

            append_jsonl_file(args.output_path,[output_row],)

        except Exception as e:
            print(f"ERROR on {code}: {e}")

    print()
    print("Done.")
    print(f"Output: {args.output_path}")
if __name__ == "__main__":
    main()