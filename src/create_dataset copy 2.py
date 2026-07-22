"""
Create a synthetic child Q/A dataset from HECAT standards.

Example usage:

    # OpenAI
    python src/create_dataset.py \
        --backend openai \
        --model gpt-4o-mini \
        --sample_size 30 \
        --temperature 1.0 \
        --max_tokens 512

    # vLLM offline inference
    VLLM_LOGGING_LEVEL=WARNING CUDA_VISIBLE_DEVICES=1 python src/create_dataset.py \
        --backend vllm \
        --model Qwen/Qwen2.5-7B-Instruct \
        --sample_size 100 \
        --temperature 0.5 \
        --max_tokens 128
"""

# Standard library imports
import argparse
import os
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from utils.file_io import load_yaml_prompt
from utils.prompters import (
    OpenAIPrompter,
    VLLMPrompter,
    BasePrompter,
    ChatPrompt,
)
from utils.schemas import (
    GeneratedQuestion,
    GeneratedAnswer,
)

PROMPT_DIR = "./data/prompts/dataset_curration"

PROMPT_MAP = {
    "AOD": "Q_AOD.yaml",
    "FN": "Q_FN.yaml",
    "MEH": "Q_MEH.yaml",
    "PA": "Q_PA.yaml",
    "PHW": "Q_PHW.yaml",
    "S": "Q_S.yaml",
    "SH": "Q_SH.yaml",
    "T": "Q_T.yaml",
    "V": "Q_V.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a synthetic child Q/A dataset using an LLM."
    )

    # Backend/model args
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openai", "vllm"],
        required=True,
        help="Generation backend.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name, e.g. gpt-4o-mini or Qwen/Qwen2.5-7B-Instruct.",
    )

    # Data args
    parser.add_argument(
        "--standards_path",
        type=str,
        default="./data/cleaned/hecat_standards.tsv",
        help="Path to cleaned HECAT standards TSV.",
    )

    parser.add_argument(
        "--answer_prompt_path",
        type=str,
        default="./data/prompts/dataset_curration/generate_answer.yaml",
        help="Path to YAML prompt for answer generation.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/generated",
        help="Directory where generated dataset will be saved.",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=30,
        help="Number of standards to sample. Use -1 for all rows.",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for sampling standards.",
    )

    # Generation args
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum generated tokens.",
    )

    # vLLM-only args
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling value. Used only for vLLM.",
    )

    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size. Used only for vLLM.",
    )

    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization. Used only for vLLM.",
    )

    return parser.parse_args()


def load_prompt_templates() -> dict[str, dict[str, str]]:
    """
    Load one YAML prompt template per HECAT module.
    """

    templates = {}

    for module_code, filename in PROMPT_MAP.items():
        path = os.path.join(PROMPT_DIR, filename)

        if not os.path.exists(path):
            print(f"WARNING: Prompt file not found: {path}")
            continue

        templates[module_code] = load_yaml_prompt(path)

    print(f"Loaded {len(templates)} module prompt templates.")

    return templates


def load_standards(
    standards_path: str,
    sample_size: int,
    random_state: int,
) -> list[dict[str, Any]]:
    """
    Load HECAT standards and optionally sample rows.
    """

    df_standards = pd.read_csv(standards_path, sep="\t")

    print(
        f"Loaded {len(df_standards)} HECAT standards "
        f"from {standards_path}."
    )

    if sample_size == -1:
        df_sample = df_standards
        print("Using all standards.")
    else:
        if sample_size > len(df_standards):
            raise ValueError(
                f"sample_size={sample_size} is larger than dataset size "
                f"({len(df_standards)}). Use -1 for all rows."
            )

        df_sample = df_standards.sample(
            n=sample_size,
            random_state=random_state,
        )

        print(f"Sampled {len(df_sample)} standards.")

    return df_sample.to_dict(orient="records")


def build_format_args(row: dict[str, Any]) -> dict[str, Any]:
    """
    Build formatting arguments used by YAML prompt templates.
    """

    return {
        "grade_range": row["grade_span"],
        "grade_span": row["grade_span"],
        "module_code": row["module_code"],
        "health_category": row["module"],
        "education_standard": row["expectation"],
        "scenario_type": row.get("scenario_type", ""),
        "situation": row.get("situation", ""),
        "generated_question": row.get("generated_question", ""),
    }


def construct_chat_prompts(
    data: list[dict[str, Any]],
    prompt_templates: dict[str, dict[str, str]],
    prompt_type: str,
) -> list[dict[str, Any]]:
    """
    Add ChatPrompt objects to each row.

    prompt_type:
        "q" for question generation
        "a" for answer generation
    """

    if prompt_type not in ("q", "a"):
        raise ValueError("prompt_type must be either 'q' or 'a'.")

    target_key = f"{prompt_type}_chat_prompt"

    for row in data:
        fmt_args = build_format_args(row)

        module_code = row["module_code"]

        if module_code not in prompt_templates:
            raise ValueError(
                f"No prompt template found for module_code='{module_code}'."
            )

        prompt_template = prompt_templates[module_code]

        row[target_key] = ChatPrompt(
            system_text=prompt_template["system_prompt"].format(
                **fmt_args
            ),
            user_text=prompt_template["user_prompt"].format(
                **fmt_args
            ),
        )

    return data


def generate_dataset_component(
    data: list[dict[str, Any]],
    prompt_templates: dict[str, dict[str, str]],
    prompt_type: str,
    output_key: str,
    schema: Any,
    prompter: BasePrompter,
) -> list[dict[str, Any]]:
    """
    Generate either questions or answers.
    """

    component_name = "questions" if prompt_type == "q" else "answers"

    data = construct_chat_prompts(
        data=data,
        prompt_templates=prompt_templates,
        prompt_type=prompt_type,
    )

    messages = [
        row[f"{prompt_type}_chat_prompt"].to_messages()
        for row in data
    ]

    print(f"Generating {component_name}...")

    results = prompter.generate_structured(
        prompts=messages,
        schema=schema,
    )

    for row, result in zip(data, results):

        if prompt_type == "q":
            row["scenario_type"] = result.scenario_type
            row["situation"] = result.situation

        row[output_key] = result.response

    print(f"Finished generating {component_name}.")

    return data


def build_prompter(args: argparse.Namespace) -> BasePrompter:
    """
    Build the correct prompter for the selected backend.
    """

    if args.backend == "openai":
        print(f"Using OpenAI model: {args.model}")

        return OpenAIPrompter(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    if args.backend == "vllm":
        print(f"Using local vLLM model: {args.model}")

        return VLLMPrompter(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    raise ValueError(f"Unknown backend: {args.backend}")


def make_output_path(
    output_dir: str,
    model: str,
    sample_size: int,
) -> str:
    """
    Build a safe output path from the model name.
    """

    os.makedirs(output_dir, exist_ok=True)

    safe_model_name = model.replace("/", "__")

    sample_label = "all" if sample_size == -1 else str(sample_size)

    filename = (
        f"{safe_model_name}"
        f"__sample_{sample_label}"
        f"__synthetic_child_qa_dataset.tsv"
    )

    return os.path.join(output_dir, filename)


def save_dataset(
    data: list[dict[str, Any]],
    output_path: str,
) -> None:
    """
    Save generated dataset as TSV.
    """

    df_output = pd.DataFrame(data)

    df_output = df_output.drop(
        columns=[
            "q_chat_prompt",
            "a_chat_prompt",
        ],
        errors="ignore",
    )

    df_output.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(f"Saved generated dataset to {output_path}.")


def main() -> None:
    args = parse_args()

    data = load_standards(
        standards_path=args.standards_path,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )

    q_templates = load_prompt_templates()

    print("Loaded question prompt templates.")

    answer_template = load_yaml_prompt(
        args.answer_prompt_path
    )

    a_templates = {
        module_code: answer_template
        for module_code in PROMPT_MAP
    }

    print("Loaded answer prompt template.")

    prompter = build_prompter(args)

    data = generate_dataset_component(
        data=data,
        prompt_templates=q_templates,
        prompt_type="q",
        output_key="generated_question",
        schema=GeneratedQuestion,
        prompter=prompter,
    )

    data = generate_dataset_component(
        data=data,
        prompt_templates=a_templates,
        prompt_type="a",
        output_key="generated_answer",
        schema=GeneratedAnswer,
        prompter=prompter,
    )

    output_path = make_output_path(
        output_dir=args.output_dir,
        model=args.model,
        sample_size=args.sample_size,
    )

    save_dataset(
        data=data,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()