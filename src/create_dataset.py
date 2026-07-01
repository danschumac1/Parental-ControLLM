"""
python src/create_dataset.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --cuda_device 0 \
    --port 8000 \
    --log_file ./vllm_server__create_dataset.log
"""

# Standard library imports
import argparse
import os

# Pip install imports
import pandas as pd

# Local imports
from utils.file_io import load_yaml_prompt
from utils.schemas import FreeResponse
from utils.vllm_utils import ChatPrompt, VllmServer, VLLMPrompter


def parse_args():
    parser = argparse.ArgumentParser(description="Create dataset using LLM.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'gpt-4o-mini').")
    parser.add_argument("--cuda_device", type=str, default="0", help="CUDA device ID (default: '0').")
    parser.add_argument("--port", type=int, default=8000, help="Port for VLLM server (default: 8000).")
    parser.add_argument("--log_file", type=str, default="./vllm_server__create_dataset.log", help="Base log file path for VLLM server.")
    return parser.parse_args()


def construct_chat_prompts(
    data: list[dict], 
    prompt_template: dict, 
    prompt_type: str
) -> list[dict]:
    """
    Construct ChatPrompt objects for every entry in the provided data.
    """
    if prompt_type not in ("q", "a"):
        raise ValueError("prompt_type must be either 'q' or 'a'")

    target_key = f"{prompt_type}_chat_prompt"

    for row in data:
        # Build a master mapping of all possible template variables
        fmt_args = {
            "grade_range": row["grade_span"],
            "health_category": row["module"],
            "education_standard": row["expectation"],
            "generated_question": row.get("generated_question", "")
        }

        # Safely unpack kwargs. Extra keys not present in template are naturally ignored.
        cp = ChatPrompt(
            system_text=prompt_template["system_prompt"].format(**fmt_args),
            user_text=prompt_template["user_prompt"].format(**fmt_args),
        )
        
        row[target_key] = cp

    return data


def generate_dataset_component(
    data: list[dict], 
    prompt_template: dict,
    prompt_type: str,
    output_key: str,
    model: str,
    cuda_device: str,
    port: int,
    log_file: str
) -> list[dict]:
    """
    Generic function to generate components (questions or answers) using the VLLM server.
    """
    component_name = "question" if prompt_type == "q" else "answer"
    
    # 1. Construct ChatPrompt objects for every entry
    data = construct_chat_prompts(data, prompt_template, prompt_type=prompt_type)
    messages = [row[f"{prompt_type}_chat_prompt"].to_messages() for row in data]

    # 2. Run the VLLM Server context
    with VllmServer(
        model=model,
        cuda_device=cuda_device,
        port=port,
        log_file=log_file,
    ):
        print(f"Started VLLM server on port {port} with model {model} for {component_name} generation.")
        prompter = VLLMPrompter(
            base_url=f"http://localhost:{port}/v1",
            model=model,
            temperature=0.0,
            max_tokens=512,
        )
        
        results = prompter.generate_structured(
            prompts=messages,
            schema=FreeResponse,
        )
        print(f"Received structured {component_name} responses from LLM.")

    # 3. Package the results back into the data dictionary
    for row, result in zip(data, results):
        row[output_key] = result.response

    return data


def main():
    args = parse_args()

    # Load HECAT standards
    df_standards = pd.read_csv("./data/cleaned/hecat_standards.tsv", sep="\t")
    print(f"Loaded {len(df_standards)} HECAT standards.")

    df_sample = df_standards.sample(n=20, random_state=42)
    data = df_sample.to_dict(orient="records")
    print(f"Sample of {len(data)} standards selected.")

    # Load prompt templates
    gen_q_template = load_yaml_prompt("./data/prompts/dataset_curration/generate_question.yaml")
    gen_a_template = load_yaml_prompt("./data/prompts/dataset_curration/generate_answer.yaml")
    print("Loaded prompt templates for question and answer generation.")

    # Dynamically split log filenames so Step 2 doesn't purge Step 1 logs
    log_base, log_ext = os.path.splitext(args.log_file)
    q_log_file = f"{log_base}_questions{log_ext}"
    a_log_file = f"{log_base}_answers{log_ext}"

    # Step 1: Generate Questions
    data = generate_dataset_component(
        data=data,
        prompt_template=gen_q_template,
        prompt_type="q",
        output_key="generated_question",
        model=args.model,
        cuda_device=args.cuda_device,
        port=args.port,
        log_file=q_log_file
    )

    # Step 2: Generate Answers
    data = generate_dataset_component(
        data=data,
        prompt_template=gen_a_template,
        prompt_type="a",
        output_key="appropriate_answer",
        model=args.model,
        cuda_device=args.cuda_device,
        port=args.port,
        log_file=a_log_file
    )

    # Step 3: Clean up and save
    df_output = pd.DataFrame(data)
    df_output = df_output.drop(columns=["q_chat_prompt", "a_chat_prompt"], errors="ignore")
    
    # Ensure directory output exists
    os.makedirs("./data/generated", exist_ok=True)
    output_path = "./data/generated/synthetic_child_qa_dataset.tsv"
    df_output.to_csv(output_path, sep="\t", index=False)
    print(f"Successfully generated dataset and saved to {output_path}")


if __name__ == "__main__":
    main()