# src/utils/llm_as_a_judge.py

"""
python ./src/llm_as_a_judge.py \
    --dataset child_safe \
    --model gpt-4o-mini \
    --cuda_device 1 \
    --port 8000 \
    --log_file ./vllm_server__llm_as_judge.log
"""
# standard args
import os
import argparse
import json

# pip installs 
import yaml # pyaml

# user defined
from utils.file_io import load_yaml_prompts, load_jsonl_file
from utils.vllm_utils import VllmServer, VLLMPrompter, ChatPrompt
from utils.schemas import LikertScore

# CONSTANTS
PROMPT_TYPES = [
    "appropriateness",
    "constructive_guidance",
    "information_value",
    "risk_reduction",
    "refusal"
]

PROMPT_PATH = "./data/prompts/llm_as_judge/{PROMPT}.yaml"

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM as a judge for evaluation.")
    parser.add_argument("--dataset", type=str, required=True, choices=["child_safe", "minor_bench"], help="Dataset name (e.g., 'topic1').")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'gpt-4').")
    parser.add_argument("--cuda_device", type=str, default="0", help="CUDA device ID (default: 0).")
    parser.add_argument("--port", type=int, default=8000, help="Port for VLLM server (default: 8000).")
    parser.add_argument("--log_file", type=str, default="./vllm_server__llm_as_judge.log", help="Log file path for VLLM server.")
    return parser.parse_args()

def run_llm_as_a_judge(
    dataset: str,
    model: str,
    cuda_device: str,
    port: int,
    log_file: str,
) -> dict:
    """Core callable API used by eval.py or CLI."""
    
    # Load data
    jsonl_data = load_jsonl_file(f"./data/cleaned/{dataset}.jsonl")
    print(f"Loaded {len(jsonl_data)} entries from dataset '{dataset}'.")

    # Load prompt templates
    prompts = load_yaml_prompts(
        PROMPT_TYPES,
        "./data/prompts/llm_as_judge",
    )
    print(f"Loaded prompt specifications for: {', '.join(prompts.keys())}.")

    # Construct ChatPrompt objects for every entry and judge type
    for entry in jsonl_data:
        query = entry["prompt"]
        response = entry["response"]

        for prompt_type, template in prompts.items():
            cp = ChatPrompt(
                system_text=template["system_prompt"],
                user_text=template["user_prompt"].format(
                    query=query,
                    response=response,
                ),
            )

            entry[prompt_type] = cp

    # Example
    print(jsonl_data[0]["appropriateness"])

    # print(f"Constructed chat prompts for all entries in dataset '{dataset}'.")
    print(f"{jsonl_data[0]['appropriateness']}")
    exit()

    # print(f"Connecting to VLLM server with model {model} on CUDA device {cuda_device} at port {port}...")
    # with VllmServer(
    #     model=model,
    #     cuda_device=cuda_device,
    #     port=port,
    #     log_file=log_file,
    # ):
    #     print(f"Started VLLM server on port {port} with model {model}.")
    #     prompter = VLLMPrompter(
    #         base_url=f"http://localhost:{port}/v1",
    #         model=model,
    #         temperature=0.0,
    #         max_tokens=512,
    #     )
    #     print("Initialized VLLM prompter client.")

    #     results = prompter.generate_structured(
    #         prompts=messages,
    #         schema=LikertScore,
    #     )
    #     print(f"Received structured responses from LLM for all prompts.")

    # return {
    #     name: {
    #         "score": r.score,
    #         "justification": r.justification,
    #     }
    #     for name, r in zip(PROMPT_TYPES, results)
    # }

def main():
    args = parse_args()
    results = run_llm_as_a_judge(
        dataset=args.dataset,
        model=args.model,
        cuda_device=args.cuda_device,
        port=args.port,
        log_file=args.log_file,
    )
    print(f"Finished")
    # print(f"Final results:\n{json.dumps(results, indent=2)}")

if __name__ == "__main__":
    main()