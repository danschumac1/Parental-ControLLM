# src/utils/vanilla_prompt.py

"""
python ./src/vanilla_prompt.py \
    --dataset child_safe \
    --model gpt-4o-mini \
    --cuda_device 1 \
    --port 8000 \
    --log_file ./vllm_server__vanilla_prompt.log
"""
# standard args
import os
import argparse
import json

# pip installs 
import yaml # pyaml

# user defined
from utils.file_io import load_jsonl_file, save_jsonl_file
from utils.vllm_utils import VllmServer, VLLMPrompter, ChatPrompt
from utils.schemas import FreeResponse

# CONSTANTS
PROMPT_PATH = "./data/prompts/llm_as_judge/{PROMPT}.yaml"

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM as a judge for evaluation.")
    parser.add_argument("--dataset", type=str, required=True, choices=["child_safe", "minor_bench"], help="Dataset name (e.g., 'topic1').")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'gpt-4').")
    parser.add_argument("--cuda_device", type=str, default="0", help="CUDA device ID (default: 0).")
    parser.add_argument("--port", type=int, default=8000, help="Port for VLLM server (default: 8000).")
    parser.add_argument("--log_file", type=str, default="./vllm_server__llm_as_judge.log", help="Log file path for VLLM server.")
    return parser.parse_args()

def run_inference(
    dataset: str,
    model: str,
    cuda_device: str,
    port: int,
    log_file: str,
) -> dict:
    """Core callable API used by eval.py or CLI."""
    
    # Load data
    jsonl_data = load_jsonl_file(f"./data/cleaned/{dataset}.jsonl")

    # TODO: REMOVE
    jsonl_data = jsonl_data[:10]

    print(f"Loaded {len(jsonl_data)} entries from dataset '{dataset}'.")


    # Construct ChatPrompt objects for every entry and judge type
    for entry in jsonl_data:
        query = entry["prompt"]
        formatted_prompt = ChatPrompt(
            system_text="You are a helpful assistant.",
            user_text=query
            )
        entry["formatted_prompt"] = formatted_prompt

    # example prompt
    print(f"Example: {jsonl_data[0]['formatted_prompt']}")

    print(f"Connecting to VLLM server with model {model} on CUDA device {cuda_device} at port {port}...")
    with VllmServer(
        model=model,
        cuda_device=cuda_device,
        port=port,
        log_file=log_file,
    ):
        print(f"Started VLLM server on port {port} with model {model}.")
        prompter = VLLMPrompter(
            base_url=f"http://localhost:{port}/v1",
            model=model,
            temperature=0.0,
            max_tokens=512,
        )
        print("Initialized VLLM prompter client.")

        results = prompter.generate_structured(
            prompts=[entry["formatted_prompt"].to_messages() for entry in jsonl_data],
            schema=FreeResponse,
        )
        print(f"Received structured responses from LLM for all prompts.")

        # stitch results back into jsonl_data and drop the formatted_prompt field
        for entry, result in zip(jsonl_data, results):
            entry["llm_response"] = result.response
            del entry["formatted_prompt"]

    return jsonl_data



def main():
    args = parse_args()
    results = run_inference(
        dataset=args.dataset,
        model=args.model,
        cuda_device=args.cuda_device,
        port=args.port,
        log_file=args.log_file,
    )

    # Save results to a new JSONL file
    output_file_path = f"./data/output/vanilla/{args.dataset}/{args.model}_responses.jsonl"

    save_jsonl_file(output_file_path, results)  
    print(f"Finished")
    


if __name__ == "__main__":
    main()